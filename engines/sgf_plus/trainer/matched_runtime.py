# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""Matched CF-S2 runtime around the unchanged official SGF objective.

Accumulation, monitoring, and update-counted EMA adapt Causal-Forcing's tested
trainer runtime. The native DMD, two-pass replay, and attention stay in this repo.
"""
import gc
import json
import os
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job


EXPECTED_METHOD = "sgf_plus"


def training_budget(config):
    target = int(config.target_student_updates)
    ratio = int(config.dfake_gen_update_ratio)
    warmup = int(getattr(config, "critic_warmup_steps", 0))
    if target <= 0 or ratio <= 0 or warmup != 0:
        raise ValueError("Matched SGF requires a positive Student budget/ratio and no warmup")
    return target, ratio, target * ratio


def validate_config(config, world_size=None):
    target, ratio, steps = training_budget(config)
    if config.baseline_method != EXPECTED_METHOD or config.distribution_loss != "dmd":
        raise ValueError("The matched runtime requires this repository's official SGF DMD")
    if config.grad_mode != "self_gradient_forcing":
        raise ValueError("Native self_gradient_forcing replay is required")
    if not config.per_rank_exit_step or not config.self_gradient_forcing_match_context:
        raise ValueError("Preserve native per-rank exits and matched replay context")
    if config.self_gradient_forcing_cache_mode != "exit":
        raise ValueError("Native SGF exit caches are required")
    if any(key.startswith(("ste_", "hgr_")) for key in config):
        raise ValueError("STE/HGR runtime options are not part of the official SGF baseline")
    dual = config.model_kwargs.get("dual_full_expert", {})
    if bool(dual.get("enabled", False)) != (EXPECTED_METHOD == "sgf_plus"):
        raise ValueError("Only SGF+ uses the official cloned memory expert")
    if ratio != 5 or int(config.max_steps) != steps:
        raise ValueError("max_steps must match the actual Student budget at ratio 5")
    if config.no_save or not config.save_final_only or not config.save_student_and_ema:
        raise ValueError("Matched runs must export final live Student and EMA weights")
    if not 0 < float(config.ema_weight) < 1 or not 0 <= int(config.ema_start_student_update) < target:
        raise ValueError("EMA must initialize before the final Student update")
    accumulation = int(config.gradient_accumulation_steps)
    if accumulation <= 0 or int(config.batch_size) <= 0:
        raise ValueError("Batch and accumulation must be positive")
    if world_size is not None:
        effective = world_size * int(config.batch_size) * accumulation
        if effective != 64 or int(config.total_batch_size) != effective:
            raise ValueError("Each Student/Critic update must accumulate global batch 64")
    return target, ratio, steps


class Trainer:
    def __init__(self, config):
        self.config = config
        self.target_student_updates, _, self.max_steps = validate_config(config)
        self.step = self.student_updates = self.critic_updates = self.ema_updates = 0
        self.data_batches_consumed = 0
        self.gradient_accumulation_steps = int(config.gradient_accumulation_steps)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        validate_config(config, self.world_size)
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb
        self.output_path = config.logdir
        memory_fraction = getattr(config, "cuda_memory_fraction", None)
        if memory_fraction is not None:
            if not 0 < float(memory_fraction) <= 1:
                raise ValueError("cuda_memory_fraction must be in (0, 1]")
            torch.cuda.set_per_process_memory_fraction(float(memory_fraction), self.device)
        # Warm up the finite-check collective before model/rollout memory grows.
        self._check_update_finite({"startup_loss": torch.zeros((), device=self.device)}, "Startup")
        from utils.misc import set_seed
        set_seed(config.seed + global_rank)
        if self.is_main_process and not self.disable_wandb:
            import wandb
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(config=OmegaConf.to_container(config, resolve=True),
                       name=config.config_name, mode="online", entity=config.wandb_entity,
                       project=config.wandb_project, dir=config.wandb_save_dir)
        self._init_tensorboard()

        from model import DMD
        self.model = DMD(config, device=self.device)
        self._load_student_checkpoint(config.generator_ckpt)
        self._write_model_parameters()
        for name in ("generator", "real_score", "fake_score", "text_encoder"):
            kwargs = dict(sharding_strategy=config.sharding_strategy,
                          mixed_precision=config.mixed_precision,
                          wrap_strategy=config[name + "_fsdp_wrap_strategy"],
                          cpu_offload=bool(getattr(config, "text_encoder_cpu_offload", False))
                          if name == "text_encoder" else False)
            if EXPECTED_METHOD == "sgf_plus" and name == "generator":
                kwargs["sync_module_states"] = True
            setattr(self.model, name, fsdp_wrap(getattr(self.model, name), **kwargs))
        # First-rollout mask construction and kernel autotuning can leave a
        # large idle allocator cache. Release it once before the teacher's
        # first FSDP collective so NCCL can allocate its communication buffers.
        # Live tensors and the retained computation graph are unchanged.
        def release_startup_cache(_module, _inputs):
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            startup_cache_hook.remove()

        startup_cache_hook = self.model.real_score.register_forward_pre_hook(
            release_startup_cache)
        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.model.fake_score.parameters() if p.requires_grad],
            lr=getattr(config, "lr_critic", config.lr),
            betas=(config.beta1_critic, config.beta2_critic), weight_decay=config.weight_decay)
        from utils.dataset import TextDataset, cycle
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        self.dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler,
            num_workers=int(getattr(config, "num_workers", 8))))
        self.generator_ema = None
        if int(config.ema_start_student_update) == 0:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=config.ema_weight)
        recovery = getattr(config, "resume_recovery", None)
        if recovery:
            self._restore_training_recovery(recovery)
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            config.runtime_world_size = self.world_size
            config.runtime_global_batch = self.world_size * config.batch_size * self.gradient_accumulation_steps
            config.runtime_expected_critic_updates = self.max_steps
            OmegaConf.save(config, os.path.join(self.output_path, "config.resolved.yaml"))
            self._progress("initialized", method=EXPECTED_METHOD, dataset_size=len(dataset))

    def _load_student_checkpoint(self, filename):
        # Load before FSDP: strict matching sees full ordinary tensors, including
        # the cloned writer in SGF+, without unsharding four resident models.
        state = torch.load(filename, map_location="cpu", weights_only=True, mmap=True)
        for key in ("generator", "model", "generator_ema"):
            if key in state:
                state = state[key]
                break
        state = {name.replace("_fsdp_wrapped_module.", "")
                     .replace("_checkpoint_wrapped_module.", "")
                     .replace("_orig_mod.", ""): value for name, value in state.items()}
        if EXPECTED_METHOD == "sgf_plus":
            from utils.checkpoint import initialize_memory_expert
            state = initialize_memory_expert(state, dict(self.model.generator.named_parameters()))
        self.model.generator.load_state_dict(state, strict=True)
        del state

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None, *,
                       loss_divisor=1, clip_gradients=True):
        self.model.eval()
        if self.config.i2v:
            raise ValueError("The matched SGF experiment is T2V")
        prompts = batch["prompts"]
        shape = list(self.config.image_or_video_shape)
        shape[0] = len(prompts)
        with torch.no_grad():
            conditions = self.model.text_encoder(text_prompts=prompts)
            if not getattr(self, "unconditional_dict", None):
                self.unconditional_dict = {
                    key: value.detach() for key, value in self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * len(prompts)).items()}
        kwargs = dict(image_or_video_shape=shape, conditional_dict=conditions,
                      unconditional_dict=self.unconditional_dict,
                      clean_latent=clean_latent, initial_latent=None)
        if train_generator:
            loss, logs = self.model.generator_loss(**kwargs)
            module, label, prefix = self.model.generator, "Student", "generator"
            max_norm = self.max_grad_norm_generator
        else:
            loss, logs = self.model.critic_loss(**kwargs)
            module, label, prefix = self.model.fake_score, "Critic", "critic"
            max_norm = self.max_grad_norm_critic
        self._check_update_finite({**logs, f"{prefix}_loss": loss}, label)
        (loss / loss_divisor).backward()
        norm = module.clip_grad_norm_(max_norm) if clip_gradients else loss.new_zeros(())
        logs = {key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in logs.items()}
        return {**logs, f"{prefix}_loss": loss.detach(), f"{prefix}_grad_norm": norm}

    def _init_tensorboard(self):
        self.tb_writer = None
        if not self.is_main_process or not getattr(self.config, "tensorboard", False):
            return
        from torch.utils.tensorboard import SummaryWriter

        flush_secs = int(getattr(self.config, "tensorboard_flush_secs", 15))
        if flush_secs <= 0:
            raise ValueError("tensorboard_flush_secs must be positive")
        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(self.output_path, "tensorboard"),
            flush_secs=flush_secs,
        )

    def _log_scalars(self, values):
        """Write the same rank-0 scalar names used by the CF experiments."""
        writer = getattr(self, "tb_writer", None)
        if not self.is_main_process or writer is None:
            return
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            writer.add_scalar(name, float(value), global_step=self.step)

    def _check_update_finite(self, values, label):
        """Stop every rank before an optimizer update if loss/grad is nonfinite."""
        if not getattr(self.config, "abort_on_nonfinite", False):
            return
        checks = [
            torch.isfinite(torch.as_tensor(value, device=self.device)).all()
            for key, value in values.items()
            if key.endswith("_loss") or key.endswith("_grad_norm")
            or key == "dmdtrain_gradient_norm"
        ]
        if "dmdtrain_nonfinite_fraction" in values:
            checks.append(torch.as_tensor(
                values["dmdtrain_nonfinite_fraction"], device=self.device
            ) == 0)
        finite = (torch.stack(checks).all() if checks else torch.ones((), device=self.device, dtype=torch.bool)).to(torch.int32)
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(
                f"Non-finite {label} loss/gradient before optimizer update at step {self.step}"
            )

    def _progress(self, phase, **extra):
        if self.is_main_process and getattr(self.config, "runtime_progress", False):
            print(json.dumps({"time": time.time(), "step": self.step,
                              "phase": phase, **extra}), flush=True)

    def _record_cuda_memory(self):
        if not getattr(self.config, "log_cuda_memory", False):
            return
        folder = os.path.join(self.output_path, "memory")
        os.makedirs(folder, exist_ok=True)
        record = {
            "time": time.time(), "step": self.step,
            "rank": dist.get_rank(), "host": os.uname().nodename,
            "scope": "global_step_including_ema_and_checkpoint",
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20,
            "allocated_mib": torch.cuda.memory_allocated(self.device) / 2**20,
        }
        with open(os.path.join(folder, f"rank{dist.get_rank()}.jsonl"), "a") as stream:
            stream.write(json.dumps(record) + "\n")

    def _save_checkpoint_file(self, state_dict, filename):
        """Publish a complete weights file only after serialization succeeds."""
        directory = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        os.makedirs(directory, exist_ok=True)
        destination = os.path.join(directory, filename)
        temporary = destination + ".tmp"
        try:
            torch.save(state_dict, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        print("Model saved to", destination, flush=True)

    def _accumulate_updates(self, train_generator):
        """Mean microbatch gradients; release each rollout graph before the next.

        Synchronize each FSDP microbatch instead of no_sync(): the latter retains
        full unsharded gradient buffers and defeats the memory goal here.
        Clip once after accumulation, then the caller performs one optimizer step.
        """
        count = self.gradient_accumulation_steps
        if count == 1:
            return self.fwdbwd_one_step(next(self.dataloader), train_generator)
        averaged = {}
        for _ in range(count):
            self._progress("student_microbatch" if train_generator else "critic_microbatch",
                           microbatch=_ + 1, accumulation=count)
            log = self.fwdbwd_one_step(
                next(self.dataloader), train_generator,
                loss_divisor=count, clip_gradients=False,
            )
            for key, value in log.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach()
                averaged[key] = averaged.get(key, 0.0) + value / count
            del log
        if train_generator:
            averaged["generator_grad_norm"] = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator
            )
        else:
            averaged["critic_grad_norm"] = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic
            )
        return averaged

    def train(self):
        try:
            self._train_loop()
        finally:
            writer = getattr(self, "tb_writer", None)
            if writer is not None:
                try:
                    writer.flush()
                finally:
                    writer.close()

    def _optimizer_update(self, train_student, values):
        label = "Student" if train_student else "Critic"
        kind = "student" if train_student else "critic"
        tick = time.perf_counter()
        self._progress(f"{kind}_update_start")
        optimizer = self.generator_optimizer if train_student else self.critic_optimizer
        optimizer.zero_grad(set_to_none=True)
        logs = self._accumulate_updates(train_student)
        self.data_batches_consumed += self.gradient_accumulation_steps
        self._check_update_finite(logs, label)
        optimizer.step()
        if train_student:
            self.student_updates += 1
            self._student_ema_update()
        else:
            self.critic_updates += 1
        values.update(logs)
        values[f"timing/{kind}_update_s"] = time.perf_counter() - tick

    def _student_ema_update(self):
        if self.generator_ema is not None:
            self.generator_ema.update(self.model.generator)
            self.ema_updates += 1
        elif self.student_updates >= int(self.config.ema_start_student_update):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

    def _write_metrics(self, values):
        if not self.is_main_process:
            return
        record = {"step": self.step, "student_updates": self.student_updates,
                  "critic_updates": self.critic_updates, "ema_updates": self.ema_updates,
                  "ema_initialized": self.generator_ema is not None,
                  "gradient_accumulation_steps": self.gradient_accumulation_steps,
                  "global_batch": self.config.total_batch_size}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            record[name] = float(value)
        print(json.dumps(record), flush=True)
        with open(os.path.join(self.output_path, "metrics.jsonl"), "a") as stream:
            stream.write(json.dumps(record) + "\n")
        self._log_scalars({key: value for key, value in record.items() if key != "step"})
        if not self.disable_wandb:
            import wandb
            wandb.log({key: value for key, value in record.items() if key != "step"}, step=self.step)

    def save(self, *, export_ema=None):
        state = fsdp_state_dict(self.model.generator)
        self._save_checkpoint_file_distributed({"generator": state}, "model.pt")
        del state
        if export_ema is False:
            return
        if self.generator_ema is None:
            raise RuntimeError("Final matched checkpoint requires initialized EMA")
        state = self.generator_ema.full_state_dict(self.model.generator)
        self._save_checkpoint_file_distributed({"generator_ema": state}, "model_ema.pt")

    def _train_loop(self):
        target, ratio, steps = training_budget(self.config)
        started = time.perf_counter()
        if self.step == steps and getattr(self.config, "resume_recovery", None):
            self._save_training_checkpoint(final=True, save_recovery_state=False)
            return
        while self.step < steps:
            if getattr(self.config, "log_cuda_memory", False):
                torch.cuda.reset_peak_memory_stats(self.device)
            values = {}
            # Zero-based 0,5,...745 => logged global steps 1,6,...746.
            if self.step % ratio == 0:
                if self.student_updates >= target:
                    raise RuntimeError("Student update budget exceeded")
                self._optimizer_update(True, values)
            self._optimizer_update(False, values)
            self.step += 1
            interval = int(getattr(self.config, "recoverable_checkpoint_interval", 0))
            if self.step == steps and not interval:
                expected_ema = target - int(self.config.ema_start_student_update)
                if (self.student_updates != target or self.critic_updates != steps
                        or self.generator_ema is None or self.ema_updates != expected_ema):
                    raise RuntimeError("Final checkpoint lacks the required optimizer/EMA updates")
                self._progress("checkpoint_start")
                tick = time.perf_counter()
                self.save()
                values["timing/checkpoint_s"] = time.perf_counter() - tick
                self._progress("checkpoint_end", elapsed_s=values["timing/checkpoint_s"])
            self._record_cuda_memory()
            elapsed = time.perf_counter() - started
            values["cost/training_loop_gpu_hours"] = elapsed * self.world_size / 3600
            self._write_metrics(values)
            if interval and (self.step % interval == 0 or self.step == steps):
                self._save_training_checkpoint(final=self.step == steps)
            if self.step == int(getattr(self.config, "early_validation_steps", 0)):
                self._write_early_validation()
            if self.step % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

    def _write_early_validation(self):
        if not self.is_main_process:
            return
        # All ranks already passed the loss and clipped-gradient MIN collective
        # before each optimizer update; these are real, retained training steps.
        record = {
            "time": time.time(), "step": self.step, "method": EXPECTED_METHOD,
            "world_size": self.world_size,
            "global_batch": self.world_size * int(self.config.batch_size) * self.gradient_accumulation_steps,
            "student_updates": self.student_updates, "critic_updates": self.critic_updates,
            "finite_losses_and_gradients_all_ranks": bool(self.config.abort_on_nonfinite),
            "retained_in_training_budget": True,
            "memory_records": "memory/rank*.jsonl",
        }
        memory_file = os.path.join(self.output_path, "memory", "rank0.jsonl")
        if os.path.exists(memory_file):
            with open(memory_file) as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            records = [item for item in records if item["step"] <= self.step]
            if records:
                record["rank0_peak_allocated_mib"] = max(item["peak_allocated_mib"] for item in records)
                record["rank0_peak_reserved_mib"] = max(item["peak_reserved_mib"] for item in records)
        destination = os.path.join(self.output_path, "EARLY_VALIDATION.json")
        with open(destination + ".tmp", "w") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        os.replace(destination + ".tmp", destination)
        self._progress("early_validation_passed", **record)

    def _write_model_parameters(self):
        if not self.is_main_process:
            return
        record = {"method": EXPECTED_METHOD, "objective": "model.dmd.DMD",
                  "gradient_mode": self.config.grad_mode,
                  "checkpoint": str(self.config.generator_ckpt),
                  "runtime": "trainer/matched_runtime.py",
                  "counted_before_fsdp": True}
        for role, attribute in (("student", "generator"), ("teacher", "real_score"),
                                ("critic", "fake_score"), ("text_encoder", "text_encoder"),
                                ("vae", "vae")):
            parameters = dict(getattr(self.model, attribute).named_parameters())
            writer = {name: value for name, value in parameters.items()
                      if ".memory_" in name or name.startswith("memory_")}
            record[role] = {
                "total": sum(p.numel() for p in parameters.values()),
                "trainable": sum(p.numel() for p in parameters.values() if p.requires_grad),
                "memory_writer_total": sum(p.numel() for p in writer.values()),
                "memory_writer_trainable": sum(p.numel() for p in writer.values() if p.requires_grad),
            }
        os.makedirs(self.output_path, exist_ok=True)
        destination = os.path.join(self.output_path, "model_parameters.json")
        with open(destination + ".tmp", "w") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        os.replace(destination + ".tmp", destination)

    def _save_checkpoint_file_distributed(self, state_dict, filename):
        """Keep every FSDP replica alive until rank 0 publishes a complete file.

        HYBRID_SHARD gathers synchronize only their shard group. A second node
        can therefore finish its gather and exit while global rank 0 is still
        serializing. The status broadcast and barrier deliberately use WORLD,
        not the model's shard group. A write failure is reported to all ranks
        before any rank enters the next gather or leaves the training process.
        """
        distributed = bool(getattr(dist, "is_initialized", lambda: False)())
        writer = dist.get_rank() == 0 if distributed else self.is_main_process
        status = [None]
        cause = None
        if writer:
            try:
                self._save_checkpoint_file(state_dict, filename)
            except Exception as exc:
                cause = exc
                status[0] = f"{type(exc).__name__}: {exc}"
        if distributed:
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise OSError(f"Checkpoint write failed for {filename}: {status[0]}") from cause
        if distributed:
            dist.barrier()

    def _recovery_data_state(self):
        return {
            "batches_consumed": getattr(self, "data_batches_consumed", 0),
            "dataset_path": os.path.realpath(self.config.data_path),
            "batch_size": int(self.config.batch_size),
            "accumulation": self.gradient_accumulation_steps,
            "seed": int(self.config.seed),
            "method": self.config.baseline_method,
        }

    def _save_training_checkpoint(self, *, final, save_recovery_state=True):
        from utils.training_recovery import save_recovery
        import shutil
        if final:
            expected = int(self.config.target_student_updates)
            if (self.student_updates != expected or self.generator_ema is None
                    or self.ema_updates != expected - int(self.config.ema_start_student_update)):
                raise RuntimeError("Final checkpoint lacks required Student/EMA updates")
        self._progress("checkpoint_start", recoverable=True)
        tick = time.perf_counter()
        # Publish an independent, sharded recovery snapshot BEFORE assembling
        # raw/EMA exports. If an export fails, training is still recoverable.
        if save_recovery_state:
            save_recovery(self, self._recovery_data_state())
        self.save(export_ema=final)
        self._record_cuda_memory()
        if final and self.is_main_process:
            # Retention is separate from save frequency. Never remove the last
            # recovery snapshot before both final exports have been published.
            for name in os.listdir(self.output_path):
                if (name.startswith("checkpoint_model_") and name[-6:].isdigit()
                        and int(name[-6:]) != self.step):
                    shutil.rmtree(os.path.join(self.output_path, name))
        dist.barrier()
        self._progress("checkpoint_end", recoverable=True,
                       elapsed_s=time.perf_counter() - tick)

    def _restore_training_recovery(self, directory):
        from utils.training_recovery import load_recovery, restore_rng
        loaded = load_recovery(self, directory)
        data = loaded["dataloader_state"]
        expected = self._recovery_data_state()
        for key in ("dataset_path", "batch_size", "accumulation", "seed", "method"):
            if data[key] != expected[key]:
                raise ValueError(f"Recovery data setting differs: {key}")
        # TextDataset uses deterministic DistributedSampler order. Recreate its
        # consumed position, then restore RNG so iterator creation cannot shift
        # diffusion noise, sampled denoising steps, or future shuffles.
        for _ in range(int(data["batches_consumed"])):
            next(self.dataloader)
        self.data_batches_consumed = int(data["batches_consumed"])
        # Keep a complete metric prefix when recovery uses a fresh run folder.
        status = [None]
        if self.is_main_process:
            try:
                source = os.path.join(loaded["source_run"], "metrics.jsonl")
                with open(source) as stream:
                    records = [json.loads(line) for line in stream if line.strip()]
                records = [row for row in records if row["step"] <= self.step]
                if [row["step"] for row in records] != list(range(1, self.step + 1)):
                    raise ValueError("Recovery metrics do not contain a complete step prefix")
                destination = os.path.join(self.output_path, "metrics.jsonl")
                os.makedirs(self.output_path, exist_ok=True)
                if os.path.exists(destination):
                    import shutil
                    shutil.copy2(destination, destination + f".before_recovery_{time.time_ns()}")
                with open(destination + ".tmp", "w") as stream:
                    for row in records:
                        stream.write(json.dumps(row) + "\n")
                os.replace(destination + ".tmp", destination)
            except Exception as exc:
                status[0] = repr(exc)
        dist.broadcast_object_list(status, src=0)
        if status[0]:
            raise RuntimeError(f"Cannot restore recovery metric prefix: {status[0]}")
        restore_rng(loaded["rng"])
        self._progress("recovery_restored", source=str(directory))
