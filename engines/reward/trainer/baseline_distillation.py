# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""Matched Student-update budgets without replacing each baseline's objective.

The legacy STE trainer and all frozen running experiments retain their existing
control flow. This trainer counts real optimizer updates, accumulates a batch
of 64 before each update, and exports only final live/EMA Student weights.
"""
import gc
import json
import os
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from trainer.distillation import Trainer as DistillationTrainer
from utils.distributed import EMA_FSDP


def training_budget(config):
    target = int(config.target_student_updates)
    ratio = int(config.dfake_gen_update_ratio)
    warmup = int(getattr(config, "critic_warmup_steps", 0))
    if target <= 0 or ratio <= 0 or warmup < 0:
        raise ValueError("Invalid Student budget, update ratio, or Critic warmup")
    return target, ratio, warmup, warmup + target * ratio


class Trainer(DistillationTrainer):
    def __init__(self, config):
        target, ratio, warmup, steps = training_budget(config)
        if config.baseline_method not in ("raven", "reward_forcing"):
            raise ValueError("Only the authorized RAVEN and Reward Forcing baselines are enabled")
        if config.distribution_loss != config.baseline_method:
            raise ValueError("Baseline objective and method must agree")
        if getattr(config, "ste_training", False):
            raise ValueError("Baseline training cannot silently enable HGR")
        if int(config.max_steps) != steps:
            raise ValueError("max_steps must match the actual Student budget and method ratio")
        if not getattr(config, "save_final_only", False) or not config.save_student_and_ema:
            raise ValueError("Baselines export final ordinary Student and EMA only")
        if config.no_save:
            raise ValueError("Final baseline weights must be saved")
        if not 0 <= int(config.ema_start_student_update) < target:
            raise ValueError("EMA must start before the final Student update")
        super().__init__(config)
        effective = self.world_size * int(config.batch_size) * self.gradient_accumulation_steps
        if effective != 64 or int(config.total_batch_size) != effective:
            raise ValueError("Each baseline optimizer update must accumulate global batch 64")
        self.target_student_updates = target
        self.student_updates = self.critic_updates = self.ema_updates = 0
        self.data_batches_consumed = 0
        # Match STE: initialize after Student update40, then apply decay only
        # after actual Student updates41..150, independent of global-step ratio.
        self.generator_ema = None
        if int(config.ema_start_student_update) == 0:
            self.generator_ema = EMA_FSDP(self.model.generator, decay=config.ema_weight)
        if hasattr(self.model, "prepare_auxiliary"):
            with torch.random.fork_rng(devices=[self.device]):
                self.model.prepare_auxiliary()
        recovery = getattr(config, "resume_recovery", None)
        if recovery:
            self._restore_training_recovery(recovery)
        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            config.runtime_world_size = self.world_size
            config.runtime_global_batch = effective
            config.runtime_expected_critic_updates = steps
            OmegaConf.save(config, os.path.join(self.output_path, "config.resolved.yaml"))

    def _build_model(self, config):
        if config.baseline_method == "reward_forcing":
            from model.reward_forcing_baseline import RewardForcingBaseline
            return RewardForcingBaseline(config, self.device)
        from model.raven_baseline import RAVENBaseline
        return RAVENBaseline(config, self.device)

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None, *,
                       loss_divisor=1, clip_gradients=True):
        self.model.eval()
        if "_raven_context" not in batch:
            self.data_batches_consumed = getattr(self, "data_batches_consumed", 0) + 1
        if self.config.i2v:
            raise ValueError("The matched baseline experiment is T2V")
        if "_raven_context" in batch:
            kwargs = dict(batch["_raven_context"])
            prompts = None
        else:
            prompts = batch["prompts"]
            shape = list(self.config.image_or_video_shape)
            shape[0] = len(prompts)
            with torch.no_grad():
                conditions = self.model.text_encoder(text_prompts=prompts)
                if not getattr(self, "unconditional_dict", None):
                    self.unconditional_dict = {
                        key: value.detach() for key, value in self.model.text_encoder(
                            text_prompts=[self.config.negative_prompt] * len(prompts)
                        ).items()
                    }
            kwargs = dict(image_or_video_shape=shape, conditional_dict=conditions,
                          unconditional_dict=self.unconditional_dict,
                          clean_latent=clean_latent, initial_latent=None)
            if self.config.baseline_method == "raven":
                with torch.no_grad():
                    kwargs["prepared_batch"] = self.model.prepare_training_batch(**kwargs)
                if getattr(self, "_cache_raven_critic_batches", False):
                    self._raven_cached_batches.append({"_raven_context": dict(kwargs)})
        if train_generator:
            if self.config.baseline_method == "reward_forcing":
                kwargs["text_prompts"] = prompts
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
        return {**logs, f"{prefix}_loss": loss.detach(), f"{prefix}_grad_norm": norm}

    def _optimizer_update(self, train_student, values):
        label = "Student" if train_student else "Critic"
        kind = "student" if train_student else "critic"
        tick = time.perf_counter()
        self._progress(f"{kind}_update_start")
        optimizer = self.generator_optimizer if train_student else self.critic_optimizer
        optimizer.zero_grad(set_to_none=True)
        logs = self._accumulate_updates(train_student)
        self._check_update_finite(logs, label)
        optimizer.step()
        if train_student:
            self.student_updates += 1
            self._student_ema_update()
        else:
            self.critic_updates += 1
        values.update(logs)
        values[f"timing/{kind}_update_s"] = time.perf_counter() - tick

    def _raven_updates(self, train_student, values):
        # RAVEN: prepare one rollout per Critic microbatch; update Critic first,
        # then reuse those exact graph-free samples for the Student update.
        self._raven_cached_batches = []
        self._cache_raven_critic_batches = train_student
        try:
            self._optimizer_update(False, values)
            if train_student:
                if len(self._raven_cached_batches) != self.gradient_accumulation_steps:
                    raise RuntimeError("RAVEN must reuse every accumulated Critic microbatch")
                original_loader = self.dataloader
                self.dataloader = iter(self._raven_cached_batches)
                try:
                    self._optimizer_update(True, values)
                finally:
                    self.dataloader = original_loader
        finally:
            self._cache_raven_critic_batches = False
            self._raven_cached_batches = []

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

    def _train_loop(self):
        target, ratio, warmup, steps = training_budget(self.config)
        started = time.perf_counter()
        if self.step == steps and getattr(self.config, "resume_recovery", None):
            # A committed final recovery may precede a failed weights export.
            # Retry publication without repeating training or overwriting it.
            self._save_training_checkpoint(final=True, save_recovery_state=False)
            return
        while self.step < steps:
            if getattr(self.config, "log_cuda_memory", False):
                torch.cuda.reset_peak_memory_stats(self.device)
            values = {}
            raven = self.config.baseline_method == "raven"
            train_student = (self.step >= warmup and
                             (self.step + int(raven) - warmup) % ratio == 0)
            if train_student and self.student_updates >= target:
                raise RuntimeError("Student update budget exceeded")
            if raven:
                self._raven_updates(train_student, values)
            else:
                if train_student:
                    self._optimizer_update(True, values)
                self._optimizer_update(False, values)
            self.step += 1
            interval = int(getattr(self.config, "recoverable_checkpoint_interval", 0))
            if self.step == steps and not interval:
                if self.student_updates != target or self.generator_ema is None:
                    raise RuntimeError("Final checkpoint lacks the required Student updates/EMA")
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
            if self.step % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

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
