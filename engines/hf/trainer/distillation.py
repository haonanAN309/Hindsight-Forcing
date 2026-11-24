# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
import gc
import logging
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os
import json


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        # NCCL allocates memory outside PyTorch's caching allocator. Leave room
        # for its lazy communication buffers even after a large STE backward.
        memory_fraction = getattr(config, "cuda_memory_fraction", None)
        if memory_fraction is not None:
            memory_fraction = float(memory_fraction)
            if not 0 < memory_fraction <= 1:
                raise ValueError("cuda_memory_fraction must be in (0, 1]")
            torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device)
        if getattr(config, "abort_on_nonfinite", False):
            # Initialize this collective before rollout fills the allocator
            # cache. No RNG, model state, or gradient is involved.
            self._check_update_finite(
                {"startup_loss": torch.zeros((), device=self.device)}, "Startup"
            )
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.max_steps = getattr(config, "max_steps", None)
        if self.max_steps is not None and int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        self.gradient_accumulation_steps = int(getattr(config, "gradient_accumulation_steps", 1))
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if getattr(config, "ste_training", False):
            effective_batch = (
                self.world_size * int(config.batch_size) * self.gradient_accumulation_steps
            )
            if int(config.total_batch_size) != effective_batch:
                raise ValueError(
                    f"STE global batch is world_size*batch_size*accumulation={effective_batch}; "
                    "set total_batch_size accordingly."
                )

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir
        self._init_tensorboard()
        if self.is_main_process and getattr(config, "ste_training", False):
            os.makedirs(self.output_path, exist_ok=True)
            # Save the chosen runtime seed as well as the actual world size.
            config.runtime_world_size = self.world_size
            config.runtime_global_batch = effective_batch
            OmegaConf.save(config, os.path.join(self.output_path, "config.resolved.yaml"))
            print(
                f"[STE] world_size={self.world_size}, global_batch="
                f"{effective_batch}, alpha={getattr(config, 'ste_alpha', 1.0)}, "
                f"history_grad={getattr(config, 'ste_history_grad', True)}, "
                f"max_steps={self.max_steps}",
                flush=True,
            )

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

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
        """Mirror CF's rank-0 metrics; do not change training collectives."""
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
        finite = torch.stack(checks).all().to(torch.int32)
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

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        if getattr(self.config, "save_student_and_ema", False):
            # Save live weights before EMA temporarily swaps the model's
            # parameters. Every rank must participate in both FSDP gathers.
            if self.is_main_process:
                self._save_checkpoint_file({"generator": generator_state_dict}, "model.pt")
            del generator_state_dict
            if self.generator_ema is not None:
                ema_state_dict = self.generator_ema.full_state_dict(self.model.generator)
                if self.is_main_process:
                    self._save_checkpoint_file({"generator_ema": ema_state_dict}, "model_ema.pt")
            return

        if self.config.ema_start_step < self.step and self.generator_ema is not None:
            state_dict = {
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            self._save_checkpoint_file(state_dict, "model.pt")

    def save_critic(self):
        print("Start gathering distributed model states...")
        
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        
        state_dict = critic_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))
            
    def fwdbwd_one_step(
        self, batch, train_generator, clean_latent=None, *,
        loss_divisor=1, clip_gradients=True,
    ):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            # clean_latent = None #original code here
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            # clean_latent = None #original code here
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
           
            (generator_loss / loss_divisor).backward()
            generator_grad_norm = (
                self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                if clip_gradients else generator_loss.new_zeros(())
            )

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})
            if getattr(self.config, "ste_training", False):
                generator_log_dict.update({
                    f"ste/generator/{key}": value
                    for key, value in self.model.inference_pipeline.last_diagnostics.items()
                })

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        (critic_loss / loss_divisor).backward()
        critic_grad_norm = (
            self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
            if clip_gradients else critic_loss.new_zeros(())
        )

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})
        if getattr(self.config, "ste_training", False):
            critic_log_dict.update({
                f"ste/critic/{key}": value
                for key, value in self.model.inference_pipeline.last_diagnostics.items()
            })

        return critic_log_dict

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

    def _train_loop(self):
        start_step = self.step
        timing_enabled = bool(getattr(self.config, "runtime_progress", False))
       
        while self.max_steps is None or self.step < int(self.max_steps):
            timing_metrics = {}
            if getattr(self.config, "log_cuda_memory", False):
                torch.cuda.reset_peak_memory_stats(self.device)
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                if timing_enabled:
                    self._progress("student_update_start")
                    student_started = time.perf_counter()
                self.generator_optimizer.zero_grad(set_to_none=True)
                
                generator_log_dict = self._accumulate_updates(True)

                self._check_update_finite(generator_log_dict, "Student")
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                if timing_enabled:
                    timing_metrics["timing/student_update_s"] = time.perf_counter() - student_started
                    self._progress("student_update_end",
                                   elapsed_s=timing_metrics["timing/student_update_s"])

            # Train the critic
            if timing_enabled:
                self._progress("critic_update_start")
                critic_started = time.perf_counter()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_log_dict = self._accumulate_updates(False)
                
            self._check_update_finite(critic_log_dict, "Critic")
            self.critic_optimizer.step()
            if timing_enabled:
                timing_metrics["timing/critic_update_s"] = time.perf_counter() - critic_started
                self._progress("critic_update_end",
                               elapsed_s=timing_metrics["timing/critic_update_s"])

            # Increment the step since we finished gradient update
            self.step += 1
            self._record_cuda_memory()

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            final_step = self.max_steps is not None and self.step == int(self.max_steps)
            if (not self.config.no_save) and (self.step - start_step) > 0 and (
                self.step % self.config.log_iters == 0 or final_step
            ):
                if timing_enabled:
                    self._progress("checkpoint_start")
                    checkpoint_started = time.perf_counter()
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
                if timing_enabled:
                    timing_metrics["timing/checkpoint_s"] = time.perf_counter() - checkpoint_started
                    self._progress("checkpoint_end",
                                   elapsed_s=timing_metrics["timing/checkpoint_s"])

            # Logging
            if self.is_main_process:
                wandb_loss_dict = dict(timing_metrics)
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                    if "dmdtrain_nonfinite_fraction" in generator_log_dict:
                        wandb_loss_dict["dmdtrain_nonfinite_fraction"] = (
                            generator_log_dict["dmdtrain_nonfinite_fraction"].mean().item()
                        )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )
                if getattr(self.config, "ste_training", False):
                    for log in (generator_log_dict if TRAIN_GENERATOR else {}, critic_log_dict):
                        wandb_loss_dict.update({
                            key: value for key, value in log.items() if key.startswith("ste/")
                        })
                    record = {
                        "step": self.step,
                        "gradient_accumulation_steps": self.gradient_accumulation_steps,
                        **wandb_loss_dict,
                    }
                    print(json.dumps(record), flush=True)
                    with open(os.path.join(self.output_path, "metrics.jsonl"), "a") as stream:
                        stream.write(json.dumps(record) + "\n")

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)
                self._log_scalars(wandb_loss_dict)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    iteration_time = current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iteration_time}, step=self.step)
                    self._log_scalars({"per iteration time": iteration_time})
                    self.previous_time = current_time
