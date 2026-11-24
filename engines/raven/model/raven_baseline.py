# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""RAVEN DMD (without CM-GRPO) on the shared CF initialization/engine.

Primary reference: https://github.com/mvp-ai-lab/RAVEN, commit
692ec211d4cd37a99a04773186fc2bf810430f78, causal_wan_t2v_dmd.py and raven.yaml.
Preserves full rollout reuse, interleaved history gradients, reverse-participation
chunk weights, native score timestep sampler and 2:1 Critic/Student updates.
"""

import torch
from model.base import BaseModel
from pipeline.raven_training import RAVENTrainingPipeline, flow_to_clean
from wan.modules.raven_model import install_raven_forward


def raven_score_timesteps(batch, frames, device, shift=5.):
    # Official UniformTrainingTimesteps: shift then affine [0.02,0.98],
    # followed by integer rounding. This is not CF's post-shift clamp.
    raw = torch.rand((batch,), dtype=torch.float64)
    warped = shift * raw / (1 + (shift - 1) * raw)
    levels = ((warped * .96 + .02) * 1000).round().long().to(device)
    return levels[:, None].expand(batch, frames)


def raven_chunk_weights(chunks, device, dtype=torch.float64):
    # Official dmd_losses:-1.0, normalized to mean element weight one.
    raw = 1 / torch.arange(chunks, 0, -1, device=device, dtype=dtype)
    return raw / raw.mean()


class RAVENBaseline(BaseModel):
    def __init__(self, args, device):
        if getattr(args, "ste_training", False) or getattr(args, "i2v", False):
            raise ValueError("RAVEN baseline is T2V interleaving, without HGR")
        if int(args.dfake_gen_update_ratio) != 2:
            raise ValueError("Preserve RAVEN's 2 Critic : 1 Student update ratio")
        super().__init__(args, device)
        self.num_frame_per_block = int(args.num_frame_per_block)
        self.generator.model.num_frame_per_block = self.num_frame_per_block
        install_raven_forward(self.generator.model)
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()
        self.inference_pipeline = None
        self.timestep_shift = float(args.timestep_shift)
        self.guidance_scale = float(args.guidance_scale)
        self.norm_clip_min = float(getattr(args, "raven_norm_clip_min", 1e-5))

    @torch.no_grad()
    def prepare_training_batch(self, image_or_video_shape, conditional_dict,
                               unconditional_dict, clean_latent=None, initial_latent=None):
        if initial_latent is not None or clean_latent is not None:
            raise ValueError("RAVEN uses its own detached full rollout")
        if self.inference_pipeline is None:
            self.inference_pipeline = RAVENTrainingPipeline(
                self.denoising_step_list, self.scheduler, self.generator,
                num_frame_per_block=self.num_frame_per_block,
                num_max_frames=image_or_video_shape[1], context_noise=0,
                model_dtype=self.dtype,
            )
        # RAVEN keeps the diffusion state/trajectory in FP32 and casts only at
        # the backbone boundary. This also preserves final clean KV inputs.
        noise = torch.randn(image_or_video_shape, device=self.device, dtype=torch.float32)
        return self.inference_pipeline.prepare(noise, conditional_dict)

    def generator_loss(self, image_or_video_shape, conditional_dict, unconditional_dict,
                       clean_latent=None, initial_latent=None, prepared_batch=None):
        if prepared_batch is None:
            raise ValueError("RAVEN Student must reuse the preceding Critic rollout")
        endpoints, trajectory = prepared_batch["endpoints"], prepared_batch["trajectory"]
        batch, frames = endpoints.shape[:2]
        selected = torch.randint(0, len(self.denoising_step_list), (batch,), device=self.device)
        noisy = trajectory[torch.arange(batch, device=self.device), selected].detach()
        timestep = self.denoising_step_list[selected, None].expand(batch, frames)
        score_t = raven_score_timesteps(batch, frames, self.device, self.timestep_shift)
        flow, _ = self.generator(noisy_image_or_video=noisy.to(self.dtype), conditional_dict=conditional_dict,
                                 timestep=timestep, clean_x=endpoints.detach().to(self.dtype))
        prediction = flow_to_clean(noisy, flow, timestep)
        with torch.no_grad():
            noise = torch.randn(prediction.shape, device=self.device, dtype=torch.float32)
            sigma = score_t[..., None, None, None].float() / 1000
            score_noisy = (1 - sigma) * prediction.detach() + sigma * noise
            model_input = score_noisy.to(self.dtype)
            fake_flow, _ = self.fake_score(model_input, conditional_dict, score_t)
            real_cond, _ = self.real_score(model_input, conditional_dict, score_t)
            real_uncond, _ = self.real_score(model_input, unconditional_dict, score_t)
            real_flow = real_uncond + self.guidance_scale * (real_cond - real_uncond)
            fake = flow_to_clean(score_noisy, fake_flow, score_t).double()
            real = flow_to_clean(score_noisy, real_flow, score_t).double()
            norm = (prediction.detach().double() - real).abs().mean((1, 2, 3, 4), keepdim=True)
            norm = norm.clamp_min(self.norm_clip_min)
            grad = (fake - real) / norm
        # Official RAVEN linear DMD surrogate; its derivative is (fake-real)/norm.
        elements = (real - fake) * (fake - prediction.double()) / norm
        chunks = frames // self.num_frame_per_block
        weights = raven_chunk_weights(chunks, elements.device, elements.dtype)
        per_chunk = elements.unflatten(1, (chunks, self.num_frame_per_block))
        loss = (per_chunk * weights[None, :, None, None, None, None]).mean()
        return loss, {"dmdtrain_gradient_norm": grad.abs().mean().detach(),
                      "raven/dmd_normalizer": norm.mean().detach(),
                      "raven/selected_step": selected.float().mean().detach() + 1,
                      "timestep": score_t.detach(),
                      "raven/nonfinite_grad_fraction": (~torch.isfinite(grad)).float().mean()}

    def critic_loss(self, image_or_video_shape, conditional_dict, unconditional_dict,
                    clean_latent=None, initial_latent=None, prepared_batch=None):
        if prepared_batch is None:
            raise ValueError("RAVEN Critic requires the prepared full rollout")
        clean = prepared_batch["endpoints"].detach()
        batch, frames = clean.shape[:2]
        timestep = raven_score_timesteps(batch, frames, self.device, self.timestep_shift)
        noise = torch.randn(clean.shape, device=self.device, dtype=torch.float32)
        sigma = timestep[..., None, None, None].float() / 1000
        noisy = (1 - sigma) * clean + sigma * noise
        flow, _ = self.fake_score(noisy.to(self.dtype), conditional_dict, timestep)
        loss = (flow.float() - (noise - clean).float()).square().mean()
        return loss, {"critic_timestep": timestep.detach()}
