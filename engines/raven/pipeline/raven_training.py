# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""Detached full rollout reused by RAVEN's Critic and interleaved Student pass.

Algorithm reference: mvp-ai-lab/RAVEN, commit
692ec211d4cd37a99a04773186fc2bf810430f78, causal_wan_t2v_dmd.py / _infer.
This is a transcription into the existing Wan wrapper, without CM-GRPO or HGR.
"""

import torch
from pipeline.self_forcing_training import SelfForcingTrainingPipeline


def flow_to_clean(noisy, flow, timestep):
    sigma = timestep.to(device=noisy.device, dtype=torch.float32)
    while sigma.ndim < noisy.ndim:
        sigma = sigma.unsqueeze(-1)
    return noisy.float() - sigma / 1000 * flow.float()


class RAVENTrainingPipeline(SelfForcingTrainingPipeline):
    def __init__(self, *args, model_dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dtype = model_dtype

    @torch.no_grad()
    def prepare(self, noise, conditional_dict):
        batch, frames = noise.shape[:2]
        chunk = self.num_frame_per_block
        if frames % chunk or len(self.denoising_step_list) != 4:
            raise ValueError("RAVEN matched run requires complete 3-frame chunks and four steps")
        model_dtype = self.model_dtype or noise.dtype
        self._initialize_kv_cache(batch, model_dtype, noise.device)
        self._initialize_crossattn_cache(batch, model_dtype, noise.device)
        trajectory = [[] for _ in self.denoising_step_list]
        endpoints = []
        try:
            for start in range(0, frames, chunk):
                current = noise[:, start:start + chunk]
                for index, level in enumerate(self.denoising_step_list):
                    trajectory[index].append(current.detach().clone())
                    timestep = torch.full((batch, chunk), int(level),
                                          device=noise.device, dtype=torch.long)
                    flow, _ = self.generator(
                        noisy_image_or_video=current.to(model_dtype), conditional_dict=conditional_dict,
                        timestep=timestep, kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=start * self.frame_seq_length,
                    )
                    endpoint = flow_to_clean(current, flow, timestep)
                    next_level = self.denoising_step_list[index + 1] if index < 3 else 0
                    sigma = float(next_level) / 1000
                    # Official ConsistencySampler draws noise even at s=0.
                    fresh_noise = torch.randn(endpoint.shape, device=noise.device, dtype=torch.float32)
                    current = ((1 - sigma) * endpoint + sigma * fresh_noise).to(noise.dtype)
                endpoints.append(current.detach())
                if start + chunk < frames:
                    self.generator(
                        noisy_image_or_video=current.to(model_dtype), conditional_dict=conditional_dict,
                        timestep=torch.zeros((batch, chunk), device=noise.device, dtype=torch.long),
                        kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                        current_start=start * self.frame_seq_length,
                    )
            return {
                "endpoints": torch.cat(endpoints, dim=1).detach(),
                "trajectory": torch.stack([torch.cat(parts, dim=1) for parts in trajectory], dim=1).detach(),
            }
        finally:
            # Only detached latent trajectory needs to survive Critic backward.
            self.kv_cache1 = None
            self.crossattn_cache = None
