# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""Full-rollout S3 with differentiable history and a latent straight-through bridge.

The returned samples are the selected-step clean predictions (the unchanged DMD
supervision contract). HF uses terminal four-step history.
Only the selected denoiser calls and history writers build autograd graphs.
"""

import math

import torch

from .self_forcing_training import SelfForcingTrainingPipeline


def history_bridge(final_sample, selected_sample, alpha):
    """Terminal forward value, alpha-scaled selected-step input derivative."""
    return final_sample.detach() + float(alpha) * (
        selected_sample - selected_sample.detach()
    )


class STETrainingPipeline(SelfForcingTrainingPipeline):
    def __init__(self, *args, ste_alpha=1.0, ste_history_grad=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ste_alpha = float(ste_alpha)
        self.ste_history_grad = bool(ste_history_grad)
        if not math.isfinite(self.ste_alpha) or self.ste_alpha < 0:
            raise ValueError("ste_alpha must be finite and non-negative")
        if self.independent_first_frame or self.denoising_step_list_first_chunk is not None:
            raise ValueError("STE currently supports the shared chunkwise T2V schedule")
        if len(self.denoising_step_list) != 4:
            raise ValueError("STE requires exactly four nonzero denoising steps")
        if not self.same_step_across_blocks or self.last_step_only:
            raise ValueError("STE requires a shared random selected step across chunks")
        self.last_diagnostics = {}

    def _sample_selected_step(self, num_chunks, device):
        return self.generate_and_sync_list(
            num_chunks, len(self.denoising_step_list), device,
        )[0]

    def _new_history(self):
        # Keep only current-chunk K/V tensors in each tuple. Never retain a full
        # cloned prefix per chunk; checkpoint replays concatenate these references.
        return [
            {"ste_k_chunks": (), "ste_v_chunks": ()}
            for _ in range(self.num_transformer_blocks)
        ]

    def _cache_view(self, history, *, write=False):
        # Every forward/checkpoint gets its own dictionary snapshot. The tensor
        # references are shared, but no committed tensor is ever overwritten.
        view = [
            {
                "ste_cache": True,
                "ste_k_chunks": layer["ste_k_chunks"],
                "ste_v_chunks": layer["ste_v_chunks"],
            }
            for layer in history
        ]
        if write:
            for layer in view:
                layer["ste_cache_output"] = {}
        return view

    @staticmethod
    def _commit(history, written_kv):
        if len(history) != len(written_kv):
            raise RuntimeError("STE history writer returned an incorrect number of layers")
        for layer, (key, value) in zip(history, written_kv):
            layer["ste_k_chunks"] = layer["ste_k_chunks"] + (key,)
            layer["ste_v_chunks"] = layer["ste_v_chunks"] + (value,)

    def _selected_bounds(self, selected):
        # Match the original DMD sampler's index-space bounds, without hardcoded
        # .cuda() so the rollout/gradient contract can also be tested on CPU.
        grid = self.scheduler.timesteps

        def grid_index(t):
            t = torch.as_tensor(t, device=grid.device, dtype=grid.dtype)
            return 1000 - int((grid - t).abs().argmin().item())

        start = grid_index(self.denoising_step_list[selected])
        end = (
            0 if selected == len(self.denoising_step_list) - 1
            else grid_index(self.denoising_step_list[selected + 1])
        )
        return start, end

    def inference_with_trajectory(
        self, noise, clean_image_or_video=None, initial_latent=None,
        return_sim_step=False, **conditional_dict,
    ):
        if initial_latent is not None or clean_image_or_video is not None:
            raise ValueError("STE S3 uses prompt-conditioned self-rollout, not GT inputs")
        batch_size, num_frames = noise.shape[:2]
        if num_frames % self.num_frame_per_block:
            raise ValueError("STE rollout frame count must contain complete chunks")
        if num_frames > self.kv_cache_size // self.frame_seq_length:
            raise ValueError("STE rollout exceeds the configured history capacity")
        num_chunks = num_frames // self.num_frame_per_block
        if num_chunks == 0:
            raise ValueError("STE rollout requires at least one chunk")

        outer_grad = torch.is_grad_enabled()
        selected = self._sample_selected_step(num_chunks, noise.device)
        history = self._new_history()
        predictions = []
        # The STE model path recomputes cross-attention K/V on each forward.
        # Reusing no-grad text K/V would silently remove projection gradients.
        crossattn_cache = [None] * self.num_transformer_blocks

        for chunk_index in range(num_chunks):
            start = chunk_index * self.num_frame_per_block
            current_start = start * self.frame_seq_length
            noisy_input = noise[:, start:start + self.num_frame_per_block].detach()
            selected_prediction = None
            for step_index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block],
                    device=noise.device, dtype=torch.int64,
                ) * current_timestep
                with torch.set_grad_enabled(outer_grad and step_index == selected):
                    _, prediction = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self._cache_view(history),
                        crossattn_cache=crossattn_cache,
                        current_start=current_start,
                    )
                if step_index == selected:
                    selected_prediction = prediction
                if step_index + 1 < len(self.denoising_step_list):
                    # Stop the denoising-chain gradient, including at the
                    # transition out of the selected step. History's graph is
                    # retained separately and read by each selected forward.
                    with torch.no_grad():
                        noisy_input = self.scheduler.add_noise(
                            prediction.detach().flatten(0, 1),
                            torch.randn_like(prediction.flatten(0, 1)),
                            self.denoising_step_list[step_index + 1] * torch.ones(
                                [batch_size * self.num_frame_per_block],
                                device=noise.device, dtype=torch.long,
                            ),
                        ).unflatten(0, prediction.shape[:2])
            predictions.append(selected_prediction)

            # Preserve the original scheduler/context_noise semantics (including
            # nearest-grid handling at t=0). Only the source and its graph change.
            with torch.set_grad_enabled(outer_grad and self.ste_history_grad):
                committed = history_bridge(prediction, selected_prediction, self.ste_alpha)
                context_timestep = torch.ones_like(timestep) * self.context_noise
                committed = self.scheduler.add_noise(
                    committed.flatten(0, 1),
                    torch.randn_like(committed.flatten(0, 1)),
                    context_timestep.flatten(0, 1),
                ).unflatten(0, committed.shape[:2])
                writer_view = self._cache_view(history, write=True)
                _, _, written_kv = self.generator(
                    noisy_image_or_video=committed,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=writer_view,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                )
            self._commit(history, written_kv)

        # Do not stash live graphs on the long-lived pipeline. Output z retains
        # precisely the history graph that DMD's single backward needs.
        output = torch.cat(predictions, dim=1)
        self.last_diagnostics = {
            "selected_step": selected + 1,
            "chunks": num_chunks,
            "denoiser_forwards": 4 * num_chunks,
            "history_forwards": num_chunks,
            "alpha": self.ste_alpha,
            "history_grad": float(outer_grad and self.ste_history_grad),
        }
        timestep_from, timestep_to = self._selected_bounds(selected)
        if return_sim_step:
            return output, timestep_from, timestep_to, selected + 1
        return output, timestep_from, timestep_to
