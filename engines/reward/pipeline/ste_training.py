# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""Full-rollout S3 with differentiable history and a latent straight-through bridge.

The returned samples are the selected-step clean predictions (the unchanged DMD
supervision contract). History values come from the terminal four-step samples.
Only the selected denoiser calls and history writers build autograd graphs.
"""

import math
from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from .self_forcing_training import SelfForcingTrainingPipeline


def history_bridge(final_sample, selected_sample, alpha):
    """Terminal forward value, alpha-scaled selected-step input derivative."""
    return final_sample.detach() + float(alpha) * (
        selected_sample - selected_sample.detach()
    )


class STETrainingPipeline(SelfForcingTrainingPipeline):
    def __init__(
        self, *args, ste_alpha=1.0, ste_history_grad=True,
        ste_writer_only=False, ste_gradient_window=None,
        ste_local_graph_chunks=None, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ste_alpha = float(ste_alpha)
        self.ste_history_grad = bool(ste_history_grad)
        self.ste_writer_only = bool(ste_writer_only)
        if ste_gradient_window is not None:
            if isinstance(ste_gradient_window, bool) or not isinstance(ste_gradient_window, int):
                raise ValueError("ste_gradient_window must be an integer or null")
            if ste_gradient_window < -1:
                raise ValueError("ste_gradient_window must be -1, null, or non-negative")
        self.ste_gradient_window = (
            None if ste_gradient_window in (None, -1) else ste_gradient_window
        )
        if ste_local_graph_chunks is not None and (
            isinstance(ste_local_graph_chunks, bool)
            or not isinstance(ste_local_graph_chunks, int)
            or ste_local_graph_chunks not in (2, 3)
        ):
            raise ValueError("ste_local_graph_chunks must be null, 2, or 3")
        self.ste_local_graph_chunks = ste_local_graph_chunks
        if not math.isfinite(self.ste_alpha) or self.ste_alpha < 0:
            raise ValueError("ste_alpha must be finite and non-negative")
        if self.ste_writer_only and (not self.ste_history_grad or self.ste_alpha != 0):
            raise ValueError(
                "ste_writer_only requires ste_history_grad=true and ste_alpha=0"
            )
        if self.ste_gradient_window is not None and not self.ste_history_grad:
            raise ValueError("a finite gradient window requires ste_history_grad=true")
        if self.ste_writer_only and self.ste_gradient_window not in (None, 0):
            raise ValueError("ste_writer_only is compatible only with gradient window 0")
        if self.ste_local_graph_chunks is not None and (
            self.ste_gradient_window is not None
            or self.ste_writer_only or not self.ste_history_grad
        ):
            raise ValueError(
                "ste_local_graph_chunks requires ste_history_grad=true and is "
                "exclusive with finite ste_gradient_window and ste_writer_only"
            )
        if self.independent_first_frame or self.denoising_step_list_first_chunk is not None:
            raise ValueError("STE currently supports the shared chunkwise T2V schedule")
        if len(self.denoising_step_list) != 4:
            raise ValueError("STE requires exactly four nonzero denoising steps")
        if not self.same_step_across_blocks or self.last_step_only:
            raise ValueError("STE requires a shared random selected step across chunks")
        self.last_diagnostics = {}

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

    @staticmethod
    def _capture_rng(device):
        # Replay must neither resample model randomness nor advance the caller's
        # RNG. Wan's current dropout is zero; this also covers stochastic layers.
        cuda_device = None
        cuda_state = None
        if device.type == "cuda":
            cuda_device = device.index
            if cuda_device is None:
                cuda_device = torch.cuda.current_device()
            cuda_state = torch.cuda.get_rng_state(cuda_device)
        return torch.get_rng_state(), cuda_device, cuda_state

    @staticmethod
    @contextmanager
    def _replay_rng(state):
        cpu_state, cuda_device, cuda_state = state
        with torch.random.fork_rng(devices=[] if cuda_device is None else [cuda_device]):
            torch.set_rng_state(cpu_state)
            if cuda_device is not None:
                torch.cuda.set_rng_state(cuda_state, cuda_device)
            yield

    def _window_target(self, target, selected, trajectory, conditional_dict, *prefix_kv):
        """One target's graph, with an exact source-chunk gradient cutoff.

        Old chunks retain their writer-only graph. Recent selected denoisers and
        writers are replayed against fixed recorded values. This removes BOTH
        direct and indirect paths to generation outputs before ``target - W``.
        Simply detaching old KV would leave such indirect paths in recent KV.
        """
        first = max(0, target - self.ste_gradient_window)
        history = self._new_history()
        offset = 0
        for layer in history:
            layer["ste_k_chunks"] = tuple(prefix_kv[offset:offset + first])
            offset += first
            layer["ste_v_chunks"] = tuple(prefix_kv[offset:offset + first])
            offset += first
        if offset != len(prefix_kv):
            raise RuntimeError("gradient-window replay received an invalid prefix")
        crossattn_cache = [None] * self.num_transformer_blocks
        for chunk in range(first, target + 1):
            record = trajectory[chunk]
            with self._replay_rng(record["selected_rng"]):
                _, prediction = self.generator(
                    noisy_image_or_video=record["selected_input"],
                    conditional_dict=conditional_dict,
                    timestep=record["selected_timestep"],
                    kv_cache=self._cache_view(history),
                    crossattn_cache=crossattn_cache,
                    current_start=record["current_start"],
                )
            if chunk == target:
                return prediction
            committed = history_bridge(record["final"], prediction, self.ste_alpha)
            committed = self.scheduler.add_noise(
                committed.flatten(0, 1), record["context_noise"].flatten(0, 1),
                record["context_timestep"].flatten(0, 1),
            ).unflatten(0, committed.shape[:2])
            with self._replay_rng(record["writer_rng"]):
                _, _, written_kv = self.generator(
                    noisy_image_or_video=committed,
                    conditional_dict=conditional_dict,
                    timestep=record["context_timestep"],
                    kv_cache=self._cache_view(history, write=True),
                    crossattn_cache=crossattn_cache,
                    current_start=record["current_start"],
                )
            self._commit(history, written_kv)

    def _inference_with_window(self, noise, selected, conditional_dict):
        """Record one rollout, then construct separately bounded target graphs.

        Whole-target activation checkpointing keeps the replay activations out
        of the forward memory peak. Its history tensors are explicit arguments,
        rather than graph-bearing closure captures. The shared writer-only
        prefix is differentiable, including writers older than the window.
        """
        batch_size, num_frames = noise.shape[:2]
        num_chunks = num_frames // self.num_frame_per_block
        history = self._new_history()
        trajectory = []
        crossattn_cache = [None] * self.num_transformer_blocks
        # Later base writers are used only to obtain forward trajectory values;
        # they can never be in an old prefix of any target replay.
        differentiable_prefix_chunks = max(0, num_chunks - 1 - self.ste_gradient_window)

        for chunk in range(num_chunks):
            start = chunk * self.num_frame_per_block
            current_start = start * self.frame_seq_length
            noisy_input = noise[:, start:start + self.num_frame_per_block].detach()
            record = {"current_start": current_start}
            for step, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block],
                    device=noise.device, dtype=torch.int64,
                ) * current_timestep
                if step == selected:
                    record.update(
                        selected_input=noisy_input.detach(),
                        selected_timestep=timestep,
                        selected_rng=self._capture_rng(noise.device),
                    )
                with torch.no_grad():
                    _, prediction = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self._cache_view(history),
                        crossattn_cache=crossattn_cache,
                        current_start=current_start,
                    )
                    if step + 1 < len(self.denoising_step_list):
                        noisy_input = self.scheduler.add_noise(
                            prediction.flatten(0, 1),
                            torch.randn_like(prediction.flatten(0, 1)),
                            self.denoising_step_list[step + 1] * torch.ones(
                                [batch_size * self.num_frame_per_block],
                                device=noise.device, dtype=torch.long,
                            ),
                        ).unflatten(0, prediction.shape[:2])
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noise = torch.randn_like(prediction.flatten(0, 1)).unflatten(
                0, prediction.shape[:2]
            )
            committed = self.scheduler.add_noise(
                prediction.detach().flatten(0, 1), context_noise.flatten(0, 1),
                context_timestep.flatten(0, 1),
            ).unflatten(0, prediction.shape[:2])
            record.update(
                final=prediction.detach(), context_noise=context_noise,
                context_timestep=context_timestep,
                writer_rng=self._capture_rng(noise.device),
            )
            # The prefix writer gets parameter gradients but no gradient to
            # generation outputs. All replay branches share these old writers.
            with torch.set_grad_enabled(chunk < differentiable_prefix_chunks):
                _, _, written_kv = self.generator(
                    noisy_image_or_video=committed,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self._cache_view(history, write=True),
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                )
            self._commit(history, written_kv)
            trajectory.append(record)

        predictions = []
        replay_denoisers = 0
        replay_writers = 0
        for target in range(num_chunks):
            first = max(0, target - self.ste_gradient_window)
            prefix_kv = tuple(
                tensor
                for layer in history
                for key in ("ste_k_chunks", "ste_v_chunks")
                for tensor in layer[key][:first]
            )

            def replay(*tensors, target=target):
                return self._window_target(
                    target, selected, trajectory, conditional_dict, *tensors
                )

            # This checkpoint encloses complete FSDP forwards, unlike Wan's
            # block checkpoints inside the wrapped model. Non-reentrant early
            # stop can interrupt a replay before FSDP's post-forward cleanup.
            # In particular, a parameterless wrapper root around the sharded
            # Wan model is then left in FORWARD_BACKWARD at the next backward
            # hook. Finish the replay normally so FSDP owns its state lifecycle.
            # Checkpointing and all gradient paths remain unchanged.
            with set_checkpoint_early_stop(False):
                predictions.append(checkpoint(replay, *prefix_kv, use_reentrant=False))
            replay_denoisers += target - first + 1
            replay_writers += target - first

        output = torch.cat(predictions, dim=1)
        self.last_diagnostics = {
            "selected_step": selected + 1, "chunks": num_chunks,
            "denoiser_forwards": 4 * num_chunks + replay_denoisers,
            "history_forwards": num_chunks + replay_writers,
            "alpha": self.ste_alpha, "history_grad": 1.0,
            "writer_only": float(self.ste_gradient_window == 0),
            "gradient_window": self.ste_gradient_window,
            "replay_denoiser_forwards": replay_denoisers,
            "replay_history_forwards": replay_writers,
            "target_checkpointing": 1.0,
        }
        return output

    @torch.no_grad()
    def prepare_local_rollout(self, noise, **conditional_dict):
        """Record the full rollout without keeping any generation/writer graph.

        Full-video DMD scoring consumes ``clean_predictions`` once. The caller
        then replays one local target, backpropagates its globally normalized
        loss, and releases that graph before requesting the next target. No
        optimizer update may occur between these replays.

        Historical K/V still covers the complete forward prefix. Locality here
        limits autograd, not what the model sees or its four-step generation.
        """
        if self.ste_local_graph_chunks is None:
            raise ValueError("prepare_local_rollout requires ste_local_graph_chunks")
        batch_size, num_frames = noise.shape[:2]
        if num_frames % self.num_frame_per_block:
            raise ValueError("STE rollout frame count must contain complete chunks")
        if num_frames > self.kv_cache_size // self.frame_seq_length:
            raise ValueError("STE rollout exceeds the configured history capacity")
        num_chunks = num_frames // self.num_frame_per_block
        if num_chunks == 0:
            raise ValueError("STE rollout requires at least one chunk")
        selected = self.generate_and_sync_list(
            num_chunks, len(self.denoising_step_list), noise.device
        )[0]
        history = self._new_history()
        trajectory = []
        predictions = []
        crossattn_cache = [None] * self.num_transformer_blocks
        for chunk in range(num_chunks):
            start = chunk * self.num_frame_per_block
            current_start = start * self.frame_seq_length
            noisy_input = noise[:, start:start + self.num_frame_per_block].detach()
            record = {"current_start": current_start}
            for step, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block],
                    device=noise.device, dtype=torch.int64,
                ) * current_timestep
                if step == selected:
                    record.update(
                        selected_input=noisy_input.detach(),
                        selected_timestep=timestep,
                        selected_rng=self._capture_rng(noise.device),
                    )
                _, prediction = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self._cache_view(history),
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                )
                if step == selected:
                    predictions.append(prediction.detach())
                if step + 1 < len(self.denoising_step_list):
                    noisy_input = self.scheduler.add_noise(
                        prediction.flatten(0, 1),
                        torch.randn_like(prediction.flatten(0, 1)),
                        self.denoising_step_list[step + 1] * torch.ones(
                            [batch_size * self.num_frame_per_block],
                            device=noise.device, dtype=torch.long,
                        ),
                    ).unflatten(0, prediction.shape[:2])
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noise = torch.randn_like(prediction.flatten(0, 1)).unflatten(
                0, prediction.shape[:2]
            )
            committed = self.scheduler.add_noise(
                prediction.detach().flatten(0, 1), context_noise.flatten(0, 1),
                context_timestep.flatten(0, 1),
            ).unflatten(0, prediction.shape[:2])
            record.update(
                final=prediction.detach(), context_noise=context_noise,
                context_timestep=context_timestep,
                writer_rng=self._capture_rng(noise.device),
            )
            _, _, written_kv = self.generator(
                noisy_image_or_video=committed,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self._cache_view(history, write=True),
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            )
            self._commit(history, tuple((k.detach(), v.detach()) for k, v in written_kv))
            trajectory.append(record)
        timestep_from, timestep_to = self._selected_bounds(selected)
        local = self.ste_local_graph_chunks
        replay_denoisers = sum(min(local, target + 1) for target in range(num_chunks))
        replay_writers = replay_denoisers - num_chunks
        self.last_diagnostics = {
            "selected_step": selected + 1, "chunks": num_chunks,
            "denoiser_forwards": 4 * num_chunks + replay_denoisers,
            "history_forwards": num_chunks + replay_writers,
            "alpha": self.ste_alpha, "history_grad": 1.0,
            "writer_only": 0.0, "gradient_window": -1,
            "local_graph_chunks": local,
            "max_live_graph_chunks": min(local, num_chunks),
            "replay_denoiser_forwards": replay_denoisers,
            "replay_history_forwards": replay_writers,
            "target_checkpointing": 0.0,
        }
        return {
            "clean_predictions": torch.cat(predictions, dim=1),
            "timestep_from": timestep_from, "timestep_to": timestep_to,
            "selected_step": selected + 1, "trajectory": trajectory,
            "history": history, "num_chunks": num_chunks,
            "num_frame_per_block": self.num_frame_per_block,
            "local_graph_chunks": local,
        }

    def replay_local_target(self, recording, target, **conditional_dict):
        """Build only one target's local graph, to be backwarded immediately.

        Chunks older than ``target - local_graph_chunks + 1`` are detached in
        both their generation and writer paths. This is deliberately different
        from ``ste_gradient_window``: that experiment retains the entire writer
        graph and all targets until its single whole-video backward.
        """
        if self.ste_local_graph_chunks is None:
            raise ValueError("replay_local_target requires ste_local_graph_chunks")
        if not torch.is_grad_enabled():
            raise RuntimeError("replay_local_target requires autograd enabled")
        if recording["local_graph_chunks"] != self.ste_local_graph_chunks:
            raise ValueError("local rollout recording uses a different graph size")
        if isinstance(target, bool) or not isinstance(target, int):
            raise ValueError("local replay target must be an integer chunk index")
        if not 0 <= target < recording["num_chunks"]:
            raise ValueError("local replay target is outside the recorded trajectory")
        first = max(0, target - self.ste_local_graph_chunks + 1)
        history = self._new_history()
        for layer, recorded in zip(history, recording["history"]):
            for name in ("ste_k_chunks", "ste_v_chunks"):
                layer[name] = tuple(value.detach() for value in recorded[name][:first])
        crossattn_cache = [None] * self.num_transformer_blocks
        for chunk in range(first, target + 1):
            record = recording["trajectory"][chunk]
            with self._replay_rng(record["selected_rng"]):
                _, prediction = self.generator(
                    noisy_image_or_video=record["selected_input"],
                    conditional_dict=conditional_dict,
                    timestep=record["selected_timestep"],
                    kv_cache=self._cache_view(history),
                    crossattn_cache=crossattn_cache,
                    current_start=record["current_start"],
                )
            if chunk == target:
                return prediction
            committed = history_bridge(record["final"], prediction, self.ste_alpha)
            committed = self.scheduler.add_noise(
                committed.flatten(0, 1), record["context_noise"].flatten(0, 1),
                record["context_timestep"].flatten(0, 1),
            ).unflatten(0, committed.shape[:2])
            with self._replay_rng(record["writer_rng"]):
                _, _, written_kv = self.generator(
                    noisy_image_or_video=committed,
                    conditional_dict=conditional_dict,
                    timestep=record["context_timestep"],
                    kv_cache=self._cache_view(history, write=True),
                    crossattn_cache=crossattn_cache,
                    current_start=record["current_start"],
                )
            self._commit(history, written_kv)

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
        if outer_grad and self.ste_local_graph_chunks is not None:
            raise RuntimeError(
                "local graph training must use prepare_local_rollout and "
                "immediate backward after each replay_local_target"
            )
        selected = self.generate_and_sync_list(
            num_chunks, len(self.denoising_step_list), noise.device
        )[0]
        if (
            outer_grad and self.ste_gradient_window is not None
            and self.ste_gradient_window < num_chunks - 1
        ):
            output = self._inference_with_window(noise, selected, conditional_dict)
            timestep_from, timestep_to = self._selected_bounds(selected)
            if return_sim_step:
                return output, timestep_from, timestep_to, selected + 1
            return output, timestep_from, timestep_to
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
                # Writer-only keeps gradients through the history encoder and
                # its earlier writer K/V, but not the historical denoiser output.
                # Explicit detach also avoids retaining a zero-scaled bridge
                # graph. Direct DMD supervision of selected_prediction stays on.
                committed = (
                    prediction.detach() if self.ste_writer_only
                    else history_bridge(prediction, selected_prediction, self.ste_alpha)
                )
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
            "writer_only": float(self.ste_writer_only),
            "gradient_window": (
                -1 if self.ste_gradient_window is None else self.ste_gradient_window
            ),
            "replay_denoiser_forwards": 0,
            "replay_history_forwards": 0,
            "target_checkpointing": 0.0,
        }
        timestep_from, timestep_to = self._selected_bounds(selected)
        if return_sim_step:
            return output, timestep_from, timestep_to, selected + 1
        return output, timestep_from, timestep_to
