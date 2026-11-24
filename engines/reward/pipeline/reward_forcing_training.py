# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""Reward Forcing early-exit rollout and immutable EMA-Sink cache.

Algorithm source: JaydenLyh/Reward-Forcing@4317f5a0d29d1cee6a0f2e80ed5f5724bee97c66.
The cache stores unrotated keys; RoPE is reapplied to packed window positions,
as in the official EMA-Sink implementation. Immutable cache snapshots make
activation-checkpoint replay independent of later denoising/commit calls.
"""
import math
import torch
import torch.distributed as dist

from wan.modules.attention import attention
from wan.modules.causal_model import CausalWanSelfAttention, causal_rope_apply


def prepare_ema_history(history, incoming_tokens, capacity, sink_tokens, decay):
    """Evict once per new chunk, mixing the evicted raw KV into the sink."""
    if not history:
        return history, 0
    old_length = history[0][0].shape[1]
    evicted = max(0, old_length + incoming_tokens - capacity)
    if not evicted:
        return history, 0
    if not 0 < sink_tokens < capacity or evicted != sink_tokens:
        raise ValueError("Reward Forcing requires one 3-frame chunk per EMA-Sink eviction")
    result = []
    for key, value in history:
        updated = []
        for tensor in (key, value):
            sink = decay * tensor[:, :sink_tokens] + (1 - decay) * tensor[:, sink_tokens:sink_tokens + evicted]
            updated.append(torch.cat((sink, tensor[:, sink_tokens + evicted:]), dim=1).detach())
        result.append(tuple(updated))
    return result, 1


class RewardWanSelfAttention(CausalWanSelfAttention):
    """Same trainable weights as Wan; use EMA-Sink's raw-KV/relative-RoPE path."""
    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask,
                kv_cache=None, current_start=0, cache_start=None):
        if kv_cache is None or not kv_cache.get("reward_cache", False):
            return super().forward(x, seq_lens, grid_sizes, freqs, block_mask,
                                   kv_cache, current_start, cache_start)
        batch, length, _ = x.shape
        query = self.norm_q(self.q(x)).view(batch, length, self.num_heads, self.head_dim)
        key = self.norm_k(self.k(x)).view(batch, length, self.num_heads, self.head_dim)
        value = self.v(x).view(batch, length, self.num_heads, self.head_dim)
        keys = tuple(kv_cache.get("ste_k_chunks", ()))
        values = tuple(kv_cache.get("ste_v_chunks", ()))
        all_key = torch.cat((*keys, key), dim=1) if keys else key
        all_value = torch.cat((*values, value), dim=1) if values else value
        frame_tokens = math.prod(grid_sizes[0][1:].tolist())
        sink_tokens = self.sink_size * frame_tokens
        recent_start = max(sink_tokens, all_key.shape[1] - self.local_attn_size * frame_tokens + sink_tokens)
        # Upstream training retains a 21-frame buffer but READS only 9 frames.
        # Inference uses a 9-frame buffer; EMA updates on actual buffer eviction.
        all_key = torch.cat((all_key[:, :sink_tokens], all_key[:, recent_start:]), dim=1)
        all_value = torch.cat((all_value[:, :sink_tokens], all_value[:, recent_start:]), dim=1)
        prefix_length = all_key.shape[1] - key.shape[1]
        if prefix_length % frame_tokens or all_key.shape[1] % frame_tokens:
            raise ValueError("EMA-Sink RoPE requires complete latent frames")
        key_grid = grid_sizes.clone()
        key_grid[:, 0] = all_key.shape[1] // frame_tokens
        query = causal_rope_apply(query, grid_sizes, freqs, start_frame=prefix_length // frame_tokens).type_as(value)
        all_key = causal_rope_apply(all_key, key_grid, freqs, start_frame=0).type_as(value)
        output = attention(query, all_key, all_value)
        writer = kv_cache.get("ste_cache_output")
        if writer is not None:
            writer.update(k=key, v=value)
        return self.o(output.flatten(2))


def install_reward_attention(model):
    """Install before FSDP wrapping; state_dict names and parameter values stay intact."""
    count = 0
    for module in model.modules():
        if isinstance(module, CausalWanSelfAttention):
            module.__class__ = RewardWanSelfAttention
            count += 1
    if not count:
        raise ValueError("No causal Wan attention blocks found")
    return count


class RewardForcingTrainingPipeline:
    def __init__(self, denoising_step_list, scheduler, generator,
                 num_frame_per_block=3, same_step_across_blocks=True,
                 last_step_only=False, context_noise=0, local_attn_size=9,
                 sink_size=3, sink_decay=0.999, num_transformer_blocks=30,
                 independent_first_frame=False, cache_frames=21,
                 inference_mode=False, **kwargs):
        if independent_first_frame or num_frame_per_block != sink_size:
            raise ValueError("Reward baseline uses 3-frame chunks and one 3-frame sink")
        if local_attn_size <= sink_size + num_frame_per_block:
            raise ValueError("EMA-Sink needs sink, recent history, and current chunk")
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        self.scheduler, self.generator = scheduler, generator
        self.num_frame_per_block = num_frame_per_block
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only, self.context_noise = last_step_only, context_noise
        self.local_attn_size, self.sink_size, self.sink_decay = local_attn_size, sink_size, sink_decay
        self.cache_frames = cache_frames
        self.inference_mode = inference_mode
        self.num_transformer_blocks = num_transformer_blocks
        self.last_runtime_stats = {}

    def _cache(self, history, writer=False):
        result = []
        for layer in range(self.num_transformer_blocks):
            cache = {"ste_cache": True, "reward_cache": True,
                     "ste_k_chunks": (history[layer][0],) if history else (),
                     "ste_v_chunks": (history[layer][1],) if history else ()}
            if writer:
                cache["ste_cache_output"] = {}
            result.append(cache)
        return result

    def inference_with_trajectory(self, noise, initial_latent=None,
                                  clean_image_or_video=None, return_sim_step=False,
                                  **conditional_dict):
        if initial_latent is not None or clean_image_or_video is not None:
            raise ValueError("This matched Reward Forcing baseline is prompt-only T2V")
        batch, frames, _, height, width = noise.shape
        chunk_frames = self.num_frame_per_block
        if frames % chunk_frames:
            raise ValueError("Incomplete Reward Forcing chunk")
        chunks = frames // chunk_frames
        selected = (torch.full((chunks,), len(self.denoising_step_list) - 1, device=noise.device, dtype=torch.long)
                    if self.inference_mode else
                    torch.randint(len(self.denoising_step_list), (chunks,), device=noise.device))
        if self.last_step_only:
            selected.fill_(len(self.denoising_step_list) - 1)
        if dist.is_initialized():
            dist.broadcast(selected, src=0)
        selected = selected.tolist()
        history, outputs = [], []
        frame_tokens = (height // 2) * (width // 2)
        ema_updates = denoiser_forwards = 0
        for chunk in range(chunks):
            start = chunk * chunk_frames
            history, updates = prepare_ema_history(
                history, chunk_frames * frame_tokens,
                self.cache_frames * frame_tokens, self.sink_size * frame_tokens,
                self.sink_decay)
            ema_updates += updates
            selected_step = selected[0] if self.same_step_across_blocks else selected[chunk]
            current = noise[:, start:start + chunk_frames]
            for index in range(selected_step + 1):
                timestep = torch.ones((batch, chunk_frames), device=noise.device, dtype=torch.long) * self.denoising_step_list[index]
                with torch.set_grad_enabled(torch.is_grad_enabled() and index == selected_step):
                    _, prediction = self.generator(
                        noisy_image_or_video=current, conditional_dict=conditional_dict,
                        timestep=timestep, kv_cache=self._cache(history),
                        crossattn_cache=None, current_start=start * frame_tokens)
                denoiser_forwards += 1
                if index < selected_step:
                    next_timestep = self.denoising_step_list[index + 1].expand(batch * chunk_frames)
                    current = self.scheduler.add_noise(
                        prediction.flatten(0, 1), torch.randn_like(prediction.flatten(0, 1)),
                        next_timestep).unflatten(0, (batch, chunk_frames))
            outputs.append(prediction)
            # Detached selected clean estimate, as in official Reward/Self Forcing.
            # Commit once; neither writer nor previous denoising graphs survive in KV.
            with torch.no_grad():
                context_t = torch.full_like(timestep, self.context_noise)
                if self.inference_mode:
                    # The released inference commits clean output without an
                    # extra zero-noise draw. Preserve its seeded RNG sequence.
                    context = prediction.detach()
                else:
                    context = self.scheduler.add_noise(
                        prediction.detach().flatten(0, 1), torch.randn_like(prediction.flatten(0, 1)),
                        context_t.flatten()).unflatten(0, (batch, chunk_frames))
                _, _, written = self.generator(
                    noisy_image_or_video=context, conditional_dict=conditional_dict,
                    timestep=context_t, kv_cache=self._cache(history, writer=True),
                    crossattn_cache=None, current_start=start * frame_tokens)
                history = [(torch.cat((history[i][0], key.detach()), 1),
                            torch.cat((history[i][1], value.detach()), 1)) if history else
                           (key.detach(), value.detach()) for i, (key, value) in enumerate(written)]
        self.last_runtime_stats = {"reward_sink_updates": ema_updates,
                                   "reward_denoiser_forwards": denoiser_forwards,
                                   "reward_writer_forwards": chunks}
        output = torch.cat(outputs, dim=1)
        if self.same_step_across_blocks:
            schedule = self.scheduler.timesteps.to(noise.device)
            from_t = 1000 - torch.argmin((schedule - self.denoising_step_list[selected[0]]).abs()).item()
            to_t = (1000 - torch.argmin((schedule - self.denoising_step_list[selected[0] + 1]).abs()).item()
                    if selected[0] + 1 < len(self.denoising_step_list) else 0)
        else:
            from_t, to_t = None, None
        result = (output, from_t, to_t)
        return (*result, selected[0] + 1) if return_sim_step else result


def reward_inference_pipeline(generator, scheduler, args):
    """Full 4-step sampling with the official 9-frame inference EMA-Sink buffer.

    Caller encodes the same prompts and decodes the returned latents. Call under
    torch.no_grad(); unlike training this always executes all four steps.
    """
    install_reward_attention(generator.model)
    device = next(generator.parameters()).device
    steps = torch.tensor(args.denoising_step_list, device=device, dtype=torch.long)
    if args.warp_denoising_step:
        schedule = torch.cat((scheduler.timesteps.to(device), steps.new_zeros(1)))
        steps = schedule[1000 - steps]
    return RewardForcingTrainingPipeline(
        steps, scheduler, generator,
        num_frame_per_block=args.num_frame_per_block,
        same_step_across_blocks=True, last_step_only=True,
        context_noise=args.context_noise,
        local_attn_size=args.model_kwargs.local_attn_size,
        sink_size=args.model_kwargs.sink_size,
        sink_decay=getattr(args, "reward_sink_decay", 0.999),
        cache_frames=getattr(args, "reward_inference_cache_frames", 9),
        inference_mode=True)
