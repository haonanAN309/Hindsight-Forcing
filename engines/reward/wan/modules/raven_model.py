# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
"""RAVEN interleaved clean-history/noisy-target forward, unchanged parameter keys.

Visibility / frame shifts / zero clean timesteps follow mvp-ai-lab/RAVEN
692ec211d4cd37a99a04773186fc2bf810430f78 (CC-BY-NC-4.0),
projects/wan_t2v/data/causal_text_only.py and meta_models/causal_wan_t2v.py.
"""

import math
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from wan.modules.causal_model import CausalWanModel


def _raven_flex_attention(query, key, value, block_mask):
    # BlockMask's internally compiled construction can mark index dimensions
    # dynamic after a second sequence length. torch 2.6's autotuning generator
    # uses Python min() on those dimensions, so specialize the metadata shapes.
    # Mask values and visibility are untouched; a new length gets a new kernel.
    for item in block_mask.as_tuple():
        if torch.is_tensor(item):
            torch._dynamo.mark_static(item)
    return flex_attention(query, key, value, block_mask=block_mask)


def compile_raven_attention():
    """Allow valid four-warp backward kernels for BF16/head_dim=128 on H100.

    torch 2.6's only default candidate uses eight warps, which its backward
    lowering then skips (PyTorch #141603). max-autotune supplies four-warp
    candidates. Kernel tile options alone cannot fix that preselection check.
    This changes kernel selection only: the interleaved mask and gradients are
    unchanged. Disable CUDA graphs because this function runs inside FSDP and
    activation checkpointing; one shared callable reuses compiled shapes.
    """
    return torch.compile(_raven_flex_attention, dynamic=False,
                         mode="max-autotune-no-cudagraphs")


_raven_attention = None


def raven_attention_backend():
    global _raven_attention
    if _raven_attention is None:
        _raven_attention = compile_raven_attention()
    return _raven_attention


def interleave_chunks(noisy, clean, timestep, chunk_size=3):
    """Inputs B,C,F,H,W; return noisy0,clean0,...,noisyLast and metadata."""
    if noisy.shape != clean.shape or noisy.shape[2] % chunk_size:
        raise ValueError("RAVEN clean/noisy clips must share a complete chunk layout")
    chunks = noisy.shape[2] // chunk_size
    values, times, frame_ids, noisy_positions = [], [], [], []
    offset = 0
    for index in range(chunks):
        start, end = index * chunk_size, (index + 1) * chunk_size
        values.append(noisy[:, :, start:end])
        times.append(timestep[:, start:end])
        frame_ids.extend(range(start, end))
        noisy_positions.extend(range(offset, offset + chunk_size))
        offset += chunk_size
        if index < chunks - 1:
            values.append(clean[:, :, start:end])
            times.append(torch.zeros_like(timestep[:, start:end]))
            frame_ids.extend(range(start, end))
            offset += chunk_size
    return (torch.cat(values, dim=2), torch.cat(times, dim=1),
            torch.tensor(frame_ids, device=noisy.device, dtype=torch.long),
            torch.tensor(noisy_positions, device=noisy.device, dtype=torch.long))


def interleaved_visibility(query, key, tokens_per_chunk, total_tokens):
    query_segment = query // tokens_per_chunk
    key_segment = key // tokens_per_chunk
    real = (query < total_tokens) & (key < total_tokens)
    # A noisy segment never acts as history. Clean segments can read older
    # clean segments and themselves; target segments read prior clean history.
    return (real & ((query_segment == key_segment)
                    | ((key_segment < query_segment) & (key_segment % 2 == 1)))) | (query == key)


def interleaved_rope_frequencies(original, frame_ids):
    """Duplicate only temporal positions; spatial coordinates stay unchanged."""
    frequencies = original.clone()
    width = original.shape[1]
    temporal_width = width - 2 * (width // 3)
    frequencies[:len(frame_ids), :temporal_width] = original[frame_ids, :temporal_width]
    return frequencies


class RAVENWanModel(CausalWanModel):
    def _forward_train(self, x, t, context, seq_len, clean_x=None, aug_t=None,
                       clip_fea=None, y=None):
        if clean_x is None:
            return super()._forward_train(x, t, context, seq_len, clip_fea=clip_fea, y=y)
        if self.patch_size[0] != 1 or clip_fea is not None or y is not None:
            raise ValueError("This RAVEN adapter is the Wan2.1 chunk-wise T2V path")
        packed, times, frame_ids, noisy_positions = interleave_chunks(
            x, clean_x, t, self.num_frame_per_block,
        )
        frame_tokens = packed.shape[-2] * packed.shape[-1] // math.prod(self.patch_size[1:])
        total_tokens = packed.shape[2] * frame_tokens
        tokens_per_chunk = self.num_frame_per_block * frame_tokens
        mask_key = (total_tokens, tokens_per_chunk, str(packed.device))
        if getattr(self, "_raven_mask_key", None) != mask_key:
            def mask_mod(batch, head, query, key):
                return interleaved_visibility(query, key, tokens_per_chunk, total_tokens)
            padded = math.ceil(total_tokens / 128) * 128
            self._raven_mask = create_block_mask(mask_mod, B=None, H=None,
                                                  Q_LEN=padded, KV_LEN=padded,
                                                  device=packed.device, _compile=True)
            self._raven_mask_key = mask_key
        old_mask, old_freqs = self.block_mask, self.freqs
        self.block_mask = self._raven_mask
        self.freqs = interleaved_rope_frequencies(old_freqs.to(packed.device), frame_ids)
        try:
            # Existing block checkpointing retains these immutable kwargs for
            # backward; restoring module attributes does not alter the tensors.
            out = super()._forward_train(
                packed, times, context, max(seq_len, total_tokens),
                clip_fea=clip_fea, y=y,
            )
            return out.index_select(2, noisy_positions)
        finally:
            self.block_mask, self.freqs = old_mask, old_freqs


def install_raven_forward(model):
    """Opt-in class extension; no new parameters or checkpoint-key changes."""
    if not isinstance(model, CausalWanModel):
        raise TypeError("RAVEN requires the causal Wan backbone")
    model.__class__ = RAVENWanModel
    for block in model.blocks:
        block.self_attn._fullseq_attention = raven_attention_backend()
    return model
