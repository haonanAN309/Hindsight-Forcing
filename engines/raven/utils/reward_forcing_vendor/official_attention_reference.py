# Modified for the Hindsight Forcing release.
# See the repository-root THIRD_PARTY_NOTICES.md for provenance and component licenses.
# Test-only reference from Reward-Forcing@4317f5a0d29d1cee6a0f2e80ed5f5724bee97c66. Apache-2.0.
import copy
import math
import torch
from torch import nn
class CausalWanSelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 compression_alpha=0.999): 
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.compression_alpha = compression_alpha  
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def incremental_update(self, new_k: torch.Tensor, new_v: torch.Tensor, 
                         current_sink_k: torch.Tensor, current_sink_v: torch.Tensor,
                         alpha: float = None) -> tuple:
        """
        Args:
            new_k: [B, sink_size, num_heads, head_dim]
            new_v:  [B, sink_size, num_heads, head_dim]
            current_sink_k:  [B, sink_size, num_heads, head_dim]
            current_sink_v:  [B, sink_size, num_heads, head_dim]
            alpha: self.compression_alpha
        Returns:
            updated_sink_k, updated_sink_v: [B, sink_size, num_heads, head_dim]
        """
        if alpha is None:
            alpha = self.compression_alpha
        
        # updated = α * current + (1-α) * new
        updated_sink_k = alpha * current_sink_k + (1 - alpha) * new_k
        updated_sink_v = alpha * current_sink_v + (1 - alpha) * new_v
        
        return updated_sink_k, updated_sink_v

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        current_end = current_start + q.shape[1]
        total_sink_tokens = self.sink_size * frame_seqlen
        
        # KV cache management
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = q.shape[1]
        
        evicted_tokens_exist = False
        evicted_k = None
        evicted_v = None
        
        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - total_sink_tokens
            
            evicted_start = total_sink_tokens
            evicted_end = total_sink_tokens + num_evicted_tokens
            evicted_k = kv_cache["k"][:, evicted_start:evicted_end].clone()
            evicted_v = kv_cache["v"][:, evicted_start:evicted_end].clone()
            evicted_tokens_exist = True
            
            kv_cache["k"][:, total_sink_tokens:total_sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, total_sink_tokens + num_evicted_tokens:total_sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, total_sink_tokens:total_sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, total_sink_tokens + num_evicted_tokens:total_sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            
            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v
        
        if evicted_tokens_exist and evicted_k is not None and evicted_k.shape[1] > 0:
            if evicted_k.shape[1] == total_sink_tokens:
                current_sink_k = kv_cache["k"][:, :total_sink_tokens]
                current_sink_v = kv_cache["v"][:, :total_sink_tokens]
                
                updated_sink_k, updated_sink_v = self.incremental_update(
                    evicted_k, evicted_v, 
                    current_sink_k, current_sink_v
                )
                
                kv_cache["k"][:, :total_sink_tokens] = updated_sink_k
                kv_cache["v"][:, :total_sink_tokens] = updated_sink_v
            else:
                min_len = min(evicted_k.shape[1], total_sink_tokens)
                kv_cache["k"][:, :min_len] = evicted_k[:, :min_len]
                kv_cache["v"][:, :min_len] = evicted_v[:, :min_len]
        
        # Prepare key and value tensors for attention
        kv_start_index = max(total_sink_tokens, local_end_index - self.max_attention_size + total_sink_tokens)
        kv_end_index = local_end_index
        
        # Calculate the actual number of tokens to use
        actual_kv_tokens = kv_end_index - kv_start_index
        
        # Ensure we have valid dimensions for RoPE application
        k_segment = kv_cache["k"][:, kv_start_index:kv_end_index]
        v_segment = kv_cache["v"][:, kv_start_index:kv_end_index]
        
        # Fix for sink_size > 1: use all sink tokens, not just a random one
        if self.sink_size > 0:
            sink_k = kv_cache["k"][:, :total_sink_tokens]
            sink_v = kv_cache["v"][:, :total_sink_tokens]
            
            k_combined = torch.cat([sink_k, k_segment], dim=1)
            v_combined = torch.cat([sink_v, v_segment], dim=1)
            
            # Update grid_sizes_kv to reflect the correct frame count
            grid_sizes_kv = copy.deepcopy(grid_sizes)
            total_combined_tokens = total_sink_tokens + actual_kv_tokens
            total_frames = total_combined_tokens // frame_seqlen
            grid_sizes_kv[:, 0] = total_frames
            
            # Calculate start frame for query RoPE
            query_start_frame = (total_sink_tokens + 
                               (local_end_index - max(total_sink_tokens, local_end_index - self.max_attention_size + total_sink_tokens)) - 
                               q.shape[1]) // frame_seqlen
            
            x = attention(
                causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=query_start_frame
                ).type_as(v),
                causal_rope_apply(
                    k_combined, grid_sizes_kv, freqs, start_frame=0
                ).type_as(v),
                v_combined,
            )
        else:
            # No sink tokens case
            grid_sizes_kv = copy.deepcopy(grid_sizes)
            grid_sizes_kv[:, 0] = actual_kv_tokens // frame_seqlen
            
            query_start_frame = (local_end_index - self.max_attention_size - q.shape[1]) // frame_seqlen
            
            x = attention(
                causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=query_start_frame
                ).type_as(v),
                causal_rope_apply(
                    k_segment, grid_sizes_kv, freqs, start_frame=0
                ).type_as(v),
                v_segment,
            )

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
