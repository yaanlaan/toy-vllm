import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from toy_vllm.config import ModelConfig
from toy_vllm.modules.layernorm import RMSNorm
from toy_vllm.modules.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb
from toy_vllm.core.forward_context import ForwardContext

try:
    from toy_vllm.kernels.paged_attn_kernel import paged_attention_decode

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands key/value heads to match query heads in Grouped Query Attention (GQA).
    (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_q_heads, seq_len, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) supporting:
      - V0: Naive full-sequence causal attention (no cache)
      - V1: Single-sequence Paged KV Cache
      - V2: Multi-sequence Continuous Batching Decode
      - V3: Triton PagedAttention Kernel Acceleration
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.is_qwen3 = (config.model_type == "qwen3")
        if self.is_qwen3:
            qkv_bias = False
            o_bias = False
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            qkv_bias = True
            o_bias = False
            self.q_norm = None
            self.k_norm = None

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=o_bias
        )

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[ForwardContext] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # 1. Linear Projections: Batch-wide GEMM
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (batch, seq_len, num_heads, head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # QK-Norm for Qwen3 architectures
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. RoPE Rotary Embeddings
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Paged KV Cache logic
        if context is not None and context.kv_cache_pool is not None:
            if context.is_prefill:
                # Prefill: Save entire prompt KV into pool slots
                if context.slot_mapping is not None:
                    context.kv_cache_pool.write_prompt_kv(
                        self.layer_idx, context.slot_mapping, k, v
                    )
                k = repeat_kv(k, self.num_kv_groups)
                v = repeat_kv(v, self.num_kv_groups)

                attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                else:
                    causal_mask = torch.triu(
                        torch.full(
                            (q_len, q_len),
                            float("-inf"),
                            device=q.device,
                            dtype=attn_weights.dtype,
                        ),
                        diagonal=1,
                    )
                    attn_weights = attn_weights + causal_mask

                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v)

            else:
                # Decode phase: q_len == 1
                if context.batch_block_tables is not None:
                    # ================= V2: Multi-sequence Batch Decode =================
                    batch_outputs: List[torch.Tensor] = []
                    for b in range(bsz):
                        block_id, offset = context.batch_current_slots[b]
                        context.kv_cache_pool.write_kv(
                            self.layer_idx,
                            block_id,
                            offset,
                            k[b : b + 1],
                            v[b : b + 1],
                        )

                        b_table = context.batch_block_tables[b]
                        b_len = context.batch_total_seq_lens[b]

                        if HAS_TRITON and q.is_cuda:
                            # Triton PagedAttention for sequence b
                            q_single = q[b : b + 1]  # (1, num_heads, 1, head_dim)
                            out_single = paged_attention_decode(
                                q_single,
                                context.kv_cache_pool.key_caches[self.layer_idx],
                                context.kv_cache_pool.value_caches[self.layer_idx],
                                b_table.unsqueeze(0),
                                torch.tensor([b_len], device=q.device, dtype=torch.int32),
                                scale=self.scale,
                            )  # (1, 1, num_heads, head_dim)
                            batch_outputs.append(out_single.transpose(1, 2))
                        else:
                            # PyTorch fallback
                            hist_k, hist_v = context.kv_cache_pool.gather_kv(
                                self.layer_idx, b_table, b_len
                            )
                            hist_k = repeat_kv(hist_k, self.num_kv_groups)
                            hist_v = repeat_kv(hist_v, self.num_kv_groups)
                            q_b = q[b : b + 1]
                            attn_w = (
                                torch.matmul(q_b, hist_k.transpose(2, 3))
                                * self.scale
                            )
                            attn_w = nn.functional.softmax(
                                attn_w, dim=-1, dtype=torch.float32
                            ).to(q.dtype)
                            out_b = torch.matmul(attn_w, hist_v)
                            batch_outputs.append(out_b)

                    attn_output = torch.cat(batch_outputs, dim=0)

                else:
                    # ================= V1/V3: Single-sequence Decode =================
                    block_id, offset = context.current_slot
                    context.kv_cache_pool.write_kv(
                        self.layer_idx, block_id, offset, k, v
                    )

                    if HAS_TRITON and q.is_cuda:
                        # V3 Triton PagedAttention Kernel Acceleration
                        table_tensor = (
                            context.block_table.unsqueeze(0)
                            if context.block_table.dim() == 1
                            else context.block_table
                        )
                        seq_len_tensor = (
                            context.seq_lens_tensor
                            if context.seq_lens_tensor is not None
                            else torch.tensor(
                                [context.total_seq_len],
                                device=q.device,
                                dtype=torch.int32,
                            )
                        )
                        attn_out = paged_attention_decode(
                            q,
                            context.kv_cache_pool.key_caches[self.layer_idx],
                            context.kv_cache_pool.value_caches[self.layer_idx],
                            table_tensor,
                            seq_len_tensor,
                            scale=self.scale,
                        )  # (1, 1, num_heads, head_dim)
                        attn_output = attn_out.transpose(1, 2)
                    else:
                        # PyTorch reference gather
                        hist_k, hist_v = context.kv_cache_pool.gather_kv(
                            self.layer_idx,
                            context.block_table,
                            context.total_seq_len,
                        )
                        hist_k = repeat_kv(hist_k, self.num_kv_groups)
                        hist_v = repeat_kv(hist_v, self.num_kv_groups)

                        attn_weights = (
                            torch.matmul(q, hist_k.transpose(2, 3)) * self.scale
                        )
                        attn_weights = nn.functional.softmax(
                            attn_weights, dim=-1, dtype=torch.float32
                        ).to(q.dtype)
                        attn_output = torch.matmul(attn_weights, hist_v)

        else:
            # V0 Fallback (no-cache)
            k = repeat_kv(k, self.num_kv_groups)
            v = repeat_kv(v, self.num_kv_groups)

            attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            else:
                causal_mask = torch.triu(
                    torch.full(
                        (q_len, q_len),
                        float("-inf"),
                        device=q.device,
                        dtype=attn_weights.dtype,
                    ),
                    diagonal=1,
                )
                attn_weights = attn_weights + causal_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)

        # 4. Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output)
