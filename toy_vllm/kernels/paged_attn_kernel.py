import math
from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_decode_kernel(
    q_ptr,  # (B, H_q, D)
    k_cache_ptr,  # (num_blocks, H_kv, block_size, D)
    v_cache_ptr,  # (num_blocks, H_kv, block_size, D)
    block_tables_ptr,  # (B, max_blocks_per_seq)
    seq_lens_ptr,  # (B,)
    out_ptr,  # (B, H_q, D)
    scale: float,
    stride_q_b: int,
    stride_q_h: int,
    stride_q_d: int,
    stride_k_b: int,
    stride_k_h: int,
    stride_k_s: int,
    stride_k_d: int,
    stride_v_b: int,
    stride_v_h: int,
    stride_v_s: int,
    stride_v_d: int,
    stride_bt_b: int,
    stride_bt_s: int,
    stride_out_b: int,
    stride_out_h: int,
    stride_out_d: int,
    num_kv_groups: int,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Triton Kernel: PagedAttention for Decode phase (q_len = 1).
    Each thread block processes a single (seq, query_head) pair.
    Iterates across physical blocks and computes online softmax attention.
    """
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // num_kv_groups

    cur_seq_len = tl.load(seq_lens_ptr + seq_idx)
    if cur_seq_len <= 0:
        return

    # 1. Load Query vector for this head: (HEAD_DIM,)
    d_offsets = tl.arange(0, HEAD_DIM)
    q_offset = (
        seq_idx * stride_q_b + head_idx * stride_q_h + d_offsets * stride_q_d
    )
    q = tl.load(q_ptr + q_offset).to(tl.float32)

    # Online softmax accumulators
    m_i = -float("inf")  # Running maximum of logits
    l_i = 0.0  # Running sum of exp(logits - m_i)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)  # Running weighted sum of V

    # Number of physical blocks holding this sequence's KV
    num_blocks = (cur_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    s_offsets = tl.arange(0, BLOCK_SIZE)

    for block_num in range(num_blocks):
        # Physical block ID from block table
        bt_offset = seq_idx * stride_bt_b + block_num * stride_bt_s
        phys_block_id = tl.load(block_tables_ptr + bt_offset)

        # Token count within this block
        block_start_tok = block_num * BLOCK_SIZE
        tokens_in_block = cur_seq_len - block_start_tok
        if tokens_in_block > BLOCK_SIZE:
            tokens_in_block = BLOCK_SIZE

        # Load K for this block: (BLOCK_SIZE, HEAD_DIM)
        k_ptrs = (
            k_cache_ptr
            + phys_block_id * stride_k_b
            + kv_head_idx * stride_k_h
            + s_offsets[:, None] * stride_k_s
            + d_offsets[None, :] * stride_k_d
        )
        mask_k = (s_offsets[:, None] < tokens_in_block) & (
            d_offsets[None, :] < HEAD_DIM
        )
        k = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Dot product q @ k.T: (BLOCK_SIZE,)
        # q: (HEAD_DIM,), k: (BLOCK_SIZE, HEAD_DIM)
        qk = tl.sum(q[None, :] * k, axis=1) * scale

        # Mask out-of-range tokens in this block
        mask_s = s_offsets < tokens_in_block
        qk = tl.where(mask_s, qk, -float("inf"))

        # Online Softmax update
        block_max = tl.max(qk, axis=0)
        m_next = tl.maximum(m_i, block_max)

        # Exponentials
        p = tl.exp(qk - m_next)
        alpha = tl.exp(m_i - m_next)

        # Update denominator
        l_next = l_i * alpha + tl.sum(p, axis=0)

        # Load V for this block: (BLOCK_SIZE, HEAD_DIM)
        v_ptrs = (
            v_cache_ptr
            + phys_block_id * stride_v_b
            + kv_head_idx * stride_v_h
            + s_offsets[:, None] * stride_v_s
            + d_offsets[None, :] * stride_v_d
        )
        v = tl.load(v_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Accumulate output: acc = acc * alpha + p @ V
        # p: (BLOCK_SIZE,), v: (BLOCK_SIZE, HEAD_DIM)
        pv = tl.sum(p[:, None] * v, axis=0)
        acc = acc * alpha + pv

        # Advance state
        m_i = m_next
        l_i = l_next

    # Final normalization by sum of exp
    out = acc / l_i

    # Store result to out_ptr: (B, H_q, D)
    out_offset = (
        seq_idx * stride_out_b
        + head_idx * stride_out_h
        + d_offsets * stride_out_d
    )
    tl.store(out_ptr + out_offset, out)


def paged_attention_decode(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Python wrapper for Triton PagedAttention Decode.
    q: shape (B, num_heads, head_dim) or (B, 1, num_heads, head_dim)
    key_cache, value_cache: shape (num_blocks, num_kv_heads, block_size, head_dim)
    block_tables: shape (B, max_blocks)
    seq_lens: shape (B,)
    """
    if q.dim() == 4:
        # q shape is (bsz, num_heads, 1, head_dim)
        q = q.squeeze(2)

    bsz, num_heads, head_dim = q.shape
    num_blocks, num_kv_heads, block_size, _ = key_cache.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    output = torch.empty(
        (bsz, num_heads, head_dim), device=q.device, dtype=q.dtype
    )
    num_kv_groups = num_heads // num_kv_heads

    grid = (bsz, num_heads)
    _paged_attention_decode_kernel[grid](
        q,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        output,
        scale=scale,
        stride_q_b=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_k_b=key_cache.stride(0),
        stride_k_h=key_cache.stride(1),
        stride_k_s=key_cache.stride(2),
        stride_k_d=key_cache.stride(3),
        stride_v_b=value_cache.stride(0),
        stride_v_h=value_cache.stride(1),
        stride_v_s=value_cache.stride(2),
        stride_v_d=value_cache.stride(3),
        stride_bt_b=block_tables.stride(0),
        stride_bt_s=block_tables.stride(1),
        stride_out_b=output.stride(0),
        stride_out_h=output.stride(1),
        stride_out_d=output.stride(2),
        num_kv_groups=num_kv_groups,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )

    return output.unsqueeze(1)  # (B, 1, num_heads, head_dim)
