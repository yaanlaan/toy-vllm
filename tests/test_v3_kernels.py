import torch
import pytest
from toy_vllm.kernels.fused_ops import fused_add_rmsnorm, fused_swiglu
from toy_vllm.kernels.paged_attn_kernel import paged_attention_decode
from toy_vllm.modules.layernorm import RMSNorm
from toy_vllm.modules.attention import repeat_kv


def test_fused_add_rmsnorm():
    """Verify Triton Fused Add+RMSNorm matches PyTorch exact calculation."""
    torch.manual_seed(42)
    hidden_size = 896
    eps = 1e-6

    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    x = torch.randn(4, 1, hidden_size, device="cuda", dtype=torch.float32)
    residual = torch.randn(
        4, 1, hidden_size, device="cuda", dtype=torch.float32
    )

    # PyTorch baseline
    py_norm = RMSNorm(hidden_size, eps=eps).to("cuda")
    py_norm.weight.data.copy_(weight)
    expected_acc = x + residual
    expected_normed = py_norm(expected_acc)

    # Triton Fused
    fused_normed, fused_acc = fused_add_rmsnorm(
        x, residual, weight=weight, eps=eps
    )

    assert torch.allclose(
        fused_acc, expected_acc, atol=1e-5, rtol=1e-5
    ), "Fused Residual mismatch!"
    assert torch.allclose(
        fused_normed, expected_normed, atol=1e-4, rtol=1e-4
    ), "Fused RMSNorm output mismatch!"
    print("[PASS] Triton Fused Add+RMSNorm accuracy test passed!")


def test_fused_swiglu():
    """Verify Triton Fused SwiGLU matches PyTorch exact calculation."""
    torch.manual_seed(42)
    size = (4, 1, 4864)
    gate = torch.randn(size, device="cuda", dtype=torch.float32)
    up = torch.randn(size, device="cuda", dtype=torch.float32)

    # PyTorch baseline
    expected = (gate * torch.sigmoid(gate)) * up

    # Triton Fused
    fused_out = fused_swiglu(gate, up)

    assert torch.allclose(
        fused_out, expected, atol=1e-5, rtol=1e-5
    ), "Fused SwiGLU mismatch!"
    print("[PASS] Triton Fused SwiGLU accuracy test passed!")


def test_triton_paged_attention_decode():
    """Verify Triton PagedAttention Decode Kernel matches PyTorch Gathered Attention."""
    torch.manual_seed(42)
    bsz = 2
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    num_blocks = 20

    # Key & Value caches in GPU pool: (num_blocks, num_kv_heads, block_size, head_dim)
    k_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    v_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )

    # Query for 2 sequences: (B, num_heads, head_dim)
    q = torch.randn(
        bsz, num_heads, head_dim, device="cuda", dtype=torch.float32
    )

    # Sequence lengths: seq 0 has 25 tokens (2 blocks), seq 1 has 40 tokens (3 blocks)
    seq_lens = torch.tensor([25, 40], device="cuda", dtype=torch.int32)
    # Block tables: (B, max_blocks)
    block_tables = torch.tensor(
        [[2, 5, -1], [1, 4, 7]], device="cuda", dtype=torch.int32
    )

    # 1. PyTorch Reference: Gather blocks and compute standard attention
    scale = 1.0 / (head_dim**0.5)
    num_kv_groups = num_heads // num_kv_heads
    ref_outputs = []

    for b in range(bsz):
        cur_len = int(seq_lens[b].item())
        n_blks = (cur_len + block_size - 1) // block_size
        b_ids = block_tables[b, :n_blks].tolist()

        k_blks = k_cache[b_ids]  # (n_blks, num_kv_heads, block_size, head_dim)
        v_blks = v_cache[b_ids]

        k_all = (
            k_blks.permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim)[:, :cur_len, :]
            .unsqueeze(0)
        )
        v_all = (
            v_blks.permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim)[:, :cur_len, :]
            .unsqueeze(0)
        )

        k_all = repeat_kv(k_all, num_kv_groups)
        v_all = repeat_kv(v_all, num_kv_groups)

        q_b = q[b : b + 1].unsqueeze(2)  # (1, num_heads, 1, head_dim)
        attn_w = torch.matmul(q_b, k_all.transpose(2, 3)) * scale
        attn_w = torch.softmax(attn_w, dim=-1)
        out_b = torch.matmul(attn_w, v_all)  # (1, num_heads, 1, head_dim)
        ref_outputs.append(out_b.squeeze(2))

    ref_out = torch.cat(ref_outputs, dim=0)  # (B, num_heads, head_dim)

    # 2. Triton PagedAttention Kernel
    triton_out = paged_attention_decode(
        q, k_cache, v_cache, block_tables, seq_lens, scale=scale
    ).squeeze(1)

    max_diff = (triton_out - ref_out).abs().max().item()
    print(f"Max absolute difference in Triton PagedAttention: {max_diff:.6f}")
    assert torch.allclose(
        triton_out, ref_out, atol=1e-4, rtol=1e-4
    ), f"Triton PagedAttention mismatch! Max diff: {max_diff}"
    print("[PASS] Triton PagedAttention Decode numerical alignment passed!")


if __name__ == "__main__":
    test_fused_add_rmsnorm()
    test_fused_swiglu()
    test_triton_paged_attention_decode()
    print("\n>>> All V3 Triton Kernel tests passed successfully! <<<")
