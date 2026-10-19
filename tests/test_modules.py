import torch
import pytest
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2Attention,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)

from toy_vllm.config import ModelConfig, SamplingParams
from toy_vllm.modules.layernorm import RMSNorm
from toy_vllm.modules.activation import SwiGLUMLP
from toy_vllm.modules.rotary_embedding import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from toy_vllm.modules.attention import GroupedQueryAttention
from toy_vllm.modules.sampler import Sampler


def test_rmsnorm_alignment():
    """Verify that our RMSNorm produces bit-level identical outputs to HuggingFace's Qwen2RMSNorm."""
    torch.manual_seed(42)
    hidden_size = 896
    eps = 1e-6

    toy_norm = RMSNorm(hidden_size, eps=eps)
    hf_norm = Qwen2RMSNorm(hidden_size, eps=eps)

    # Initialize weights identically
    weight = torch.randn(hidden_size)
    toy_norm.weight.data.copy_(weight)
    hf_norm.weight.data.copy_(weight)

    x = torch.randn(2, 16, hidden_size)

    toy_out = toy_norm(x)
    hf_out = hf_norm(x)

    assert torch.allclose(
        toy_out, hf_out, atol=1e-5, rtol=1e-5
    ), "RMSNorm output mismatch!"
    print("[PASS] RMSNorm alignment test passed!")


def test_swiglu_alignment():
    """Verify SwiGLU MLP matches HuggingFace Qwen2MLP output exactly."""
    torch.manual_seed(42)
    hidden_size = 896
    intermediate_size = 4864

    hf_config = Qwen2Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
    )
    hf_mlp = Qwen2MLP(hf_config)
    toy_mlp = SwiGLUMLP(hidden_size, intermediate_size)

    # Copy weights
    toy_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
    toy_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
    toy_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)

    x = torch.randn(2, 8, hidden_size)
    toy_out = toy_mlp(x)
    hf_out = hf_mlp(x)

    assert torch.allclose(
        toy_out, hf_out, atol=1e-5, rtol=1e-5
    ), "SwiGLU output mismatch!"
    print("[PASS] SwiGLU MLP alignment test passed!")


def test_rope_alignment():
    """Verify RoPE output matches HuggingFace apply_rotary_pos_emb."""
    torch.manual_seed(42)
    hidden_size = 896
    num_heads = 14
    head_dim = hidden_size // num_heads
    seq_len = 10
    bsz = 2

    toy_rope = RotaryEmbedding(dim=head_dim, base=1000000.0)
    hf_config = Qwen2Config(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=2,
        rope_theta=1000000.0,
    )
    hf_rope = Qwen2RotaryEmbedding(hf_config)

    position_ids = (
        torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
    )  # (2, 10)
    q = torch.randn(bsz, num_heads, seq_len, head_dim)
    k = torch.randn(bsz, 2, seq_len, head_dim)

    # Toy rope
    toy_cos, toy_sin = toy_rope(q, position_ids)
    toy_q, toy_k = apply_rotary_pos_emb(q, k, toy_cos, toy_sin)

    # HF rope
    hf_cos, hf_sin = hf_rope(q, position_ids)
    hf_q, hf_k = hf_apply_rotary_pos_emb(q, k, hf_cos, hf_sin)

    assert torch.allclose(
        toy_q, hf_q, atol=1e-5, rtol=1e-5
    ), "RoPE Query mismatch!"
    assert torch.allclose(toy_k, hf_k, atol=1e-5, rtol=1e-5), "RoPE Key mismatch!"
    print("[PASS] RoPE alignment test passed!")


def test_attention_alignment():
    """Verify GroupedQueryAttention matches HuggingFace Qwen2Attention output."""
    torch.manual_seed(42)
    config = ModelConfig(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )
    hf_config = Qwen2Config(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
    )

    toy_attn = GroupedQueryAttention(config)
    hf_attn = Qwen2Attention(hf_config, layer_idx=0)

    # Copy weights and biases
    toy_attn.q_proj.weight.data.copy_(hf_attn.q_proj.weight.data)
    toy_attn.q_proj.bias.data.copy_(hf_attn.q_proj.bias.data)
    toy_attn.k_proj.weight.data.copy_(hf_attn.k_proj.weight.data)
    toy_attn.k_proj.bias.data.copy_(hf_attn.k_proj.bias.data)
    toy_attn.v_proj.weight.data.copy_(hf_attn.v_proj.weight.data)
    toy_attn.v_proj.bias.data.copy_(hf_attn.v_proj.bias.data)
    toy_attn.o_proj.weight.data.copy_(hf_attn.o_proj.weight.data)

    bsz, seq_len = 1, 8
    x = torch.randn(bsz, seq_len, config.hidden_size)
    position_ids = (
        torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)
    )  # (1, 8)

    toy_out = toy_attn(x, position_ids)

    # HF attention forward
    hf_rotary = Qwen2RotaryEmbedding(hf_config)
    hf_cos, hf_sin = hf_rotary(x, position_ids)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    )[None, None, :, :]
    hf_out, _ = hf_attn(
        x, position_embeddings=(hf_cos, hf_sin), attention_mask=causal_mask
    )

    assert torch.allclose(
        toy_out, hf_out, atol=1e-4, rtol=1e-4
    ), "Attention output mismatch!"
    print("[PASS] Attention alignment test passed!")


def test_sampler():
    """Test greedy sampling and temperature/top-k logic."""
    sampler = Sampler()
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])

    # Greedy test
    params_greedy = SamplingParams(temperature=0.0)
    token = sampler.sample(logits, params_greedy)
    assert token == 1, f"Expected 1, got {token}"

    # Top-K = 1 test
    params_top1 = SamplingParams(top_k=1, temperature=1.0)
    token = sampler.sample(logits, params_top1)
    assert token == 1, f"Expected 1, got {token}"

    print("[PASS] Sampler tests passed!")


if __name__ == "__main__":
    test_rmsnorm_alignment()
    test_swiglu_alignment()
    test_rope_alignment()
    test_attention_alignment()
    test_sampler()
    print("\n>>> All V0 module alignment tests passed successfully! <<<")
