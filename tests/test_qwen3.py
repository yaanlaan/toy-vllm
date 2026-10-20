import torch
import pytest
from transformers import AutoTokenizer
from toy_vllm.config import ModelConfig, SamplingParams
from toy_vllm.models.qwen2 import Qwen2ForCausalLM
from toy_vllm.engine.llm_engine import LLMEngine

QWEN3_PATH = r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"


def test_qwen3_config_and_structure():
    """Verify Qwen3 configuration parameters and QK-Norm layer instantiation."""
    config = ModelConfig.from_pretrained(QWEN3_PATH)
    assert config.model_type == "qwen3"
    assert config.head_dim == 128
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.attention_bias is False

    model = Qwen2ForCausalLM(config)
    # Check that q_norm and k_norm are present in attention layers
    assert hasattr(model.model.layers[0].self_attn, "q_norm")
    assert model.model.layers[0].self_attn.q_norm is not None
    assert hasattr(model.model.layers[0].self_attn, "k_norm")
    assert model.model.layers[0].self_attn.k_norm is not None

    print("[PASS] Qwen3 config and QK-Norm structure verification passed!")


def test_qwen3_inference_alignment():
    """Verify Qwen3-0.6B runs with both V0 (no-cache) and V1/V3 (Paged KV Cache) with identical output."""
    engine = LLMEngine(model_path=QWEN3_PATH, device="cuda")

    prompt = "计算斐波那契数列前5项："
    params = SamplingParams(temperature=0.0, max_new_tokens=20)

    # V0: No cache
    tokens_v0 = list(
        engine.stream_generate(prompt, sampling_params=params, use_kv_cache=False)
    )
    # V1: Paged KV Cache
    tokens_v1 = list(
        engine.stream_generate(prompt, sampling_params=params, use_kv_cache=True)
    )

    text_v0 = "".join(tokens_v0)
    text_v1 = "".join(tokens_v1)

    print(f"Qwen3 V0 Output: {repr(text_v0)}")
    print(f"Qwen3 V1 Output: {repr(text_v1)}")

    assert tokens_v0 == tokens_v1, "Qwen3 V0 and V1 outputs mismatch!"
    assert (
        engine.block_allocator.get_num_free_blocks() == engine.num_gpu_blocks
    ), "Memory leak in Qwen3 PagedAttention!"
    print("[PASS] Qwen3-0.6B inference and Paged KV Cache alignment passed!")


if __name__ == "__main__":
    test_qwen3_config_and_structure()
    test_qwen3_inference_alignment()
    print("\n>>> All Qwen3 verification tests passed successfully! <<<")
