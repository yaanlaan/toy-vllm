import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as HFQwen2ForCausalLM,
)

from toy_vllm.config import ModelConfig
from toy_vllm.models.qwen2 import Qwen2ForCausalLM


def test_full_model_alignment():
    """Verify end-to-end full Qwen2 model forward pass matches HuggingFace exactly."""
    torch.manual_seed(42)

    # Use a miniature 2-layer config for lightning-fast exact testing
    toy_config = ModelConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )

    hf_config = Qwen2Config(
        hidden_size=toy_config.hidden_size,
        intermediate_size=toy_config.intermediate_size,
        num_hidden_layers=toy_config.num_hidden_layers,
        num_attention_heads=toy_config.num_attention_heads,
        num_key_value_heads=toy_config.num_key_value_heads,
        vocab_size=toy_config.vocab_size,
        rms_norm_eps=toy_config.rms_norm_eps,
        rope_theta=toy_config.rope_theta,
        tie_word_embeddings=False,
    )

    hf_model = HFQwen2ForCausalLM(hf_config)
    toy_model = Qwen2ForCausalLM(toy_config)

    # Copy all weights from HF model to Toy model directly using state_dict
    toy_model.load_state_dict(hf_model.state_dict())

    toy_model.eval()
    hf_model.eval()

    input_ids = torch.tensor([[10, 45, 128, 9, 250, 88]])  # (1, 6)

    with torch.no_grad():
        toy_logits = toy_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    # Check that logits match
    max_diff = (toy_logits - hf_logits).abs().max().item()
    print(f"Max absolute logits difference: {max_diff:.6f}")
    assert torch.allclose(
        toy_logits, hf_logits, atol=1e-4, rtol=1e-4
    ), f"Full model logits mismatch! Max diff: {max_diff}"
    print("[PASS] Full Qwen2 Model forward alignment test passed!")


if __name__ == "__main__":
    test_full_model_alignment()
