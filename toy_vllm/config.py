from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration aligned with Qwen2 / Qwen2.5 / Qwen3 / Llama style models."""
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: Optional[int] = None
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    torch_dtype: str = "bfloat16"
    tie_word_embeddings: bool = True
    attention_bias: Optional[bool] = None
    model_type: str = "qwen2"
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.attention_bias is None:
            # Qwen3 defaults to no bias, Qwen2 uses bias on QKV
            self.attention_bias = False if self.model_type == "qwen3" else True

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_pretrained(cls, model_path_or_dir: str) -> "ModelConfig":
        config_path = Path(model_path_or_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path_or_dir}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class SamplingParams:
    """Sampling hyperparameters for text generation."""
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 64
    repetition_penalty: float = 1.0
    stop_token_ids: List[int] = field(default_factory=lambda: [151643, 151645])
