import os
import glob
import json
from pathlib import Path
from typing import Optional, List, Dict
import torch
import torch.nn as nn
from safetensors.torch import load_file

from toy_vllm.config import ModelConfig
from toy_vllm.modules.layernorm import RMSNorm
from toy_vllm.modules.attention import GroupedQueryAttention
from toy_vllm.modules.activation import SwiGLUMLP
from toy_vllm.core.forward_context import ForwardContext


class Qwen2DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Block for Qwen2 / Qwen2.5.
    Structure:
        x = x + self_attn(input_layernorm(x))
        x = x + mlp(post_attention_layernorm(x))
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[ForwardContext] = None,
    ) -> torch.Tensor:
        # Pre-LN Self Attention with Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            context=context,
        )
        hidden_states = residual + hidden_states

        # Pre-LN MLP with Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2Model(nn.Module):
    """
    Qwen2 base Transformer model without LM Head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[ForwardContext] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            if context is not None and not context.is_prefill:
                if context.batch_total_seq_lens is not None:
                    # Multi-sequence decode: each sequence has its own current position
                    pos_list = [
                        [total_l - 1] for total_l in context.batch_total_seq_lens
                    ]
                    position_ids = torch.tensor(
                        pos_list, dtype=torch.long, device=input_ids.device
                    )
                else:
                    # Single-sequence decode
                    position_ids = torch.tensor(
                        [[context.total_seq_len - 1]],
                        dtype=torch.long,
                        device=input_ids.device,
                    )
            else:
                position_ids = (
                    torch.arange(
                        seq_len, dtype=torch.long, device=input_ids.device
                    )
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                context=context,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    """
    Full Qwen2 / Qwen2.5 Causal Language Model with LM Head and safetensors loader.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[ForwardContext] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            context=context,
        )
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, model_dir: str, device: str = "cuda") -> None:
        """
        Loads safetensors weights directly into model parameters.
        Supports single or sharded safetensors files.
        """
        model_path = Path(model_dir)
        safetensor_files = sorted(list(model_path.glob("*.safetensors")))
        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_dir}!"
            )

        print(
            f"[toy-vllm] Loading {len(safetensor_files)} safetensor shard(s) into {device}..."
        )
        loaded_state_dict: Dict[str, torch.Tensor] = {}
        for sf_path in safetensor_files:
            tensors = load_file(str(sf_path), device="cpu")
            loaded_state_dict.update(tensors)

        # Load into module state_dict
        missing, unexpected = self.load_state_dict(
            loaded_state_dict, strict=False
        )
        print(
            f"[toy-vllm] Loaded weights: {len(loaded_state_dict)} tensors. "
            f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        target_dtype = getattr(torch, self.config.torch_dtype, torch.float16)
        self.to(device=device, dtype=target_dtype)
        self.eval()
