import math
from typing import Optional, List, Dict
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file

from toy_vllm.config import ModelConfig
from toy_vllm.modules.layernorm import RMSNorm
from toy_vllm.modules.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb
from toy_vllm.modules.activation import SwiGLUMLP
from toy_vllm.modules.attention import repeat_kv
from toy_vllm.core.forward_context import ForwardContext
from toy_vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from toy_vllm.distributed.parallel_linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class ParallelGroupedQueryAttention(nn.Module):
    """
    Tensor-Parallel Grouped Query Attention (GQA).
    - Q, K, V Projections: ColumnParallelLinear
    - Local Attention over partition of heads
    - O Projection: RowParallelLinear with All-Reduce
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        # Local head counts on this partition
        self.local_num_heads = self.num_heads // self.tp_world_size
        self.local_num_kv_heads = max(1, self.num_kv_heads // self.tp_world_size)
        self.num_kv_groups = self.local_num_heads // self.local_num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = ColumnParallelLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
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

        # 1. ColumnParallel Linear Projections (No cross-GPU communication)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (batch, local_num_heads, seq_len, head_dim)
        q = q.view(bsz, q_len, self.local_num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(
            bsz, q_len, self.local_num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            bsz, q_len, self.local_num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # 2. RoPE
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Local Attention computation (Zero cross-GPU communication)
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

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            bsz, q_len, self.local_num_heads * self.head_dim
        )

        # 4. RowParallel O Projection: computes local GEMM + single All-Reduce
        output = self.o_proj(attn_output)
        return output


class ParallelSwiGLUMLP(nn.Module):
    """
    Tensor-Parallel SwiGLU MLP.
    - Gate & Up Projections: ColumnParallelLinear
    - Local SwiGLU activation
    - Down Projection: RowParallelLinear with All-Reduce
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local ColumnParallel Projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Local SwiGLU: SiLU(gate) * up (No communication needed!)
        activated = nn.functional.silu(gate) * up

        # RowParallel Down Projection: local GEMM + single All-Reduce
        return self.down_proj(activated)


class ParallelQwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ParallelGroupedQueryAttention(
            config, layer_idx=layer_idx
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = ParallelSwiGLUMLP(
            config.hidden_size, config.intermediate_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[ForwardContext] = None,
    ) -> torch.Tensor:
        # 1. Attn block (includes 1st All-Reduce inside o_proj)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            context=context,
        )
        hidden_states = residual + hidden_states

        # 2. MLP block (includes 2nd All-Reduce inside down_proj)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class ParallelQwen2ForCausalLM(nn.Module):
    """
    Tensor-Parallel Qwen2 Model supporting distributed inference across multiple GPUs.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                ParallelQwen2DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[ForwardContext] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
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
        logits = self.lm_head(hidden_states)
        return logits

    def load_partition_weights(
        self, state_dict: Dict[str, torch.Tensor], rank: int
    ) -> None:
        """Loads and slices weights for the specific tensor-parallel rank."""
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                state_dict["model.embed_tokens.weight"]
            )
            self.norm.weight.copy_(state_dict["model.norm.weight"])
            if not self.config.tie_word_embeddings:
                self.lm_head.weight.copy_(state_dict["lm_head.weight"])

            for i, layer in enumerate(self.layers):
                prefix = f"model.layers.{i}."
                layer.input_layernorm.weight.copy_(
                    state_dict[f"{prefix}input_layernorm.weight"]
                )
                layer.post_attention_layernorm.weight.copy_(
                    state_dict[f"{prefix}post_attention_layernorm.weight"]
                )

                # ColumnParallel Attention Projections
                layer.self_attn.q_proj.load_master_weight(
                    state_dict[f"{prefix}self_attn.q_proj.weight"], rank=rank
                )
                layer.self_attn.q_proj.load_master_bias(
                    state_dict[f"{prefix}self_attn.q_proj.bias"], rank=rank
                )

                layer.self_attn.k_proj.load_master_weight(
                    state_dict[f"{prefix}self_attn.k_proj.weight"], rank=rank
                )
                layer.self_attn.k_proj.load_master_bias(
                    state_dict[f"{prefix}self_attn.k_proj.bias"], rank=rank
                )

                layer.self_attn.v_proj.load_master_weight(
                    state_dict[f"{prefix}self_attn.v_proj.weight"], rank=rank
                )
                layer.self_attn.v_proj.load_master_bias(
                    state_dict[f"{prefix}self_attn.v_proj.bias"], rank=rank
                )

                # RowParallel O Projection
                layer.self_attn.o_proj.load_master_weight(
                    state_dict[f"{prefix}self_attn.o_proj.weight"], rank=rank
                )

                # ColumnParallel MLP Projections
                layer.mlp.gate_proj.load_master_weight(
                    state_dict[f"{prefix}mlp.gate_proj.weight"], rank=rank
                )
                layer.mlp.up_proj.load_master_weight(
                    state_dict[f"{prefix}mlp.up_proj.weight"], rank=rank
                )

                # RowParallel MLP Down Projection
                layer.mlp.down_proj.load_master_weight(
                    state_dict[f"{prefix}mlp.down_proj.weight"], rank=rank
                )

            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.embed_tokens.weight
