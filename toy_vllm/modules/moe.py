from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_vllm.modules.activation import SwiGLUMLP


class MoERouter(nn.Module):
    """
    Top-K Gating Router for Mixture of Experts.
    Computes gating logits, selects top-k experts, and normalizes weights via local softmax.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (..., hidden_size)
        Returns:
            top_k_weights: (..., top_k) normalized routing probabilities
            top_k_indices: (..., top_k) selected expert IDs
            routing_logits: (..., num_experts) raw routing scores
        """
        routing_logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(
            routing_logits, self.top_k, dim=-1
        )
        top_k_weights = F.softmax(
            top_k_logits, dim=-1, dtype=torch.float32
        ).to(x.dtype)
        return top_k_weights, top_k_indices, routing_logits


class MoEExpert(nn.Module):
    """
    Individual Feed-Forward Expert based on SwiGLU.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MoEMLP(nn.Module):
    """
    Mixture-of-Experts MLP Layer aligned with Qwen2-MoE / DeepSeek architectures.
    Features:
      - Dedicated Shared Expert(s) for general knowledge
      - Top-K Gating Router for sparse routing
      - Highly-efficient 'Expert-selects-Token' permuted batch GEMM execution
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts

        self.router = MoERouter(hidden_size, num_experts, top_k=top_k)
        self.experts = nn.ModuleList(
            [
                MoEExpert(hidden_size, intermediate_size)
                for _ in range(num_experts)
            ]
        )

        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [
                    MoEExpert(
                        hidden_size, intermediate_size * num_shared_experts
                    )
                ]
            )
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]

        # 1. Shared Expert Forward (if configured)
        if self.shared_experts is not None:
            shared_output = sum(
                shared_exp(x_flat) for shared_exp in self.shared_experts
            )
        else:
            shared_output = torch.zeros_like(x_flat)

        # 2. Router decisions
        top_k_weights, top_k_indices, _ = self.router(x_flat)

        # 3. Industrial 'Expert-selects-Token' Permutation Pipeline
        # Pre-allocate output buffer
        routed_output = torch.zeros_like(x_flat)

        # Flat view of selected experts and corresponding weights
        # flat_top_k_indices: (num_tokens * top_k,)
        flat_topk_indices = top_k_indices.view(-1)
        flat_topk_weights = top_k_weights.view(-1)

        # Original token index mapping for each assignment: (num_tokens * top_k,)
        token_indices = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )

        # Group tokens by assigned expert and execute single Batched GEMM per expert
        for expert_id in range(self.num_experts):
            mask = flat_topk_indices == expert_id
            if not mask.any():
                continue

            selected_tokens = token_indices[mask]
            assigned_weights = flat_topk_weights[mask].unsqueeze(1)

            # Continuous chunk of tokens assigned to this expert
            expert_in = x_flat[selected_tokens]
            expert_out = self.experts[expert_id](expert_in)

            # Scatter-add weighted result back to token positions
            routed_output.index_add_(
                0, selected_tokens, assigned_weights * expert_out
            )

        # 4. Final Combination: Shared + Routed
        total_output = shared_output + routed_output
        return total_output.reshape(orig_shape)

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Naive token-by-token loop baseline used for testing and mathematical equivalence checks.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]

        shared_output = (
            self.shared_experts[0](x_flat)
            if self.shared_experts is not None
            else 0.0
        )
        top_k_weights, top_k_indices, _ = self.router(x_flat)

        routed_output = torch.zeros_like(x_flat)
        for i in range(num_tokens):
            tok = x_flat[i : i + 1]
            for k in range(self.top_k):
                exp_id = int(top_k_indices[i, k].item())
                weight = top_k_weights[i, k]
                routed_output[i : i + 1] += weight * self.experts[exp_id](tok)

        return (shared_output + routed_output).reshape(orig_shape)
