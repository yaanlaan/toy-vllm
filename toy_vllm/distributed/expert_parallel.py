from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn

from toy_vllm.modules.moe import MoERouter, MoEExpert


class ExpertParallelMoEMLP(nn.Module):
    """
    Expert-Parallel (EP) Mixture of Experts Layer.
    Partitions total experts across multiple GPUs:
      - Each GPU holds only E // ep_world_size experts (saving massive VRAM)
      - Implements dual All-to-All communication:
          1. Token dispatch (send tokens to destination expert's GPU)
          2. Result recovery (send computed features back to origin GPU)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        total_experts: int = 8,
        top_k: int = 2,
        ep_world_size: int = 2,
        ep_rank: int = 0,
        num_shared_experts: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.total_experts = total_experts
        self.top_k = top_k
        self.ep_world_size = ep_world_size
        self.ep_rank = ep_rank

        assert total_experts % ep_world_size == 0, (
            f"total_experts ({total_experts}) must be divisible by ep_world_size ({ep_world_size})"
        )
        self.local_num_experts = total_experts // ep_world_size
        self.start_expert_id = ep_rank * self.local_num_experts
        self.end_expert_id = self.start_expert_id + self.local_num_experts

        # Router replicated on each rank
        self.router = MoERouter(hidden_size, total_experts, top_k=top_k)

        # Local expert partition assigned to this rank only
        self.local_experts = nn.ModuleList(
            [
                MoEExpert(hidden_size, intermediate_size)
                for _ in range(self.local_num_experts)
            ]
        )

        # Replicated Shared Expert
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

    def is_local_expert(self, expert_id: int) -> bool:
        return self.start_expert_id <= expert_id < self.end_expert_id

    def get_expert_rank(self, expert_id: int) -> int:
        return expert_id // self.local_num_experts

    def load_master_weights(
        self,
        master_router_weight: torch.Tensor,
        all_expert_weights: List[Dict[str, torch.Tensor]],
        shared_expert_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Loads weights, slicing only the local partition of experts for this rank."""
        self.router.gate.weight.data.copy_(master_router_weight)

        for local_idx in range(self.local_num_experts):
            global_idx = self.start_expert_id + local_idx
            exp_state = all_expert_weights[global_idx]
            self.local_experts[local_idx].mlp.gate_proj.weight.data.copy_(
                exp_state["gate_proj.weight"]
            )
            self.local_experts[local_idx].mlp.up_proj.weight.data.copy_(
                exp_state["up_proj.weight"]
            )
            self.local_experts[local_idx].mlp.down_proj.weight.data.copy_(
                exp_state["down_proj.weight"]
            )

        if self.shared_experts is not None and shared_expert_weights is not None:
            self.shared_experts[0].mlp.gate_proj.weight.data.copy_(
                shared_expert_weights["gate_proj.weight"]
            )
            self.shared_experts[0].mlp.up_proj.weight.data.copy_(
                shared_expert_weights["up_proj.weight"]
            )
            self.shared_experts[0].mlp.down_proj.weight.data.copy_(
                shared_expert_weights["down_proj.weight"]
            )

    def forward_local_computation(
        self, token_chunk: torch.Tensor, global_expert_id: int
    ) -> torch.Tensor:
        """Runs GEMM on this rank's local expert."""
        assert self.is_local_expert(
            global_expert_id
        ), f"Expert {global_expert_id} not on rank {self.ep_rank}"
        local_idx = global_expert_id - self.start_expert_id
        return self.local_experts[local_idx](token_chunk)
