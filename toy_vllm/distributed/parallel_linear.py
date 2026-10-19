import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from toy_vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from toy_vllm.distributed.communication import tensor_model_parallel_all_reduce


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with weight matrix partitioned along output features (columns).
    Formula:
        W = [W_1, W_2, ..., W_k]
        Y_i = X @ W_i.T
    No cross-GPU communication required when followed by row-parallel operations!
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        self.tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        assert out_features % self.tp_world_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_world_size "
            f"({self.tp_world_size})"
        )
        self.out_features_per_partition = out_features // self.tp_world_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_parallel = F.linear(x, self.weight, self.bias)
        return output_parallel

    def load_master_weight(
        self, master_weight: torch.Tensor, rank: Optional[int] = None
    ) -> None:
        """Slices and loads the partition for the specified rank."""
        if rank is None:
            rank = self.tp_rank
        start = rank * self.out_features_per_partition
        end = start + self.out_features_per_partition
        self.weight.data.copy_(master_weight[start:end, :])

    def load_master_bias(
        self, master_bias: torch.Tensor, rank: Optional[int] = None
    ) -> None:
        if self.bias is None:
            return
        if rank is None:
            rank = self.tp_rank
        start = rank * self.out_features_per_partition
        end = start + self.out_features_per_partition
        self.bias.data.copy_(master_bias[start:end])


class RowParallelLinear(nn.Module):
    """
    Linear layer with weight matrix partitioned along input features (rows).
    Formula:
        W = [W_1; W_2; ...; W_k]
        Y = All-Reduce-Sum(X_i @ W_i.T) + Bias
    Directly consumes the column-parallel output and executes a single All-Reduce.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        self.tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        assert in_features % self.tp_world_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_world_size "
            f"({self.tp_world_size})"
        )
        self.in_features_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matrix multiply
        output_parallel = F.linear(x, self.weight)

        # All-reduce sum across all partitions
        output_ = tensor_model_parallel_all_reduce(output_parallel)

        if self.bias is not None:
            output_ = output_ + self.bias
        return output_

    def load_master_weight(
        self, master_weight: torch.Tensor, rank: Optional[int] = None
    ) -> None:
        """Slices and loads the partition for the specified rank."""
        if rank is None:
            rank = self.tp_rank
        start = rank * self.in_features_per_partition
        end = start + self.in_features_per_partition
        self.weight.data.copy_(master_weight[:, start:end])

    def load_master_bias(self, master_bias: torch.Tensor) -> None:
        if self.bias is not None:
            self.bias.data.copy_(master_bias)
