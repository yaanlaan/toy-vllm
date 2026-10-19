from typing import Optional
import torch
import torch.distributed as dist

_TP_WORLD_SIZE: int = 1
_TP_RANK: int = 0
_TP_GROUP: Optional[dist.ProcessGroup] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    rank: int = 0,
    backend: str = "gloo",
) -> None:
    """
    Initializes Tensor Parallelism state.
    Supports single-process simulation (when torch.distributed is not initialized)
    or standard multi-process torch.distributed environments.
    """
    global _TP_WORLD_SIZE, _TP_RANK, _TP_GROUP
    _TP_WORLD_SIZE = tensor_model_parallel_size
    _TP_RANK = rank

    if dist.is_initialized():
        _TP_GROUP = dist.group.WORLD
    else:
        _TP_GROUP = None


def get_tensor_model_parallel_world_size() -> int:
    return _TP_WORLD_SIZE


def get_tensor_model_parallel_rank() -> int:
    return _TP_RANK


def set_tensor_model_parallel_rank(rank: int) -> None:
    global _TP_RANK
    _TP_RANK = rank


def set_tensor_model_parallel_world_size(world_size: int) -> None:
    global _TP_WORLD_SIZE
    _TP_WORLD_SIZE = world_size


def destroy_model_parallel() -> None:
    global _TP_WORLD_SIZE, _TP_RANK, _TP_GROUP
    _TP_WORLD_SIZE = 1
    _TP_RANK = 0
    _TP_GROUP = None
