from typing import Optional, Callable, List
import threading
import torch
import torch.distributed as dist

from toy_vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    _TP_GROUP,
)

_MOCK_ALL_REDUCE_HOOK: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class MockAllReduceGroup:
    """
    Thread-safe All-Reduce simulator for multi-rank execution within a single Python process.
    Uses barriers to coordinate lockstep computation across parallel worker threads.
    """

    def __init__(self, world_size: int = 2):
        self.world_size = world_size
        self.barrier_entry = threading.Barrier(world_size)
        self.barrier_exit = threading.Barrier(world_size)
        self.buffer: List[torch.Tensor] = []
        self.lock = threading.Lock()
        self.reduced_result: Optional[torch.Tensor] = None

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        with self.lock:
            self.buffer.append(tensor)

        # Wait until all ranks deposit their partition
        self.barrier_entry.wait()

        with self.lock:
            if self.reduced_result is None:
                res = self.buffer[0].clone()
                for t in self.buffer[1:]:
                    res = res + t
                self.reduced_result = res

        out = self.reduced_result.clone()

        # Wait until all ranks finish reading result
        self.barrier_exit.wait()

        with self.lock:
            self.buffer.clear()
            self.reduced_result = None

        return out


def set_mock_all_reduce_hook(
    hook: Optional[Callable[[torch.Tensor], torch.Tensor]]
) -> None:
    global _MOCK_ALL_REDUCE_HOOK
    _MOCK_ALL_REDUCE_HOOK = hook


def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs an all-reduce (SUM) across the tensor model parallel process group.
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return tensor

    if _MOCK_ALL_REDUCE_HOOK is not None:
        return _MOCK_ALL_REDUCE_HOOK(tensor)

    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TP_GROUP)
        return tensor

    return tensor
