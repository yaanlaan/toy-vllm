from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
from toy_vllm.core.block_manager import KVCachePool


@dataclass
class ForwardContext:
    """
    Context passed through Transformer layers during forward inference.
    Supports both single-sequence and multi-sequence Continuous Batching,
    and is strictly CUDA Graph capture compliant.
    """

    kv_cache_pool: Optional[KVCachePool] = None
    is_prefill: bool = True

    # Prefill phase: slot mapping for all prompt tokens of a single sequence
    slot_mapping: Optional[List[Tuple[int, int]]] = None

    # Single-sequence Decode phase (V1/V3 backwards-compatible)
    block_table: Optional[torch.Tensor] = None
    current_slot: Optional[Tuple[int, int]] = None
    total_seq_len: int = 0
    seq_lens_tensor: Optional[torch.Tensor] = None

    # Multi-sequence Continuous Batching Decode phase (V2)
    batch_block_tables: Optional[List[torch.Tensor]] = None
    batch_current_slots: Optional[List[Tuple[int, int]]] = None
    batch_total_seq_lens: Optional[List[int]] = None
