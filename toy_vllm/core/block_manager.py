from typing import List, Dict, Tuple, Optional, Union
import math
import torch
from toy_vllm.config import ModelConfig


class BlockAllocator:
    """
    Manages physical GPU memory blocks (pages).
    Maintains a free list and reference counting for each block.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Stack/list of available physical block IDs
        self.free_blocks: List[int] = list(range(num_blocks - 1, -1, -1))
        # Reference counting for copy-on-write / prefix caching
        self.ref_counts: Dict[int, int] = {i: 0 for i in range(num_blocks)}

    def allocate(self) -> int:
        """Allocates a single free physical block."""
        if not self.free_blocks:
            raise MemoryError(
                f"[BlockAllocator] Out of physical blocks! Total blocks: {self.num_blocks}"
            )
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id: int) -> None:
        """Decrements reference count and returns block to free list if count reaches 0."""
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid block_id: {block_id}")
        if self.ref_counts[block_id] <= 0:
            raise RuntimeError(f"Double-free on block_id: {block_id}")

        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)

    def free_blocks_list(self, block_ids: List[int]) -> None:
        """Frees a list of physical blocks."""
        for b_id in block_ids:
            self.free(b_id)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def reset(self) -> None:
        """Resets the allocator back to initial empty state."""
        self.free_blocks = list(range(self.num_blocks - 1, -1, -1))
        self.ref_counts = {i: 0 for i in range(self.num_blocks)}


class BlockTable:
    """
    Represents the logical-to-physical block mapping for a single sequence/request.
    Maintains an active GPU tensor of block IDs to eliminate CPU-to-GPU transfer overhead.
    """

    def __init__(self, block_size: int = 16, device: str = "cuda"):
        self.block_size = block_size
        self.device = device
        self.block_ids: List[int] = []
        self._gpu_tensor: Optional[torch.Tensor] = None
        self.num_tokens: int = 0

    @property
    def gpu_block_ids(self) -> torch.Tensor:
        if self._gpu_tensor is None or len(self._gpu_tensor) != len(
            self.block_ids
        ):
            self._gpu_tensor = torch.tensor(
                self.block_ids, device=self.device, dtype=torch.long
            )
        return self._gpu_tensor

    def allocate_for_prompt(
        self, prompt_len: int, allocator: BlockAllocator
    ) -> List[Tuple[int, int]]:
        """
        Allocates enough physical blocks to hold the initial prompt tokens.
        Returns a list of (physical_block_id, offset) for each token slot in the prompt.
        """
        num_blocks_needed = math.ceil(prompt_len / self.block_size)
        for _ in range(num_blocks_needed):
            self.block_ids.append(allocator.allocate())

        self._gpu_tensor = None  # Invalidate cached tensor
        slot_mapping: List[Tuple[int, int]] = []
        for i in range(prompt_len):
            logical_block_idx = i // self.block_size
            offset = i % self.block_size
            physical_block_id = self.block_ids[logical_block_idx]
            slot_mapping.append((physical_block_id, offset))

        self.num_tokens = prompt_len
        return slot_mapping

    def append_token(self, allocator: BlockAllocator) -> Tuple[int, int]:
        """
        Calculates the slot for the next generated token.
        If the current block is full (or if no block exists), allocates a new physical block.
        Returns (physical_block_id, offset).
        """
        logical_block_idx = self.num_tokens // self.block_size
        offset = self.num_tokens % self.block_size

        if offset == 0:
            # Need a new physical block
            new_block_id = allocator.allocate()
            self.block_ids.append(new_block_id)
            self._gpu_tensor = None  # Invalidate cached tensor

        physical_block_id = self.block_ids[logical_block_idx]
        self.num_tokens += 1
        return physical_block_id, offset

    def free_all(self, allocator: BlockAllocator) -> None:
        """Frees all physical blocks allocated to this sequence."""
        allocator.free_blocks_list(self.block_ids)
        self.block_ids = []
        self._gpu_tensor = None
        self.num_tokens = 0


class KVCachePool:
    """
    Global GPU physical memory pool holding Key and Value tensors for all layers.
    Shape per layer: (num_blocks, num_kv_heads, block_size, head_dim)
    """

    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int = 16,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        self.key_caches: List[torch.Tensor] = []
        self.value_caches: List[torch.Tensor] = []

        # Allocate contiguous pool for each layer
        for _ in range(config.num_hidden_layers):
            k_cache = torch.empty(
                (
                    num_blocks,
                    config.num_key_value_heads,
                    block_size,
                    config.head_dim,
                ),
                device=device,
                dtype=dtype,
            )
            v_cache = torch.empty(
                (
                    num_blocks,
                    config.num_key_value_heads,
                    block_size,
                    config.head_dim,
                ),
                device=device,
                dtype=dtype,
            )
            self.key_caches.append(k_cache)
            self.value_caches.append(v_cache)

    def write_kv(
        self,
        layer_idx: int,
        physical_block_id: int,
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Writes a single token's K and V into the specified slot.
        k, v shape: (1, num_kv_heads, 1, head_dim)
        """
        self.key_caches[layer_idx][
            physical_block_id, :, offset, :
        ] = k.squeeze()
        self.value_caches[layer_idx][
            physical_block_id, :, offset, :
        ] = v.squeeze()

    def write_prompt_kv(
        self,
        layer_idx: int,
        slot_mapping: List[Tuple[int, int]],
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Writes prompt tokens' K and V into their allocated slots.
        k, v shape: (1, num_kv_heads, prompt_len, head_dim)
        """
        k_sq = k.squeeze(0)
        v_sq = v.squeeze(0)
        for i, (b_id, offset) in enumerate(slot_mapping):
            self.key_caches[layer_idx][b_id, :, offset, :] = k_sq[:, i, :]
            self.value_caches[layer_idx][b_id, :, offset, :] = v_sq[:, i, :]

    def gather_kv(
        self,
        layer_idx: int,
        block_table: Union[List[int], torch.Tensor],
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gathers all historical K and V tokens for a sequence up to total_len.
        """
        # Slice the blocks needed: (num_blocks_needed, num_kv_heads, block_size, head_dim)
        k_blocks = self.key_caches[layer_idx][block_table]
        v_blocks = self.value_caches[layer_idx][block_table]

        # Reshape to (num_kv_heads, num_blocks_needed * block_size, head_dim)
        k_all = (
            k_blocks.permute(1, 0, 2, 3)
            .reshape(self.config.num_key_value_heads, -1, self.config.head_dim)[
                :, :total_len, :
            ]
            .unsqueeze(0)
        )
        v_all = (
            v_blocks.permute(1, 0, 2, 3)
            .reshape(self.config.num_key_value_heads, -1, self.config.head_dim)[
                :, :total_len, :
            ]
            .unsqueeze(0)
        )

        return k_all, v_all
