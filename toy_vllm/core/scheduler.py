from typing import List, Optional
import math
from dataclasses import dataclass

from toy_vllm.core.sequence import Sequence, SequenceStatus
from toy_vllm.core.block_manager import BlockAllocator


@dataclass
class SchedulerOutputs:
    """Outputs from the scheduler for a single execution step."""

    prefill_seqs: List[Sequence]
    decode_seqs: List[Sequence]

    def is_empty(self) -> bool:
        return len(self.prefill_seqs) == 0 and len(self.decode_seqs) == 0


class Scheduler:
    """
    Continuous Batching Scheduler.
    Dynamically schedules requests at the iteration (step) level.
    Manages WAITING, RUNNING, and FINISHED sequence transitions with memory protection.
    """

    def __init__(
        self,
        block_allocator: BlockAllocator,
        max_num_seqs: int = 32,
        max_num_batched_tokens: int = 2048,
    ):
        self.block_allocator = block_allocator
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []

    def add_sequence(self, seq: Sequence) -> None:
        """Enqueues a new sequence into the WAITING queue."""
        self.waiting.append(seq)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    def schedule(self) -> SchedulerOutputs:
        """
        Core scheduling decision:
        1. If there are RUNNING sequences, prioritize scheduling them for batch Decode.
           Verify that free blocks exist if any sequence crosses a block boundary.
        2. If capacity allows and RUNNING has room (or is empty), schedule from WAITING for Prefill.
        """
        # ================= Policy 1: Prioritize Running Decode =================
        if self.running:
            # Check how many running sequences might need a new block in this step
            num_blocks_needed = 0
            for seq in self.running:
                # If offset is 0, append_token will require a new physical block
                if seq.total_len % seq.block_table.block_size == 0:
                    num_blocks_needed += 1

            if (
                self.block_allocator.get_num_free_blocks()
                >= num_blocks_needed
            ):
                # We can safely run all running sequences in a single decode batch
                decode_seqs = list(self.running)
                return SchedulerOutputs(
                    prefill_seqs=[], decode_seqs=decode_seqs
                )

        # ================= Policy 2: Admit Waiting Sequences (Prefill) =================
        prefill_seqs: List[Sequence] = []
        while self.waiting and len(self.running) < self.max_num_seqs:
            candidate = self.waiting[0]
            prompt_len = len(candidate.prompt_token_ids)
            blocks_needed = math.ceil(
                prompt_len / candidate.block_table.block_size
            )

            # Check if allocator has enough blocks
            if self.block_allocator.get_num_free_blocks() >= blocks_needed:
                seq = self.waiting.pop(0)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                prefill_seqs.append(seq)
                # Currently schedule 1 prefill per step for simplicity and fast turnaround
                break
            else:
                # Memory full, wait for running sequences to finish and free blocks
                break

        if prefill_seqs:
            return SchedulerOutputs(prefill_seqs=prefill_seqs, decode_seqs=[])

        # If we had running sequences but couldn't prefill, fall back to decode
        if self.running:
            return SchedulerOutputs(
                prefill_seqs=[], decode_seqs=list(self.running)
            )

        return SchedulerOutputs(prefill_seqs=[], decode_seqs=[])

    def free_sequence(self, seq: Sequence) -> None:
        """Reclaims all physical memory blocks allocated to this sequence and marks it FINISHED."""
        seq.block_table.free_all(self.block_allocator)
        seq.status = SequenceStatus.FINISHED
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
