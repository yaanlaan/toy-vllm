from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field

from toy_vllm.config import SamplingParams
from toy_vllm.core.block_manager import BlockTable


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class Sequence:
    """
    Represents an ongoing sequence/request in the LLM Engine.
    Tracks token state, sampling parameters, block table, and lifecycle status.
    """

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        block_size: int = 16,
        device: str = "cuda",
    ):
        self.seq_id = seq_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.sampling_params = sampling_params
        self.status = SequenceStatus.WAITING

        self.block_table = BlockTable(block_size=block_size, device=device)

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    def get_last_token_id(self) -> int:
        """Returns the most recent token ID (either from prompt or output)."""
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token_id(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED


@dataclass
class RequestOutput:
    """User-facing output container for a sequence."""

    seq_id: int
    prompt: str
    text: str
    token_ids: List[int]
    finished: bool
