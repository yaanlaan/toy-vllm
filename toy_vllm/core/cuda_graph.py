from typing import Optional, List, Dict
import torch
import torch.nn as nn

from toy_vllm.core.forward_context import ForwardContext


class CUDAGraphRunner:
    """
    Manages CUDA Graph capture and replay for Decode steps (q_len = 1).
    Eliminates Python CPU Launch Overhead by executing the entire multi-layer
    forward pass as a single hardware-dispatched graph.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None

        # Static pre-allocated buffers
        self.static_input_ids: Optional[torch.Tensor] = None
        self.static_position_ids: Optional[torch.Tensor] = None
        self.static_seq_lens: Optional[torch.Tensor] = None
        self.static_logits: Optional[torch.Tensor] = None
        self.static_context: Optional[ForwardContext] = None

        self.batch_size: int = 1
        self.is_captured: bool = False

    def capture(
        self,
        kv_cache_pool,
        batch_size: int = 1,
        dummy_block_table: Optional[torch.Tensor] = None,
        warmup_iters: int = 3,
    ) -> None:
        """
        Warms up and captures the static Decode execution graph.
        """
        if not torch.cuda.is_available():
            return

        self.batch_size = batch_size
        self.static_input_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=self.device
        )
        self.static_position_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=self.device
        )
        self.static_seq_lens = torch.ones(
            (batch_size,), dtype=torch.int32, device=self.device
        )

        if dummy_block_table is None:
            dummy_block_table = torch.zeros(
                (1,), dtype=torch.long, device=self.device
            )

        self.static_context = ForwardContext(
            kv_cache_pool=kv_cache_pool,
            is_prefill=False,
            block_table=dummy_block_table,
            current_slot=(0, 0),
            total_seq_len=1,
            seq_lens_tensor=self.static_seq_lens,
        )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        # 1. Warmup steps to establish static memory allocator addresses
        with torch.cuda.stream(stream):
            for _ in range(warmup_iters):
                self.static_logits = self.model(
                    self.static_input_ids,
                    position_ids=self.static_position_ids,
                    context=self.static_context,
                )
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # 2. Capture the Graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.static_logits = self.model(
                self.static_input_ids,
                position_ids=self.static_position_ids,
                context=self.static_context,
            )

        self.is_captured = True
        print(
            f"[toy-vllm] CUDA Graph captured successfully for Decode (batch_size={batch_size})!"
        )

    def replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        context: ForwardContext,
    ) -> torch.Tensor:
        """
        Copies runtime tokens and context into static buffers and replays the graph.
        """
        if not self.is_captured or self.graph is None:
            raise RuntimeError(
                "CUDA Graph must be captured before calling replay!"
            )

        # Update static inputs
        self.static_input_ids.copy_(input_ids)
        self.static_position_ids.copy_(position_ids)
        self.static_seq_lens.fill_(context.total_seq_len)

        # Update static context attributes
        self.static_context.block_table = context.block_table
        self.static_context.current_slot = context.current_slot
        self.static_context.total_seq_len = context.total_seq_len
        self.static_context.batch_block_tables = context.batch_block_tables
        self.static_context.batch_current_slots = context.batch_current_slots
        self.static_context.batch_total_seq_lens = context.batch_total_seq_lens

        # Replay entire 24-layer forward graph in a single hardware invocation
        self.graph.replay()
        return self.static_logits
