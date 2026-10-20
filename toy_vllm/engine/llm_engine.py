from typing import Optional, Iterator, List, Dict
import torch
from transformers import AutoTokenizer

from toy_vllm.config import ModelConfig, SamplingParams
from toy_vllm.models.qwen2 import Qwen2ForCausalLM
from toy_vllm.modules.sampler import Sampler
from toy_vllm.core.block_manager import (
    BlockAllocator,
    BlockTable,
    KVCachePool,
)
from toy_vllm.core.forward_context import ForwardContext
from toy_vllm.core.sequence import Sequence, SequenceStatus, RequestOutput
from toy_vllm.core.scheduler import Scheduler, SchedulerOutputs
from toy_vllm.core.cuda_graph import CUDAGraphRunner


class LLMEngine:
    """
    High-Performance LLM Inference Engine supporting:
      - V0: Naive Autoregressive full-sequence recomputation (use_kv_cache=False)
      - V1: Single-sequence Paged KV Cache (use_kv_cache=True)
      - V2: Continuous Batching dynamic iteration-level scheduler
      - V3: Triton PagedAttention, Fused Ops & CUDA Graph acceleration
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        num_gpu_blocks: int = 1024,
        block_size: int = 16,
        max_num_seqs: int = 32,
        use_cuda_graph: bool = False,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.config = ModelConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.max_num_seqs = max_num_seqs
        self.use_cuda_graph = use_cuda_graph

        model_name = self.config.model_type.upper() if hasattr(self.config, "model_type") else "Qwen"
        print(
            f"[toy-vllm] Initializing {model_name} ({self.config.num_hidden_layers} layers, "
            f"hidden_size={self.config.hidden_size}) on {self.device}..."
        )
        self.model = Qwen2ForCausalLM(self.config)
        self.model.load_weights(model_path, device=self.device)
        self.sampler = Sampler()

        # Initialize global Paged KV Cache Pool & Allocator
        target_dtype = getattr(torch, self.config.torch_dtype, torch.float16)
        print(
            f"[toy-vllm] Allocating Paged KV Cache pool: {num_gpu_blocks} blocks "
            f"(block_size={block_size}, dtype={target_dtype})..."
        )
        self.block_allocator = BlockAllocator(
            num_blocks=num_gpu_blocks, block_size=block_size
        )
        self.kv_cache_pool = KVCachePool(
            config=self.config,
            num_blocks=num_gpu_blocks,
            block_size=block_size,
            device=self.device,
            dtype=target_dtype,
        )

        # V2 Continuous Batching Scheduler
        self.scheduler = Scheduler(
            block_allocator=self.block_allocator,
            max_num_seqs=max_num_seqs,
        )
        self._next_seq_id: int = 0
        self._all_sequences: Dict[int, Sequence] = {}

        # V3 CUDA Graph Runner
        self.cuda_graph_runner = CUDAGraphRunner(
            self.model, device=self.device
        )
        if self.use_cuda_graph and self.device == "cuda":
            self.cuda_graph_runner.capture(
                self.kv_cache_pool, batch_size=1, warmup_iters=3
            )

    # ================= V2 Continuous Batching API =================

    def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> int:
        """Adds a new request to the engine's waiting queue."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        seq_id = self._next_seq_id
        self._next_seq_id += 1

        prompt_token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(
            seq_id=seq_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            block_size=self.block_size,
            device=self.device,
        )
        self._all_sequences[seq_id] = seq
        self.scheduler.add_sequence(seq)
        return seq_id

    @torch.no_grad()
    def step(self) -> List[RequestOutput]:
        """
        Executes one iteration-level scheduling decision and forward step.
        """
        outputs: List[RequestOutput] = []
        sched_out = self.scheduler.schedule()
        if sched_out.is_empty():
            return outputs

        # 1. Prefill Step (Single Sequence)
        if sched_out.prefill_seqs:
            seq = sched_out.prefill_seqs[0]
            prompt_len = len(seq.prompt_token_ids)
            slot_mapping = seq.block_table.allocate_for_prompt(
                prompt_len, self.block_allocator
            )

            context = ForwardContext(
                kv_cache_pool=self.kv_cache_pool,
                is_prefill=True,
                slot_mapping=slot_mapping,
                total_seq_len=prompt_len,
            )

            input_tensor = torch.tensor(
                [seq.prompt_token_ids], device=self.device, dtype=torch.long
            )
            logits = self.model(input_tensor, context=context)
            last_logits = logits[:, -1, :]

            next_token_id = self.sampler.sample(
                last_logits,
                params=seq.sampling_params,
                generated_tokens=seq.output_token_ids,
            )
            seq.append_token_id(next_token_id)

            stop_ids = set(seq.sampling_params.stop_token_ids)
            if self.tokenizer.eos_token_id is not None:
                stop_ids.add(self.tokenizer.eos_token_id)

            is_finished = (next_token_id in stop_ids) or (
                seq.output_len >= seq.sampling_params.max_new_tokens
            )

            if is_finished:
                self.scheduler.free_sequence(seq)

            text = self.tokenizer.decode(
                seq.output_token_ids, skip_special_tokens=True
            )
            outputs.append(
                RequestOutput(
                    seq_id=seq.seq_id,
                    prompt=seq.prompt,
                    text=text,
                    token_ids=list(seq.output_token_ids),
                    finished=is_finished,
                )
            )
            return outputs

        # 2. Multi-Sequence Batch Decode Step
        if sched_out.decode_seqs:
            decode_seqs = sched_out.decode_seqs
            bsz = len(decode_seqs)

            batch_current_slots = []
            batch_block_tables = []
            batch_total_seq_lens = []
            batch_input_ids = []

            for seq in decode_seqs:
                target_slot = seq.block_table.append_token(self.block_allocator)
                batch_current_slots.append(target_slot)
                batch_block_tables.append(seq.block_table.gpu_block_ids)
                seq_total_len = seq.total_len + 1
                batch_total_seq_lens.append(seq_total_len)
                batch_input_ids.append([seq.get_last_token_id()])

            context = ForwardContext(
                kv_cache_pool=self.kv_cache_pool,
                is_prefill=False,
                batch_current_slots=batch_current_slots,
                batch_block_tables=batch_block_tables,
                batch_total_seq_lens=batch_total_seq_lens,
            )

            input_tensor = torch.tensor(
                batch_input_ids, device=self.device, dtype=torch.long
            )
            logits = self.model(input_tensor, context=context)

            stop_ids = set()
            if self.tokenizer.eos_token_id is not None:
                stop_ids.add(self.tokenizer.eos_token_id)

            for b, seq in enumerate(decode_seqs):
                last_logits = logits[b, 0, :]
                next_token_id = self.sampler.sample(
                    last_logits,
                    params=seq.sampling_params,
                    generated_tokens=seq.output_token_ids,
                )
                seq.append_token_id(next_token_id)

                seq_stop_ids = stop_ids.union(
                    seq.sampling_params.stop_token_ids
                )
                is_finished = (next_token_id in seq_stop_ids) or (
                    seq.output_len >= seq.sampling_params.max_new_tokens
                )

                if is_finished:
                    self.scheduler.free_sequence(seq)

                text = self.tokenizer.decode(
                    seq.output_token_ids, skip_special_tokens=True
                )
                outputs.append(
                    RequestOutput(
                        seq_id=seq.seq_id,
                        prompt=seq.prompt,
                        text=text,
                        token_ids=list(seq.output_token_ids),
                        finished=is_finished,
                    )
                )

        return outputs

    def generate_batch(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """Runs Continuous Batching generation for a list of prompts."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        seq_ids = [self.add_request(p, sampling_params) for p in prompts]
        completed: Dict[int, RequestOutput] = {}

        while self.scheduler.has_unfinished_seqs():
            step_outputs = self.step()
            for ro in step_outputs:
                if ro.finished:
                    completed[ro.seq_id] = ro

        return [completed[sid] for sid in seq_ids]

    # ================= V0 / V1 / V3 Single-Sequence Stream API =================

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        use_kv_cache: bool = True,
    ) -> Iterator[str]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        prompt_len = input_ids.shape[-1]
        generated_token_ids: List[int] = []

        stop_token_ids = set(sampling_params.stop_token_ids)
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.add(self.tokenizer.eos_token_id)

        if not use_kv_cache:
            # V0 Naive Fallback
            curr_sequence = input_ids
            for _ in range(sampling_params.max_new_tokens):
                logits = self.model(curr_sequence)
                last_token_logits = logits[:, -1, :]
                next_token_id = self.sampler.sample(
                    last_token_logits,
                    params=sampling_params,
                    generated_tokens=generated_token_ids,
                )
                if next_token_id in stop_token_ids:
                    break
                generated_token_ids.append(next_token_id)
                yield self.tokenizer.decode(
                    [next_token_id], skip_special_tokens=True
                )
                next_tensor = torch.tensor(
                    [[next_token_id]], device=self.device, dtype=torch.long
                )
                curr_sequence = torch.cat([curr_sequence, next_tensor], dim=-1)
            return

        # V1/V3 Paged KV Cache Single-sequence Generation
        block_table = BlockTable(block_size=self.block_size, device=self.device)
        try:
            slot_mapping = block_table.allocate_for_prompt(
                prompt_len, self.block_allocator
            )
            prefill_context = ForwardContext(
                kv_cache_pool=self.kv_cache_pool,
                is_prefill=True,
                slot_mapping=slot_mapping,
                total_seq_len=prompt_len,
            )
            logits = self.model(input_ids, context=prefill_context)
            last_token_logits = logits[:, -1, :]

            next_token_id = self.sampler.sample(
                last_token_logits,
                params=sampling_params,
                generated_tokens=generated_token_ids,
            )

            total_seq_len = prompt_len
            if next_token_id not in stop_token_ids:
                generated_token_ids.append(next_token_id)
                yield self.tokenizer.decode(
                    [next_token_id], skip_special_tokens=True
                )

                for _ in range(sampling_params.max_new_tokens - 1):
                    total_seq_len += 1
                    target_slot = block_table.append_token(self.block_allocator)

                    decode_context = ForwardContext(
                        kv_cache_pool=self.kv_cache_pool,
                        is_prefill=False,
                        block_table=block_table.gpu_block_ids,
                        current_slot=target_slot,
                        total_seq_len=total_seq_len,
                        seq_lens_tensor=torch.tensor([total_seq_len], device=self.device, dtype=torch.int32),
                    )
                    single_input = torch.tensor(
                        [[next_token_id]], device=self.device, dtype=torch.long
                    )

                    if (
                        self.use_cuda_graph
                        and self.cuda_graph_runner.is_captured
                    ):
                        pos = torch.tensor(
                            [[total_seq_len - 1]],
                            device=self.device,
                            dtype=torch.long,
                        )
                        logits = self.cuda_graph_runner.replay(
                            single_input, pos, decode_context
                        )
                    else:
                        logits = self.model(
                            single_input, context=decode_context
                        )

                    last_token_logits = logits[:, -1, :]
                    next_token_id = self.sampler.sample(
                        last_token_logits,
                        params=sampling_params,
                        generated_tokens=generated_token_ids,
                    )
                    if next_token_id in stop_token_ids:
                        break

                    generated_token_ids.append(next_token_id)
                    yield self.tokenizer.decode(
                        [next_token_id], skip_special_tokens=True
                    )
        finally:
            block_table.free_all(self.block_allocator)

    def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        use_kv_cache: bool = True,
    ) -> str:
        chunks = list(
            self.stream_generate(
                prompt, sampling_params=sampling_params, use_kv_cache=use_kv_cache
            )
        )
        return "".join(chunks)
