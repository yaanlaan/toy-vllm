import pytest
from toy_vllm.config import SamplingParams
from toy_vllm.core.block_manager import BlockAllocator
from toy_vllm.core.sequence import Sequence, SequenceStatus
from toy_vllm.core.scheduler import Scheduler
from toy_vllm.engine.llm_engine import LLMEngine

MODEL_DIR = r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def test_scheduler_logic():
    """Verify Scheduler queue transitions, capacity control, and memory reclamation."""
    allocator = BlockAllocator(
        num_blocks=5, block_size=16
    )  # Only 5 blocks total
    scheduler = Scheduler(block_allocator=allocator, max_num_seqs=4)

    # Prompt of 30 tokens requires ceil(30/16) = 2 blocks
    params = SamplingParams(max_new_tokens=10)
    seq0 = Sequence(
        seq_id=0,
        prompt="p0",
        prompt_token_ids=list(range(30)),
        sampling_params=params,
        block_size=16,
        device="cpu",
    )
    seq1 = Sequence(
        seq_id=1,
        prompt="p1",
        prompt_token_ids=list(range(30)),
        sampling_params=params,
        block_size=16,
        device="cpu",
    )
    seq2 = Sequence(
        seq_id=2,
        prompt="p2",
        prompt_token_ids=list(range(30)),
        sampling_params=params,
        block_size=16,
        device="cpu",
    )

    scheduler.add_sequence(seq0)
    scheduler.add_sequence(seq1)
    scheduler.add_sequence(seq2)

    assert len(scheduler.waiting) == 3
    assert len(scheduler.running) == 0

    # Step 1: Schedule first prefill (seq0 admitted)
    out1 = scheduler.schedule()
    assert len(out1.prefill_seqs) == 1
    assert out1.prefill_seqs[0].seq_id == 0
    # Simulate allocating 2 blocks for seq0 prompt
    seq0.block_table.allocate_for_prompt(30, allocator)
    assert allocator.get_num_free_blocks() == 3

    # Step 2: Next schedule: RUNNING seq0 has priority for decode
    out2 = scheduler.schedule()
    assert len(out2.decode_seqs) == 1
    assert out2.decode_seqs[0].seq_id == 0

    # Finish seq0 and free
    scheduler.free_sequence(seq0)
    assert allocator.get_num_free_blocks() == 5
    assert len(scheduler.running) == 0

    # Step 3: Now seq1 can be admitted from WAITING
    out3 = scheduler.schedule()
    assert len(out3.prefill_seqs) == 1
    assert out3.prefill_seqs[0].seq_id == 1
    scheduler.free_sequence(out3.prefill_seqs[0])

    print("[PASS] Scheduler queuing and state transition logic passed!")


def test_continuous_batching_execution():
    """Verify end-to-end multi-request batch generation with dynamic early termination."""
    engine = LLMEngine(model_path=MODEL_DIR, device="cuda")

    prompts = [
        "1 + 1 等于几？简短回答。",
        "简要介绍什么是勾股定理？",
        "列举计算机科学中常见的三种排序算法名称：",
    ]

    # Format prompts with Qwen chat template
    chat_prompts = [
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        for p in prompts
    ]

    params = SamplingParams(
        temperature=0.0,
        max_new_tokens=30,
    )

    print("\nRunning generate_batch with 3 concurrent requests...")
    outputs = engine.generate_batch(chat_prompts, sampling_params=params)

    assert len(outputs) == 3
    for ro in outputs:
        assert ro.finished is True
        assert len(ro.token_ids) > 0
        print(f"Request {ro.seq_id} (Generated {len(ro.token_ids)} tokens):")
        print(f"Text: {ro.text.strip()}\n")

    # Assert complete memory cleanup
    assert (
        engine.block_allocator.get_num_free_blocks() == engine.num_gpu_blocks
    ), "Memory leak detected in Continuous Batching!"
    print(
        "[PASS] Continuous Batching multi-request generation passed with zero leaks!"
    )


if __name__ == "__main__":
    test_scheduler_logic()
    test_continuous_batching_execution()
    print("\n>>> All V2 Continuous Batching tests passed successfully! <<<")
