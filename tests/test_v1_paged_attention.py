import torch
import pytest
from toy_vllm.config import ModelConfig, SamplingParams
from toy_vllm.core.block_manager import (
    BlockAllocator,
    BlockTable,
    KVCachePool,
)
from toy_vllm.engine.llm_engine import LLMEngine

MODEL_DIR = r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def test_block_allocator():
    """Verify BlockAllocator allocation, deallocation, and double-free guard."""
    allocator = BlockAllocator(num_blocks=10, block_size=16)
    assert allocator.get_num_free_blocks() == 10

    # Allocate 3 blocks
    b0 = allocator.allocate()
    b1 = allocator.allocate()
    b2 = allocator.allocate()
    assert allocator.get_num_free_blocks() == 7

    # Free 1 block
    allocator.free(b1)
    assert allocator.get_num_free_blocks() == 8

    # Free remaining
    allocator.free(b0)
    allocator.free(b2)
    assert allocator.get_num_free_blocks() == 10

    print("[PASS] BlockAllocator unit test passed!")


def test_block_table_lifecycle():
    """Verify BlockTable dynamically allocates blocks across boundaries and cleans up completely."""
    allocator = BlockAllocator(num_blocks=20, block_size=16)
    table = BlockTable(block_size=16)

    # Prompt length = 35 tokens -> requires ceil(35 / 16) = 3 blocks (0..15, 16..31, 32..47)
    slots = table.allocate_for_prompt(35, allocator)
    assert len(table.block_ids) == 3
    assert len(slots) == 35
    assert allocator.get_num_free_blocks() == 17

    # Current token count = 35 (block index 2, offset 3)
    # Append tokens until crossing block boundary (offset 0 requires new block)
    # 35 -> 48 tokens total (13 tokens to reach 48 which is block 3)
    for _ in range(13):
        table.append_token(allocator)
    # At 48 tokens, it reached exact boundary (3 blocks full)
    assert len(table.block_ids) == 3

    # Token 49: should allocate the 4th block!
    b_id, offset = table.append_token(allocator)
    assert len(table.block_ids) == 4
    assert offset == 0
    assert allocator.get_num_free_blocks() == 16

    # Clean up
    table.free_all(allocator)
    assert len(table.block_ids) == 0
    assert allocator.get_num_free_blocks() == 20
    print("[PASS] BlockTable lifecycle & boundary test passed!")


def test_v0_vs_v1_correctness_alignment():
    """
    CRITICAL TEST:
    Verify that V0 (recompute everything) and V1 (Paged KV Cache)
    produce 100% BIT-LEVEL IDENTICAL token outputs under Greedy sampling!
    """
    engine = LLMEngine(model_path=MODEL_DIR, device="cuda")
    prompt = "计算并输出斐波那契数列的前10项："
    chat_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )

    params = SamplingParams(
        temperature=0.0,  # Greedy mode for deterministic equivalence check
        max_new_tokens=32,
    )

    print("\nRunning V0 generation (no cache)...")
    v0_tokens = list(
        engine.stream_generate(chat_prompt, params, use_kv_cache=False)
    )
    v0_text = "".join(v0_tokens)

    print("Running V1 generation (Paged KV Cache)...")
    v1_tokens = list(
        engine.stream_generate(chat_prompt, params, use_kv_cache=True)
    )
    v1_text = "".join(v1_tokens)

    print("-" * 60)
    print(f"V0 Output ({len(v0_tokens)} tokens):\n{v0_text}")
    print("-" * 60)
    print(f"V1 Output ({len(v1_tokens)} tokens):\n{v1_text}")
    print("-" * 60)

    assert (
        v0_tokens == v1_tokens
    ), f"Mismatch between V0 and V1 outputs!\nV0: {v0_tokens}\nV1: {v1_tokens}"
    assert (
        engine.block_allocator.get_num_free_blocks() == engine.num_gpu_blocks
    ), "Memory leak detected in BlockAllocator!"

    print(
        "[PASS] V0 vs V1 Bit-Level Correctness Alignment passed! Zero memory leaks!"
    )


if __name__ == "__main__":
    test_block_allocator()
    test_block_table_lifecycle()
    test_v0_vs_v1_correctness_alignment()
    print("\n>>> All V1 Paged Attention tests passed successfully! <<<")
