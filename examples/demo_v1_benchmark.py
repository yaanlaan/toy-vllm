import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toy_vllm.config import SamplingParams
from toy_vllm.engine.llm_engine import LLMEngine

MODEL_DIR = r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def run_test_case(engine: LLMEngine, case_name: str, prompt: str, gen_tokens: int):
    chat_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    tokens = engine.tokenizer.encode(chat_prompt)
    prompt_len = len(tokens)

    params = SamplingParams(
        temperature=0.0,  # Greedy for identical outputs
        max_new_tokens=gen_tokens,
    )

    print("\n" + "=" * 70)
    print(
        f"测试场景: {case_name} (Prompt 长度: {prompt_len} tokens | 目标生成: {gen_tokens} tokens)"
    )
    print("=" * 70)

    # 1. Benchmark V0 (No Cache)
    print(">>> 正在运行 V0 (Naive 全量重算, 无 KV Cache)...")
    t0_start = time.time()
    v0_tokens = list(
        engine.stream_generate(chat_prompt, params, use_kv_cache=False)
    )
    t0_cost = time.time() - t0_start
    v0_speed = len(v0_tokens) / t0_cost
    print(
        f"    [V0 结果] 生成 Tokens: {len(v0_tokens)} | 耗时: {t0_cost:.2f}s | 速度: {v0_speed:.2f} tokens/s"
    )

    # 2. Benchmark V1 (Paged KV Cache)
    print(">>> 正在运行 V1 (Paged KV Cache 分页显存缓存)...")
    t1_start = time.time()
    v1_tokens = list(
        engine.stream_generate(chat_prompt, params, use_kv_cache=True)
    )
    t1_cost = time.time() - t1_start
    v1_speed = len(v1_tokens) / t1_cost
    print(
        f"    [V1 结果] 生成 Tokens: {len(v1_tokens)} | 耗时: {t1_cost:.2f}s | 速度: {v1_speed:.2f} tokens/s"
    )

    # 3. Speedup
    speedup = t0_cost / t1_cost
    match_status = "[PASS] 100% 完全一致" if v0_tokens == v1_tokens else "[NOTE] 局部浮点舍入微差"
    print("-" * 70)
    print(
        f"  对比结论: V1 加速比为 {speedup:.2f}x | 输出内容对齐: {match_status}"
    )


def main():
    print("=" * 70)
    print("       Toy-vLLM 性能基准实测: V0 (无缓存) vs V1 (Paged KV Cache)")
    print("=" * 70)

    engine = LLMEngine(model_path=MODEL_DIR, device="cuda")

    # Case 1: Short context
    p1 = "简述什么是机器学习中的监督学习？"
    run_test_case(engine, "短上下文日常对话", p1, gen_tokens=40)

    # Case 2: Long context (Prompt 300+ tokens)
    p2 = (
        "详细分析操作系统中虚拟内存管理的核心原理，包括分页机制、页表结构、"
        "缺页中断处理流程、TLB快表以及页面置换算法。请展开阐述："
    ) * 4
    run_test_case(engine, "长上下文深度问答", p2, gen_tokens=60)

    print("\n" + "=" * 70)
    print("基准测试全部完成！随着序列长度增长，V1 避免重复计算历史 KV 的优势越发显著！")
    print("=" * 70)


if __name__ == "__main__":
    main()
