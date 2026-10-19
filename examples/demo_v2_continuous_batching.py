import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toy_vllm.config import SamplingParams
from toy_vllm.engine.llm_engine import LLMEngine

MODEL_DIR = r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    print("=" * 75)
    print("    Toy-vLLM V2 性能压测: 单请求串行排队 vs 连续批处理 (Continuous Batching)")
    print("=" * 75)

    engine = LLMEngine(model_path=MODEL_DIR, device="cuda")

    # 8 different user prompts with varying response lengths
    raw_prompts = [
        "1 + 1 等于几？简短回答。",
        "简要介绍什么是勾股定理？",
        "列举常见的三种排序算法名称：",
        "什么是操作系统的虚拟内存？简要回答。",
        "地球的卫星叫什么名字？简短回答。",
        "TCP 和 UDP 的核心区别是什么？简要概括。",
        "什么是 Python 中的全局解释器锁 (GIL)？",
        "太阳系中最大的行星是哪一颗？",
    ]

    chat_prompts = [
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        for p in raw_prompts
    ]

    params = SamplingParams(
        temperature=0.0,
        max_new_tokens=40,
    )

    num_reqs = len(chat_prompts)
    print(f"\n[测试任务] 共有 {num_reqs} 个长短不一的用户并发请求需要处理。")
    print("-" * 75)

    # ================= 1. Sequential Mode (No Batching) =================
    print(">>> 正在运行 方式 1: 串行排队处理 (Sequential Processing)...")
    seq_start = time.time()
    seq_total_tokens = 0
    seq_results = []
    for p in chat_prompts:
        output_text = engine.generate(p, sampling_params=params)
        tokens = engine.tokenizer.encode(output_text)
        seq_total_tokens += len(tokens)
        seq_results.append(output_text)
    seq_time = time.time() - seq_start
    seq_throughput = seq_total_tokens / seq_time
    print(
        f"    [串行排队结果] 生成总 Tokens: {seq_total_tokens} | 耗时: {seq_time:.2f}s | 吞吐: {seq_throughput:.2f} tokens/s"
    )

    # ================= 2. Continuous Batching Mode =================
    print(
        "\n>>> 正在运行 方式 2: 连续批处理 (Continuous Batching, 并发调度)..."
    )
    cb_start = time.time()
    cb_outputs = engine.generate_batch(chat_prompts, sampling_params=params)
    cb_time = time.time() - cb_start
    cb_total_tokens = sum(len(ro.token_ids) for ro in cb_outputs)
    cb_throughput = cb_total_tokens / cb_time
    print(
        f"    [连续批处理结果] 生成总 Tokens: {cb_total_tokens} | 耗时: {cb_time:.2f}s | 吞吐: {cb_throughput:.2f} tokens/s"
    )

    # ================= 3. Comparison Summary =================
    throughput_boost = cb_throughput / seq_throughput
    time_reduction = (1 - cb_time / seq_time) * 100

    print("=" * 75)
    print("                         多并发压测结论汇总")
    print("=" * 75)
    print(
        f"  • 串行排队总耗时:        {seq_time:.2f}s  (系统总吞吐: {seq_throughput:.2f} tokens/s)"
    )
    print(
        f"  • 连续批处理总耗时:      {cb_time:.2f}s  (系统总吞吐: {cb_throughput:.2f} tokens/s)"
    )
    print(
        f"  • 响应耗时缩减:          \033[1;32m{time_reduction:.1f}%\033[0m"
    )
    print(
        f"  • 系统服务吞吐跃升:      \033[1;32m{throughput_boost:.2f}x\033[0m"
    )
    print("=" * 75)

    print("\n[展示前 3 个生成样例]")
    for i in range(3):
        print(f"[{i+1}] Prompt: {raw_prompts[i]}")
        print(f"    Answer: {cb_outputs[i].text.strip()}\n")


if __name__ == "__main__":
    main()
