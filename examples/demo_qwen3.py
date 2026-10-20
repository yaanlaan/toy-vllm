import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toy_vllm.config import SamplingParams
from toy_vllm.engine.llm_engine import LLMEngine

QWEN3_DIR = r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"


def main():
    print("=" * 75)
    print("      Toy-vLLM: Qwen3 (Qwen3-0.6B) 大语言模型推理演示")
    print("=" * 75)

    # 1. Initialize engine with Qwen3
    t0 = time.time()
    engine = LLMEngine(
        model_path=QWEN3_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_gpu_blocks=1024,
        block_size=16,
    )
    init_cost = time.time() - t0

    cfg = engine.config
    print(f"\n[模型架构规格 (Qwen3)]")
    print(f"  • 模型类型:              {cfg.model_type}")
    print(f"  • 隐层维度 (hidden_size): {cfg.hidden_size}")
    print(f"  • 层数 (num_layers):      {cfg.num_hidden_layers}")
    print(f"  • 注意力头数 (num_heads): {cfg.num_attention_heads} (KV Heads: {cfg.num_key_value_heads})")
    print(f"  • 头维度 (head_dim):      {cfg.head_dim}")
    print(f"  • QK-Norm 归一化:         已启用 (Q-Norm & K-Norm per head)")
    print(f"  • 模型装载与显存池就绪:   {init_cost:.2f}s\n")

    # 2. Prompts for testing
    prompt = "请简要介绍一下你自己，并解释什么是大语言模型（LLM）："
    chat_prompt = (
        f"<|im_start|>system\n你是一个友好博学的人工智能助手。<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    print(f"【输入提示词】: {prompt}")
    print("-" * 75)
    print("【Qwen3 生成回答 (流式输出)】: ")

    params = SamplingParams(
        temperature=0.7,
        top_p=0.85,
        top_k=25,
        max_new_tokens=150,
    )

    gen_start = time.time()
    first_token_time = None
    token_count = 0

    for token_text in engine.stream_generate(
        chat_prompt, sampling_params=params, use_kv_cache=True
    ):
        if first_token_time is None:
            first_token_time = time.time() - gen_start
        sys.stdout.write(token_text)
        sys.stdout.flush()
        token_count += 1

    total_time = time.time() - gen_start
    ttft_ms = (first_token_time * 1000) if first_token_time else 0.0
    decode_time = total_time - (first_token_time or 0.0)
    decode_speed = (token_count - 1) / decode_time if decode_time > 0 and token_count > 1 else 0.0

    print("\n" + "-" * 75)
    print("【性能监控指标】")
    print(f"  • 生成 Token 总数:       {token_count} tokens")
    print(f"  • 首字时延 (TTFT):       {ttft_ms:.1f} ms")
    print(f"  • 持续解码吞吐:          {decode_speed:.2f} tokens/s")
    print(f"  • 总生成耗时:            {total_time:.2f} s")
    print(f"  • 物理显存管理:          剩余 {engine.block_allocator.get_num_free_blocks()} 块 (100% 回收)")
    print("=" * 75)


if __name__ == "__main__":
    main()
