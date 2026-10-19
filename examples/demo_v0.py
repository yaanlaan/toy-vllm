import sys
import os
import time

# Ensure toy_vllm is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toy_vllm.config import SamplingParams
from toy_vllm.engine.llm_engine import LLMEngine

MODEL_DIR = r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    print("=" * 60)
    print("      Toy-vLLM V0: 极简自回归大模型推理引擎 Demo")
    print("=" * 60)

    start_time = time.time()
    # 1. Initialize engine and load model weights
    engine = LLMEngine(model_path=MODEL_DIR, device="cuda")
    load_time = time.time() - start_time
    print(f"[toy-vllm] Model loaded in {load_time:.2f}s\n")

    # 2. Test prompt
    prompt = "简要介绍一下什么是大模型推理框架中的自回归生成机制？"
    # Format with Qwen chat template
    chat_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )

    print(f"Prompt: {prompt}")
    print("-" * 60)
    print("生成回答 (Streaming 流式生成): ")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=128,
    )

    gen_start = time.time()
    token_count = 0

    for token_text in engine.stream_generate(
        chat_prompt, sampling_params=sampling_params
    ):
        sys.stdout.write(token_text)
        sys.stdout.flush()
        token_count += 1

    gen_time = time.time() - gen_start
    print("\n" + "-" * 60)
    print(
        f"[统计] 生成 Token 数量: {token_count} | 耗时: {gen_time:.2f}s | 速度: {token_count / gen_time:.2f} tokens/s"
    )
    print(
        f"[提示] V0 采用无 KV Cache 的全序列前向重算，随长度增加每步耗时呈 O(N) 线性递增。"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
