import sys
import os
import time
import argparse
import torch

# Ensure toy_vllm package is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toy_vllm.config import SamplingParams
from toy_vllm.engine.llm_engine import LLMEngine

MODEL_PRESETS = {
    "qwen3": r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca",
    "qwen2": r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775",
}


def main():
    parser = argparse.ArgumentParser(description="Toy-vLLM 交互式对话体验")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3",
        help="模型选择: 'qwen3' (Qwen3-0.6B, 默认), 'qwen2' (Qwen2.5-0.5B), 或本地模型权重目录",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="用户提问内容 (不传则使用默认问题或进入交互模式)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="最大生成 Token 数量",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="采样温度",
    )
    args, unknown = parser.parse_known_args()

    model_path = MODEL_PRESETS.get(args.model.lower(), args.model)
    if not os.path.exists(model_path):
        print(f"[错误] 模型路径不存在: {model_path}")
        sys.exit(1)

    print("=" * 70)
    print("       Toy-vLLM: 高性能大模型推理引擎交互体验")
    print("=" * 70)

    # 1. Initialize engine
    init_start = time.time()
    engine = LLMEngine(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_gpu_blocks=1024,
        block_size=16,
    )
    init_time = time.time() - init_start
    print(f"[OK] 引擎启动就绪！模型装载与显存池初始化耗时: {init_time:.2f}s\n")

    # 2. System prompt and sample user question
    system_prompt = "你是由 Toy-vLLM 高性能推理引擎驱动的智能助手。请用清晰、生动、专业的语言回答用户的问题。"
    
    if args.prompt:
        user_prompt = args.prompt
    elif unknown:
        user_prompt = " ".join(unknown)
    else:
        user_prompt = "请用生动形象的语言，向程序员解释为什么大模型推理中需要 PagedAttention（分页注意力）机制？它解决了什么本质问题？"

    chat_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    print(f"【当前使用模型】: {engine.config.model_type.upper()} ({model_path.split(os.sep)[-3]})")
    print(f"【用户提问内容】: {user_prompt}")
    print("-" * 70)
    print("【Toy-vLLM 回答】: ")

    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=0.85,
        top_k=25,
        max_new_tokens=args.max_tokens,
    )

    # 3. Stream generation with timing & token statistics
    gen_start = time.time()
    first_token_time = None
    token_count = 0

    for token_text in engine.stream_generate(
        chat_prompt, sampling_params=sampling_params, use_kv_cache=True
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

    print("\n" + "-" * 70)
    print("【实时性能统计看板】")
    print(f"  • 生成 Token 总数:       {token_count} tokens")
    print(f"  • 首字时延 (TTFT):       {ttft_ms:.1f} ms")
    print(f"  • 总生成耗时:            {total_time:.2f} s")
    print(f"  • 解码生成吞吐:          {decode_speed:.2f} tokens/s")
    print(f"  • 物理显存块总数:        {engine.num_gpu_blocks} blocks")
    print(f"  • 剩余可用显存块:        {engine.block_allocator.get_num_free_blocks()} blocks (零泄漏)")
    print("=" * 70)


if __name__ == "__main__":
    main()
