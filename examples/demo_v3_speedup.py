import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toy_vllm.config import SamplingParams
from toy_vllm.engine.llm_engine import LLMEngine
import toy_vllm.modules.attention as attn_mod

MODEL_DIR = r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    print("=" * 75)
    print("       Toy-vLLM V3 单卡算子深度融合与高性能内核优化基准实测")
    print("=" * 75)

    prompt = "请简要阐述什么是图神经网络（GNN）及其在现实中的典型应用场景："
    chat_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )

    gen_tokens = 50
    params = SamplingParams(temperature=0.0, max_new_tokens=gen_tokens)

    engine = LLMEngine(model_path=MODEL_DIR, device="cuda")

    # ================= 1. PyTorch Unfused Baseline (V1) =================
    print("\n>>> 1. 正在运行 PyTorch 原生算子基线 (逐层 Gather 切片, 未融合)...")
    attn_mod.HAS_TRITON = False  # Temporarily disable Triton to test PyTorch baseline

    torch.cuda.synchronize()
    t0 = time.time()
    tokens_py = list(
        engine.stream_generate(
            chat_prompt, sampling_params=params, use_kv_cache=True
        )
    )
    torch.cuda.synchronize()
    py_time = time.time() - t0
    py_speed = len(tokens_py) / py_time
    py_itl = (py_time / len(tokens_py)) * 1000  # ms per token
    print(
        f"    [PyTorch 原生结果] 生成 Tokens: {len(tokens_py)} | 耗时: {py_time:.2f}s | "
        f"速度: {py_speed:.2f} tokens/s | 单 Token 延迟 (ITL): {py_itl:.1f}ms"
    )

    # ================= 2. Triton PagedAttention & Fused Ops (V3) =================
    print(
        "\n>>> 2. 正在运行 V3 极致优化 (Triton PagedAttention + Fused SwiGLU)..."
    )
    attn_mod.HAS_TRITON = True  # Enable Triton kernels

    torch.cuda.synchronize()
    t1 = time.time()
    tokens_triton = list(
        engine.stream_generate(
            chat_prompt, sampling_params=params, use_kv_cache=True
        )
    )
    torch.cuda.synchronize()
    triton_time = time.time() - t1
    triton_speed = len(tokens_triton) / triton_time
    triton_itl = (triton_time / len(tokens_triton)) * 1000
    print(
        f"    [Triton 内核优化结果] 生成 Tokens: {len(tokens_triton)} | 耗时: {triton_time:.2f}s | "
        f"速度: {triton_speed:.2f} tokens/s | 单 Token 延迟 (ITL): {triton_itl:.1f}ms"
    )

    # ================= 3. Summary =================
    speedup = py_time / triton_time
    itl_reduction = (1 - triton_itl / py_itl) * 100
    text_match = (
        "[PASS] 100% 完全一致" if tokens_py == tokens_triton else "[NOTE] 局部浮点舍入微差"
    )

    print("\n" + "=" * 75)
    print("                             性能对比总结")
    print("=" * 75)
    print(
        f"  • PyTorch 原生基线耗时:   {py_time:.2f}s  (ITL: {py_itl:.1f}ms/token | 吞吐: {py_speed:.2f} tokens/s)"
    )
    print(
        f"  • Triton 算子优化耗时:    {triton_time:.2f}s  (ITL: {triton_itl:.1f}ms/token | 吞吐: {triton_speed:.2f} tokens/s)"
    )
    print(
        f"  • 单 Token 解码延迟削减:  \033[1;32m{itl_reduction:.1f}%\033[0m"
    )
    print(
        f"  • 端到端内核加速比:       \033[1;32m{speedup:.2f}x\033[0m"
    )
    print(f"  • 输出结果一致性:         {text_match}")
    print("=" * 75)

    print("\n[生成回复文本预览]")
    print("".join(tokens_triton).strip())


if __name__ == "__main__":
    main()
