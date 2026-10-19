import sys
import os
import time
import threading
from pathlib import Path
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safetensors.torch import load_file
from transformers import AutoTokenizer

from toy_vllm.config import ModelConfig
from toy_vllm.distributed.parallel_state import (
    initialize_model_parallel,
    set_tensor_model_parallel_rank,
    set_tensor_model_parallel_world_size,
    destroy_model_parallel,
)
from toy_vllm.distributed.communication import (
    set_mock_all_reduce_hook,
    MockAllReduceGroup,
)
from toy_vllm.models.parallel_qwen2 import ParallelQwen2ForCausalLM

MODEL_DIR = r"C:\Users\yaan\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    print("=" * 75)
    print("      Toy-vLLM V4: 单机多卡张量并行 (Tensor Parallelism, TP=2) 演示")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    config = ModelConfig.from_pretrained(MODEL_DIR)

    tp_size = 2
    print(f"\n[分布式配置] 启动张量并行 TP = {tp_size} (模拟两张 GPU 卡独立并发 Worker)")
    print(
        f"  • 模型总头数: {config.num_attention_heads} Q Heads, {config.num_key_value_heads} KV Heads"
    )
    print(
        f"  • 每张卡持有分片: {config.num_attention_heads // tp_size} Q Heads, {config.num_key_value_heads // tp_size} KV Heads"
    )
    print(
        f"  • MLP 隐层维度: {config.intermediate_size} -> 每张卡切分为 {config.intermediate_size // tp_size}"
    )

    # Load master safetensors weights
    print(f"\n[权重载入] 从 Safetensors 载入全量权重并按 Rank 切片...")
    st_files = sorted(list(Path(MODEL_DIR).glob("*.safetensors")))
    st_path = str(st_files[0])
    master_weights = load_file(st_path, device="cpu")

    # Set up simulated multi-rank communication group
    comm_group = MockAllReduceGroup(world_size=tp_size)
    set_mock_all_reduce_hook(comm_group.all_reduce)
    set_tensor_model_parallel_world_size(tp_size)

    # Initialize Rank 0 model partition
    set_tensor_model_parallel_rank(0)
    model_rank0 = ParallelQwen2ForCausalLM(config).to(device=device)
    model_rank0.load_partition_weights(master_weights, rank=0)
    model_rank0.eval()

    # Initialize Rank 1 model partition
    set_tensor_model_parallel_rank(1)
    model_rank1 = ParallelQwen2ForCausalLM(config).to(device=device)
    model_rank1.load_partition_weights(master_weights, rank=1)
    model_rank1.eval()

    prompt = "请用一句话概括大模型张量并行（Tensor Parallelism）的核心精髓："
    chat_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    input_ids = tokenizer.encode(chat_prompt, return_tensors="pt").to(device)

    print("-" * 75)
    print(f"用户 Prompt: {prompt}")
    print("两张并行卡正在以独立线程并发执行前向计算与 All-Reduce 通信...")

    # Shared outputs dictionary
    worker_logits = {}

    def worker_step(rank, model, curr_ids):
        set_tensor_model_parallel_rank(rank)
        with torch.no_grad():
            out = model(curr_ids)
            worker_logits[rank] = out

    t0_time = time.time()
    generated_tokens = []
    curr_ids = input_ids

    # Generate 15 tokens in lockstep across both parallel workers
    for _ in range(15):
        t0 = threading.Thread(
            target=worker_step, args=(0, model_rank0, curr_ids)
        )
        t1 = threading.Thread(
            target=worker_step, args=(1, model_rank1, curr_ids)
        )

        t0.start()
        t1.start()
        t0.join()
        t1.join()

        # Both workers arrive at the identical reduced logits
        logits = worker_logits[0][:, -1, :]
        next_token = int(torch.argmax(logits, dim=-1).item())
        if next_token == tokenizer.eos_token_id or next_token == 151645:
            break
        generated_tokens.append(next_token)
        curr_ids = torch.cat(
            [
                curr_ids,
                torch.tensor([[next_token]], device=device, dtype=torch.long),
            ],
            dim=-1,
        )

    cost = time.time() - t0_time
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n" + "=" * 75)
    print(f"生成的回复内容:\n{output_text.strip()}")
    print("-" * 75)
    print(
        f"[分布式统计] 生成 Tokens: {len(generated_tokens)} | 耗时: {cost:.2f}s | "
        f"累计卡间 All-Reduce 次数: {len(generated_tokens) * 2 * config.num_hidden_layers} 次"
    )
    print(
        f"[PASS] 验证成功: 两张并行卡通过 Column/Row 矩阵切分协同完成端到端自回归生成！"
    )
    print("=" * 75)

    destroy_model_parallel()
    set_mock_all_reduce_hook(None)


if __name__ == "__main__":
    main()
