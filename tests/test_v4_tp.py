import threading
import torch
import pytest
from toy_vllm.config import ModelConfig
from toy_vllm.distributed.parallel_state import (
    initialize_model_parallel,
    destroy_model_parallel,
    set_tensor_model_parallel_rank,
    set_tensor_model_parallel_world_size,
)
from toy_vllm.distributed.communication import (
    set_mock_all_reduce_hook,
    MockAllReduceGroup,
)
from toy_vllm.distributed.parallel_linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from toy_vllm.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from toy_vllm.models.parallel_qwen2 import (
    ParallelQwen2DecoderLayer,
    ParallelQwen2ForCausalLM,
)


def test_column_row_parallel_math():
    """Verify ColumnParallel + RowParallel is mathematically identical to single Linear layer."""
    torch.manual_seed(42)
    in_dim = 128
    hidden_dim = 256
    out_dim = 128
    tp_size = 2

    w_col = torch.randn(hidden_dim, in_dim)
    b_col = torch.randn(hidden_dim)
    w_row = torch.randn(out_dim, hidden_dim)
    b_row = torch.randn(out_dim)

    x = torch.randn(2, 4, in_dim)

    h_ref = torch.nn.functional.linear(x, w_col, b_col)
    y_ref = torch.nn.functional.linear(h_ref, w_row, b_row)

    set_tensor_model_parallel_world_size(tp_size)

    with torch.no_grad():
        set_tensor_model_parallel_rank(0)
        col0 = ColumnParallelLinear(in_dim, hidden_dim, bias=True)
        row0 = RowParallelLinear(hidden_dim, out_dim, bias=True)
        col0.load_master_weight(w_col, rank=0)
        col0.load_master_bias(b_col, rank=0)
        row0.load_master_weight(w_row, rank=0)
        row0.load_master_bias(b_row)

        set_tensor_model_parallel_rank(1)
        col1 = ColumnParallelLinear(in_dim, hidden_dim, bias=True)
        row1 = RowParallelLinear(hidden_dim, out_dim, bias=True)
        col1.load_master_weight(w_col, rank=1)
        col1.load_master_bias(b_col, rank=1)
        row1.load_master_weight(w_row, rank=1)
        row1.load_master_bias(b_row)

        h0 = col0(x)
        h1 = col1(x)

        out_row0 = torch.nn.functional.linear(h0, row0.weight)
        out_row1 = torch.nn.functional.linear(h1, row1.weight)
        y_parallel = (out_row0 + out_row1) + b_row

    diff = (y_parallel - y_ref).abs().max().item()
    print(f"Max absolute diff in Parallel Linear: {diff:.6f}")
    assert torch.allclose(
        y_parallel, y_ref, atol=1e-3, rtol=1e-3
    ), f"Parallel Linear output mismatch! Max diff: {diff}"
    destroy_model_parallel()
    print(
        "[PASS] ColumnParallel + RowParallel mathematical equivalence test passed!"
    )


def test_tensor_parallel_transformer_layer():
    """Verify single Transformer Decoder Layer with TP=2 matches single-card output with <1e-5 error."""
    torch.manual_seed(42)
    config = ModelConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )

    ref_layer = Qwen2DecoderLayer(config, 0)
    ref_layer.eval()

    set_tensor_model_parallel_world_size(2)

    with torch.no_grad():
        set_tensor_model_parallel_rank(0)
        tp_layer_0 = ParallelQwen2DecoderLayer(config, 0)
        tp_layer_0.input_layernorm.weight.copy_(ref_layer.input_layernorm.weight)
        tp_layer_0.post_attention_layernorm.weight.copy_(
            ref_layer.post_attention_layernorm.weight
        )
        tp_layer_0.self_attn.q_proj.load_master_weight(
            ref_layer.self_attn.q_proj.weight, rank=0
        )
        tp_layer_0.self_attn.q_proj.load_master_bias(
            ref_layer.self_attn.q_proj.bias, rank=0
        )
        tp_layer_0.self_attn.k_proj.load_master_weight(
            ref_layer.self_attn.k_proj.weight, rank=0
        )
        tp_layer_0.self_attn.k_proj.load_master_bias(
            ref_layer.self_attn.k_proj.bias, rank=0
        )
        tp_layer_0.self_attn.v_proj.load_master_weight(
            ref_layer.self_attn.v_proj.weight, rank=0
        )
        tp_layer_0.self_attn.v_proj.load_master_bias(
            ref_layer.self_attn.v_proj.bias, rank=0
        )
        tp_layer_0.self_attn.o_proj.load_master_weight(
            ref_layer.self_attn.o_proj.weight, rank=0
        )
        tp_layer_0.mlp.gate_proj.load_master_weight(
            ref_layer.mlp.gate_proj.weight, rank=0
        )
        tp_layer_0.mlp.up_proj.load_master_weight(
            ref_layer.mlp.up_proj.weight, rank=0
        )
        tp_layer_0.mlp.down_proj.load_master_weight(
            ref_layer.mlp.down_proj.weight, rank=0
        )

        set_tensor_model_parallel_rank(1)
        tp_layer_1 = ParallelQwen2DecoderLayer(config, 0)
        tp_layer_1.input_layernorm.weight.copy_(ref_layer.input_layernorm.weight)
        tp_layer_1.post_attention_layernorm.weight.copy_(
            ref_layer.post_attention_layernorm.weight
        )
        tp_layer_1.self_attn.q_proj.load_master_weight(
            ref_layer.self_attn.q_proj.weight, rank=1
        )
        tp_layer_1.self_attn.q_proj.load_master_bias(
            ref_layer.self_attn.q_proj.bias, rank=1
        )
        tp_layer_1.self_attn.k_proj.load_master_weight(
            ref_layer.self_attn.k_proj.weight, rank=1
        )
        tp_layer_1.self_attn.k_proj.load_master_bias(
            ref_layer.self_attn.k_proj.bias, rank=1
        )
        tp_layer_1.self_attn.v_proj.load_master_weight(
            ref_layer.self_attn.v_proj.weight, rank=1
        )
        tp_layer_1.self_attn.v_proj.load_master_bias(
            ref_layer.self_attn.v_proj.bias, rank=1
        )
        tp_layer_1.self_attn.o_proj.load_master_weight(
            ref_layer.self_attn.o_proj.weight, rank=1
        )
        tp_layer_1.mlp.gate_proj.load_master_weight(
            ref_layer.mlp.gate_proj.weight, rank=1
        )
        tp_layer_1.mlp.up_proj.load_master_weight(
            ref_layer.mlp.up_proj.weight, rank=1
        )
        tp_layer_1.mlp.down_proj.load_master_weight(
            ref_layer.mlp.down_proj.weight, rank=1
        )

        x = torch.randn(1, 4, 256)
        pos = torch.arange(4).unsqueeze(0)

        ref_out = ref_layer(x, pos)

        # Parallel forward with intermediate All-Reduce
        norm_x = tp_layer_0.input_layernorm(x)
        attn0 = tp_layer_0.self_attn(norm_x, pos)
        attn1 = tp_layer_1.self_attn(norm_x, pos)
        attn_allreduce = attn0 + attn1
        h = x + attn_allreduce

        norm_h = tp_layer_0.post_attention_layernorm(h)
        mlp0 = tp_layer_0.mlp(norm_h)
        mlp1 = tp_layer_1.mlp(norm_h)
        mlp_allreduce = mlp0 + mlp1
        tp_out = h + mlp_allreduce

    max_diff = (tp_out - ref_out).abs().max().item()
    print(f"Max absolute diff in Transformer Layer TP=2: {max_diff:.8f}")
    assert (
        max_diff < 1e-5
    ), f"Transformer Layer TP output mismatch! Max diff: {max_diff}"

    destroy_model_parallel()
    print(
        "[PASS] Tensor Parallel (TP=2) Transformer Layer numerical alignment passed!"
    )


def test_concurrent_multithread_full_model_tp():
    """Verify full Qwen2 model with multithreaded lockstep workers matches single-card."""
    torch.manual_seed(42)
    config = ModelConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )

    single_model = Qwen2ForCausalLM(config)
    single_model.eval()

    tp_size = 2
    comm_group = MockAllReduceGroup(world_size=tp_size)
    set_mock_all_reduce_hook(comm_group.all_reduce)

    set_tensor_model_parallel_world_size(tp_size)

    # Instantiate two models for the two worker ranks
    set_tensor_model_parallel_rank(0)
    tp_model_0 = ParallelQwen2ForCausalLM(config)
    tp_model_0.load_partition_weights(single_model.state_dict(), rank=0)
    tp_model_0.eval()

    set_tensor_model_parallel_rank(1)
    tp_model_1 = ParallelQwen2ForCausalLM(config)
    tp_model_1.load_partition_weights(single_model.state_dict(), rank=1)
    tp_model_1.eval()

    input_ids = torch.tensor([[10, 45, 99, 120]])

    results = {}

    def worker_run(rank, model):
        set_tensor_model_parallel_rank(rank)
        with torch.no_grad():
            out = model(input_ids)
            results[rank] = out

    t0 = threading.Thread(target=worker_run, args=(0, tp_model_0))
    t1 = threading.Thread(target=worker_run, args=(1, tp_model_1))

    t0.start()
    t1.start()
    t0.join()
    t1.join()

    with torch.no_grad():
        ref_out = single_model(input_ids)

    diff = (results[0] - ref_out).abs().max().item()
    print(f"Max absolute diff in Multi-threaded Full Model TP=2: {diff:.8f}")
    assert (
        diff < 1e-4
    ), f"Full Model TP output mismatch! Max diff: {diff}"

    destroy_model_parallel()
    set_mock_all_reduce_hook(None)
    print(
        "[PASS] Multi-threaded Full Model Tensor Parallel (TP=2) lockstep test passed!"
    )


if __name__ == "__main__":
    test_column_row_parallel_math()
    test_tensor_parallel_transformer_layer()
    test_concurrent_multithread_full_model_tp()
    print("\n>>> All V4 Tensor Parallelism tests passed successfully! <<<")
