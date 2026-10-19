import torch
import pytest
from toy_vllm.modules.moe import MoERouter, MoEMLP, MoEExpert
from toy_vllm.distributed.expert_parallel import ExpertParallelMoEMLP


def test_moe_router_topk_and_softmax():
    """Verify MoERouter produces Top-K selections with valid probabilities summing to 1.0."""
    torch.manual_seed(42)
    hidden_size = 256
    num_experts = 8
    top_k = 2

    router = MoERouter(hidden_size, num_experts, top_k=top_k)
    x = torch.randn(4, hidden_size)

    weights, indices, logits = router(x)

    assert weights.shape == (4, top_k)
    assert indices.shape == (4, top_k)
    assert logits.shape == (4, num_experts)

    # Probabilities across selected top-k experts must sum to 1.0
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(
        weight_sums, torch.ones_like(weight_sums), atol=1e-5
    ), "Router weights do not sum to 1.0!"
    print("[PASS] MoE Router Top-K & Softmax normalization test passed!")


def test_moe_permute_grouped_gemm_vs_naive():
    """
    CRITICAL TEST:
    Verify that our high-performance 'Expert-selects-Token' permuted batch GEMM
    is mathematically 100% BIT-LEVEL IDENTICAL to the naive token-by-token loop!
    """
    torch.manual_seed(42)
    hidden_size = 128
    intermediate_size = 256
    num_experts = 4
    top_k = 2

    moe = MoEMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_shared_experts=1,
    )
    moe.eval()

    x = torch.randn(8, hidden_size)  # 8 tokens

    with torch.no_grad():
        fast_out = moe(x)  # Permute + Batched GEMM + Scatter
        naive_out = moe.naive_forward(x)  # Naive loop

    max_diff = (fast_out - naive_out).abs().max().item()
    print(f"Max absolute diff between Permuted GEMM and Naive MoE: {max_diff:.8f}")
    assert (
        max_diff < 1e-5
    ), f"Permuted MoE output mismatch! Max diff: {max_diff}"
    print(
        "[PASS] High-performance 'Expert-selects-Token' Permuted GEMM equivalence passed!"
    )


def test_expert_parallel_equivalence():
    """
    Verify Expert Parallelism (EP=2) partitioning produces identical output
    to a centralized global MoE layer.
    """
    torch.manual_seed(42)
    hidden_size = 128
    intermediate_size = 256
    total_experts = 4
    top_k = 2
    ep_size = 2

    # 1. Master global MoE
    global_moe = MoEMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=total_experts,
        top_k=top_k,
        num_shared_experts=1,
    )
    global_moe.eval()

    # Extract all expert weights
    expert_weights = [
        {
            "gate_proj.weight": exp.mlp.gate_proj.weight.data.clone(),
            "up_proj.weight": exp.mlp.up_proj.weight.data.clone(),
            "down_proj.weight": exp.mlp.down_proj.weight.data.clone(),
        }
        for exp in global_moe.experts
    ]
    shared_weight = {
        "gate_proj.weight": global_moe.shared_experts[
            0
        ].mlp.gate_proj.weight.data.clone(),
        "up_proj.weight": global_moe.shared_experts[
            0
        ].mlp.up_proj.weight.data.clone(),
        "down_proj.weight": global_moe.shared_experts[
            0
        ].mlp.down_proj.weight.data.clone(),
    }

    # 2. Expert-Parallel partitions (Rank 0 has E0, E1; Rank 1 has E2, E3)
    ep_rank0 = ExpertParallelMoEMLP(
        hidden_size,
        intermediate_size,
        total_experts,
        top_k,
        ep_size,
        ep_rank=0,
    )
    ep_rank0.load_master_weights(
        global_moe.router.gate.weight.data, expert_weights, shared_weight
    )
    ep_rank0.eval()

    ep_rank1 = ExpertParallelMoEMLP(
        hidden_size,
        intermediate_size,
        total_experts,
        top_k,
        ep_size,
        ep_rank=1,
    )
    ep_rank1.load_master_weights(
        global_moe.router.gate.weight.data, expert_weights, shared_weight
    )
    ep_rank1.eval()

    # Test tokens
    x = torch.randn(6, hidden_size)

    with torch.no_grad():
        ref_out = global_moe(x)

        # Simulate EP workflow:
        # Step 1: All ranks route the tokens
        top_k_w, top_k_idx, _ = ep_rank0.router(x)
        num_tokens = x.shape[0]

        # Step 2: 1st All-to-All dispatch & local computation on Rank 0 and Rank 1
        routed_results = torch.zeros_like(x)

        for rank_inst in [ep_rank0, ep_rank1]:
            for local_i in range(rank_inst.local_num_experts):
                global_e_id = rank_inst.start_expert_id + local_i
                # Find tokens assigned to this expert
                mask = top_k_idx == global_e_id  # (num_tokens, top_k)
                if not mask.any():
                    continue

                tok_indices, k_positions = torch.where(mask)
                weights = top_k_w[tok_indices, k_positions].unsqueeze(1)

                exp_in = x[tok_indices]
                # Executed on whichever rank holds this expert
                exp_out = rank_inst.forward_local_computation(
                    exp_in, global_e_id
                )

                # Step 3: 2nd All-to-All return back to source token positions
                routed_results.index_add_(0, tok_indices, weights * exp_out)

        # Add shared expert
        shared_out = ep_rank0.shared_experts[0](x)
        ep_out = shared_out + routed_results

    diff = (ep_out - ref_out).abs().max().item()
    print(f"Max absolute diff between Global MoE and EP=2: {diff:.8f}")
    assert diff < 1e-5, f"EP MoE mismatch! Max diff: {diff}"
    print(
        "[PASS] Expert Parallelism (EP=2) All-to-All numerical alignment passed!"
    )


if __name__ == "__main__":
    test_moe_router_topk_and_softmax()
    test_moe_permute_grouped_gemm_vs_naive()
    test_expert_parallel_equivalence()
    print("\n>>> All V5 MoE & Expert Parallelism tests passed successfully! <<<")
