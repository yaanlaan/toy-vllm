import torch
import torch.nn as nn
from typing import Optional, List
from toy_vllm.config import SamplingParams


class Sampler(nn.Module):
    """
    Logits Sampler supporting Greedy, Temperature, Top-K, Top-P, and Repetition Penalty.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generated_tokens: Optional[List[int]] = None,
    ) -> int:
        """
        Samples a single token id from logits for a single sequence.
        logits: shape (vocab_size,) or (1, vocab_size)
        """
        logits = logits.squeeze()

        # 1. Repetition Penalty
        if (
            params.repetition_penalty != 1.0
            and generated_tokens
            and len(generated_tokens) > 0
        ):
            penalty = params.repetition_penalty
            unique_tokens = list(set(generated_tokens))
            penalized_logits = logits[unique_tokens]
            # If logit > 0: logit / penalty, else: logit * penalty
            penalized_logits = torch.where(
                penalized_logits > 0,
                penalized_logits / penalty,
                penalized_logits * penalty,
            )
            logits[unique_tokens] = penalized_logits

        # 2. Greedy Sampling
        if params.temperature <= 1e-5 or params.top_k == 1:
            return int(torch.argmax(logits, dim=-1).item())

        # 3. Temperature Scaling
        logits = logits / params.temperature

        # 4. Top-K Truncation
        if params.top_k > 0:
            top_k = min(params.top_k, logits.size(-1))
            val, _ = torch.topk(logits, top_k)
            kth_val = val[-1]
            logits = torch.where(logits < kth_val, float("-inf"), logits)

        # 5. Top-P (Nucleus) Truncation
        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > params.top_p
            # Shift the indices to the right to keep also the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        # 6. Softmax and Multinomial Sampling
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())
