import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from toy_vllm.kernels.fused_ops import fused_swiglu

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class SwiGLUMLP(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network.
    Formula:
        y = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    Uses Triton fused kernel if input is on CUDA.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        if HAS_TRITON and x.is_cuda:
            activated = fused_swiglu(gate, up)
        else:
            activated = F.silu(gate) * up

        return self.down_proj(activated)
