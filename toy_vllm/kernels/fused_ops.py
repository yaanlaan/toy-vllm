import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    out_res_ptr,
    hidden_size: int,
    eps: float,
    HAS_RES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton Kernel: Fused Residual Add + Root Mean Square Normalization.
    Computes:
        acc = x + residual (if HAS_RES else x)
        out = (acc * rsqrt(mean(acc^2) + eps)) * weight
        out_residual = acc
    """
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    row_offset = row_idx * hidden_size
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    if HAS_RES:
        res = tl.load(res_ptr + row_offset + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        acc = x + res
    else:
        acc = x

    # Store updated residual
    if out_res_ptr is not None:
        tl.store(out_res_ptr + row_offset + cols, acc, mask=mask)

    # Compute variance & RMS
    variance = tl.sum(acc * acc, axis=0) / hidden_size
    rsqrt = tl.rsqrt(variance + eps)

    # Normalize and scale
    normed = acc * rsqrt * weight
    tl.store(out_ptr + row_offset + cols, normed, mask=mask)


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Python wrapper for Triton Fused Add + RMSNorm.
    Returns:
        output: normed tensor with same shape and dtype as x
        new_residual: accumulated residual tensor
    """
    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    num_rows = x_2d.shape[0]

    has_res = residual is not None
    if has_res:
        res_2d = residual.reshape(-1, hidden_size)
    else:
        res_2d = None

    output = torch.empty_like(x_2d)
    new_residual = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    grid = (num_rows,)

    _fused_add_rmsnorm_kernel[grid](
        x_2d,
        res_2d,
        weight,
        output,
        new_residual,
        hidden_size=hidden_size,
        eps=eps,
        HAS_RES=has_res,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(orig_shape), new_residual.reshape(orig_shape)


@triton.jit
def _fused_swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton Kernel: Fused SwiGLU activation.
    Computes:
        out = (gate * sigmoid(gate)) * up
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    g = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU: g * sigmoid(g)
    silu_g = g / (1.0 + tl.exp(-g))
    out = silu_g * u

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for Triton Fused SwiGLU.
    """
    assert (
        gate.shape == up.shape
    ), f"Shape mismatch: {gate.shape} vs {up.shape}"
    output = torch.empty_like(gate)
    n_elements = gate.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _fused_swiglu_kernel[grid](
        gate,
        up,
        output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
