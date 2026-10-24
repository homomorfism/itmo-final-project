"""Group-wise asymmetric int4 quantization of a 2D weight matrix W[N, K].

For every group of `group_size` consecutive elements along K (per output row n):
    scale = (max - min) / 15,  zero = round(-min / scale)        (min <= 0 <= max)
    q     = clamp(round(w / scale) + zero, 0, 15)
    w_hat = (q - zero) * scale

Storage layout (GPTQ-style, convenient for X @ W^T where W^T is [K, N]):
    qweight : int32 [K // 8, N]    -- 8 int4 values packed along K, nibble i = row 8*k8 + i
    scales  : fp16  [K // G, N]
    zeros   : fp16  [K // G, N]    -- integer-valued zero points
qweight takes N*K/2 bytes vs N*K*2 bytes for fp16: exactly 4x smaller.
"""
import torch
import triton
import triton.language as tl

from .utils import bf16_bits_to_fp32


@triton.jit
def _quantize_int4_kernel(
    w_ptr, qw_ptr, s_ptr, z_ptr,
    N,
    stride_wn, stride_wk,
    stride_qk, stride_qn,
    stride_sg, stride_sn,
    W_BF16: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    g = tl.program_id(1)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask_n = rn < N

    w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk, mask=mask_n[:, None], other=0)
    if W_BF16:
        w = bf16_bits_to_fp32(w)
    else:
        w = w.to(tl.float32)

    wmax = tl.maximum(tl.max(w, axis=1), 0.0)
    wmin = tl.minimum(tl.min(w, axis=1), 0.0)
    scale = (wmax - wmin) / 15.0
    scale = tl.where(scale == 0.0, 1.0, scale)
    # quantize against the fp16-rounded scale so that dequantization is consistent
    scale = scale.to(tl.float16).to(tl.float32)
    zero = tl.minimum(tl.maximum(tl.floor(-wmin / scale + 0.5), 0.0), 15.0)

    q = tl.floor(w / scale[:, None] + 0.5) + zero[:, None]
    q = tl.minimum(tl.maximum(q, 0.0), 15.0).to(tl.int32)

    # pack 8 nibbles along K into one int32 (bits are disjoint, so sum == bitwise or)
    q = tl.reshape(q, (BLOCK_N, GROUP_SIZE // 8, 8))
    shifts = tl.arange(0, 8) * 4
    packed = tl.sum(q << shifts[None, None, :], axis=2)  # [BLOCK_N, GROUP_SIZE // 8]

    rk8 = g * (GROUP_SIZE // 8) + tl.arange(0, GROUP_SIZE // 8)
    tl.store(qw_ptr + rk8[None, :] * stride_qk + rn[:, None] * stride_qn, packed, mask=mask_n[:, None])
    tl.store(s_ptr + g * stride_sg + rn * stride_sn, scale.to(tl.float16), mask=mask_n)
    tl.store(z_ptr + g * stride_sg + rn * stride_sn, zero.to(tl.float16), mask=mask_n)


def quantize_int4(w: torch.Tensor, group_size: int = 128):
    """Quantize W[N, K] (fp16 or bf16, CUDA) -> (qweight int32 [K/8, N], scales fp16 [K/G, N], zeros fp16 [K/G, N])."""
    assert w.dim() == 2 and w.is_cuda
    assert w.dtype in (torch.float16, torch.bfloat16)
    N, K = w.shape
    if group_size == -1:
        group_size = K
    assert K % group_size == 0 and group_size % 8 == 0
    assert triton.next_power_of_2(group_size) == group_size, "group_size must be a power of 2"

    qweight = torch.empty((K // 8, N), dtype=torch.int32, device=w.device)
    scales = torch.empty((K // group_size, N), dtype=torch.float16, device=w.device)
    zeros = torch.empty_like(scales)

    w_bf16 = w.dtype == torch.bfloat16
    w_arg = w.view(torch.int16) if w_bf16 else w
    block_n = max(1, min(64, 8192 // group_size))
    grid = (triton.cdiv(N, block_n), K // group_size)
    _quantize_int4_kernel[grid](
        w_arg, qweight, scales, zeros,
        N,
        w_arg.stride(0), w_arg.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        W_BF16=w_bf16,
        GROUP_SIZE=group_size,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return qweight, scales, zeros


# ----------------------------------------------------------------------------- torch references


def quantize_int4_ref(w: torch.Tensor, group_size: int = 128):
    """Pure-torch reference of `quantize_int4` (same layout, same rounding)."""
    N, K = w.shape
    if group_size == -1:
        group_size = K
    wg = w.float().reshape(N, K // group_size, group_size)
    wmax = wg.amax(-1).clamp(min=0)
    wmin = wg.amin(-1).clamp(max=0)
    scale = (wmax - wmin) / 15
    scale = torch.where(scale == 0, torch.ones_like(scale), scale).half().float()
    zero = torch.floor(-wmin / scale + 0.5).clamp(0, 15)
    q = (torch.floor(wg / scale[..., None] + 0.5) + zero[..., None]).clamp(0, 15).to(torch.int32)
    q = q.reshape(N, K // 8, 8)
    shifts = torch.arange(0, 32, 4, device=w.device, dtype=torch.int32)
    packed = (q << shifts).sum(-1, dtype=torch.int32)  # wraps like int32 arithmetic
    return packed.t().contiguous(), scale.half().t().contiguous(), zero.half().t().contiguous()


def unpack_int4_ref(qweight: torch.Tensor) -> torch.Tensor:
    """int32 [K/8, N] -> int32 [N, K] with values in [0, 15]."""
    K8, N = qweight.shape
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    q = (qweight.t()[:, :, None] >> shifts) & 0xF  # [N, K/8, 8]
    return q.reshape(N, K8 * 8)


def dequantize_int4_ref(qweight, scales, zeros, dtype=torch.float16) -> torch.Tensor:
    """Reconstruct W_hat[N, K] in fp32, then round once to the requested dtype."""
    q = unpack_int4_ref(qweight).float()
    N, K = q.shape
    G = K // scales.shape[0]
    s = scales.t().repeat_interleave(G, dim=1).float()
    z = zeros.t().repeat_interleave(G, dim=1).float()
    return ((q - z) * s).to(dtype)
