"""Bit-level bf16 <-> fp16/fp32 helpers.

Triton 3.1 on Volta (sm_70) cannot emit `cvt` instructions involving bf16
(they require sm_80), and `tl.dot` on bf16 operands silently produces wrong
results there. So bf16 tensors are passed to kernels as raw int16 bits and
converted manually; the actual matmul runs on fp16 tensor cores.
"""
import triton
import triton.language as tl

FP16_MAX = 65504.0


@triton.jit
def bf16_bits_to_fp32(x):
    """int16 tensor holding bf16 bits -> fp32 (exact)."""
    return (x.to(tl.int32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def bf16_bits_to_fp16(x):
    """int16 tensor holding bf16 bits -> fp16 (exact inside fp16 range, saturating outside)."""
    f = bf16_bits_to_fp32(x)
    f = tl.minimum(tl.maximum(f, -65504.0), 65504.0)
    return f.to(tl.float16)


@triton.jit
def fp32_to_bf16_bits(x):
    """fp32 -> int16 bf16 bits with round-to-nearest-even."""
    u = x.to(tl.int32, bitcast=True)
    u = u + (((u >> 16) & 1) + 0x7FFF)
    return (u >> 16).to(tl.int16)
