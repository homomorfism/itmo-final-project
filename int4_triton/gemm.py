"""Y = X @ W^T with X in bf16/fp16 and W either int4-packed (W4A16) or bf16/fp16 (W16A16 baseline).

The int4 weights are unpacked and dequantized on the fly inside the K-loop, so the
full-precision weight matrix is never materialized in global memory.

Three launch paths:
  * large M (prefill): classic tiled GEMM, result written directly in the output dtype;
  * small M: split-K with a deterministic reduction, without atomics or zero-fill;
  * M == 1 on Ampere and newer: a single reduction-based GEMV kernel.
"""
import torch
import triton
import triton.language as tl

from .utils import bf16_bits_to_fp16, fp32_to_bf16_bits

SMALL_M = 64


def _cfg(bm, bn, bk, warps, split_k=1, stages=2):
    return triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": split_k, "GROUP_M": 8},
        num_warps=warps,
        num_stages=stages,
    )


LARGE_M_CONFIGS = [
    _cfg(bm, bn, bk, w, stages=stages)
    for stages in (2, 3, 4)
    for bm, bn, bk, w in [
        (32, 64, 128, 4),
        (32, 128, 64, 4),
        (64, 64, 64, 4),
        (64, 128, 32, 4),
        (64, 128, 64, 4),
        (64, 64, 128, 4),
        (64, 128, 128, 4),
        (64, 256, 32, 8),
        (128, 64, 64, 4),
        (128, 128, 32, 4),
        (128, 128, 32, 8),
        (128, 128, 64, 8),
        (128, 128, 128, 8),
        (128, 256, 64, 8),
        (128, 256, 32, 8),
        (256, 128, 32, 8),
    ]
]

SMALL_M_CONFIGS = [
    _cfg(bm, bn, bk, w, sk)
    for bm in (16, 32)
    for bn, bk, w in [(32, 128, 4), (64, 32, 4), (64, 64, 4), (64, 128, 4), (128, 64, 4)]
    for sk in (1, 2, 4, 8, 16)
]


def _prune(configs, named_args, **kwargs):
    K = named_args["K"]
    G = kwargs.get("GROUP_SIZE", named_args.get("GROUP_SIZE"))
    w_int4 = kwargs.get("W_INT4", named_args.get("W_INT4"))
    legacy_bf16 = kwargs.get("LEGACY_BF16", named_args.get("LEGACY_BF16"))
    out = [
        c
        for c in configs
        if K % (c.kwargs["BLOCK_K"] * c.kwargs["SPLIT_K"]) == 0
        and (not w_int4 or G % c.kwargs["BLOCK_K"] == 0)
        and (not legacy_bf16 or c.num_stages == 2)
    ]
    return out or configs


@triton.jit
def _matmul_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_sg, stride_sn,
    stride_ym, stride_yn,
    GROUP_SIZE: tl.constexpr,
    W_INT4: tl.constexpr,
    IO_BF16: tl.constexpr,
    LEGACY_BF16: tl.constexpr,
    PARTIAL_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # grouped launch order for better L2 reuse
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_len = K // SPLIT_K
    k_start = pid_k * k_len

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < M
    mask_n = rn < N

    x_ptrs = x_ptr + rm[:, None] * stride_xm + (k_start + rk)[None, :] * stride_xk
    if W_INT4:
        # W^T tile [BLOCK_K, BLOCK_N]: row k lives in packed row k // 8, nibble k % 8.
        # Load each packed int32 once ([BLOCK_K // 8, BLOCK_N]) and expand to 8 rows in registers.
        rk8 = k_start // 8 + tl.arange(0, BLOCK_K // 8)
        w_ptrs = w_ptr + rk8[:, None] * stride_wk + rn[None, :] * stride_wn
        shifts = (tl.arange(0, 8) * 4)[None, :, None]
    else:
        w_ptrs = w_ptr + (k_start + rk)[:, None] * stride_wk + rn[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_start + k_len, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0)
        if LEGACY_BF16:
            x = bf16_bits_to_fp16(x)

        if W_INT4:
            q = tl.load(w_ptrs, mask=mask_n[None, :], other=0)
            q = (q[:, None, :] >> shifts) & 0xF  # [BLOCK_K // 8, 8, BLOCK_N]
            q = tl.reshape(q, (BLOCK_K, BLOCK_N))
            g = k // GROUP_SIZE
            s = tl.load(s_ptr + g * stride_sg + rn * stride_sn, mask=mask_n, other=0.0)
            z = tl.load(z_ptr + g * stride_sg + rn * stride_sn, mask=mask_n, other=0.0)
            # Dequantize in fp32, then round once to the tensor-core operand dtype.
            # An fp16 intermediate adds a conversion and double-rounds native bf16.
            w = (q - z[None, :].to(tl.int32)).to(tl.float32) * s[None, :].to(tl.float32)
            w_ptrs += (BLOCK_K // 8) * stride_wk
        else:
            w = tl.load(w_ptrs, mask=mask_n[None, :], other=0)
            if LEGACY_BF16:
                w = bf16_bits_to_fp16(w)
            w_ptrs += BLOCK_K * stride_wk

        acc += tl.dot(x, w.to(x.dtype))
        x_ptrs += BLOCK_K * stride_xk

    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    mask = mask_m[:, None] & mask_n[None, :]
    if PARTIAL_OUT:
        tl.store(y_ptrs + pid_k * M * N, acc, mask=mask)
    elif LEGACY_BF16:
        tl.store(y_ptrs, fp32_to_bf16_bits(acc), mask=mask)
    else:
        tl.store(y_ptrs, acc, mask=mask)


_KEY = ["M", "N", "K", "W_INT4", "IO_BF16", "LEGACY_BF16", "GROUP_SIZE"]
_matmul_large_m = triton.autotune(
    configs=LARGE_M_CONFIGS, key=_KEY, prune_configs_by={"early_config_prune": _prune},
    warmup=5, rep=10, use_cuda_graph=True,
)(_matmul_kernel)
_matmul_small_m = triton.autotune(
    configs=SMALL_M_CONFIGS, key=_KEY, prune_configs_by={"early_config_prune": _prune},
    warmup=5, rep=10, use_cuda_graph=True,
)(_matmul_kernel)


@triton.jit
def _reduce_split_k(partials, output, SIZE: tl.constexpr, SPLITS: tl.constexpr,
                    LEGACY_BF16: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, triton.next_power_of_2(SPLITS))
    values = tl.load(partials + splits[:, None] * SIZE + i[None, :],
                     mask=(splits[:, None] < SPLITS) & (i[None, :] < SIZE), other=0)
    value = tl.sum(values, axis=0)
    if LEGACY_BF16:
        tl.store(output + i, fp32_to_bf16_bits(value), mask=i < SIZE)
    else:
        tl.store(output + i, value, mask=i < SIZE)


@triton.autotune(
    configs=[triton.Config({"BN": n}, num_warps=w) for n in (4, 8, 16, 32) for w in (4, 8)],
    key=["N", "K", "GROUP_SIZE", "W_INT4"],
    warmup=5, rep=10, use_cuda_graph=True,
)
@triton.jit
def _gemv_kernel(x, weights, scales, zeros, output,
                 N: tl.constexpr, K: tl.constexpr,
                 SX: tl.constexpr, SWK: tl.constexpr, SWN: tl.constexpr,
                 SSG: tl.constexpr, SSN: tl.constexpr,
                 GROUP_SIZE: tl.constexpr, W_INT4: tl.constexpr, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    if W_INT4:
        k8 = tl.arange(0, triton.next_power_of_2(K) // 8)
        packed = tl.load(weights + k8[:, None] * SWK + n[None, :] * SWN,
                         (k8[:, None] < K // 8) & (n[None, :] < N), other=0)
        s = tl.load(scales + (k8[:, None] * 8 // GROUP_SIZE) * SSG + n[None, :] * SSN,
                    (k8[:, None] < K // 8) & (n[None, :] < N), other=0)
        z = tl.load(zeros + (k8[:, None] * 8 // GROUP_SIZE) * SSG + n[None, :] * SSN,
                    (k8[:, None] < K // 8) & (n[None, :] < N), other=0)
        # Keep one accumulator per packed word. Expanding all eight nibbles at once
        # multiplies register pressure and forces narrow, poorly coalesced N tiles.
        acc = tl.zeros((triton.next_power_of_2(K) // 8, BN), tl.float32)
        for nibble in tl.static_range(8):
            a = tl.load(x + (k8 * 8 + nibble) * SX, k8 < K // 8, other=0)
            q = (packed >> (nibble * 4)) & 15
            w = ((q - z.to(tl.int32)).to(tl.float32) * s.to(tl.float32)).to(a.dtype)
            acc += a.to(tl.float32)[:, None] * w.to(tl.float32)
        value = tl.sum(acc, axis=0)
    else:
        k = tl.arange(0, triton.next_power_of_2(K))
        a = tl.load(x + k * SX, k < K, other=0)
        w = tl.load(weights + k[:, None] * SWK + n[None, :] * SWN,
                    (k[:, None] < K) & (n[None, :] < N), other=0)
        value = tl.sum(a.to(tl.float32)[:, None] * w.to(tl.float32), axis=0)
    tl.store(output + n, value, mask=n < N)


# The autotuner + JIT dispatch costs ~250us of CPU time per call in Triton 3.1, which dominates
# decode. After the first (tuned) launch for a given specialization we keep the compiled kernel
# and launch it directly (~20us).
_LAUNCH_CACHE = {}
_DEVICE_CAPABILITY = {}


def _as_kernel_arg(t: torch.Tensor, legacy_bf16=False) -> torch.Tensor:
    return t.view(torch.int16) if legacy_bf16 and t.dtype == torch.bfloat16 else t


def _matmul(x: torch.Tensor, w_arg, s, z, N, stride_wk, stride_wn, group_size, w_int4) -> torch.Tensor:
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    io_bf16 = x.dtype == torch.bfloat16
    device_index = x.device.index
    if device_index not in _DEVICE_CAPABILITY:
        _DEVICE_CAPABILITY[device_index] = torch.cuda.get_device_capability(x.device)
    modern = _DEVICE_CAPABILITY[device_index][0] >= 8
    legacy_bf16 = io_bf16 and not modern
    w_arg = _as_kernel_arg(w_arg, legacy_bf16)
    if M == 1 and modern:
        y = torch.empty((M, N), dtype=x.dtype, device=x.device)
        args = (x2, w_arg, s, z, y, N, K, x2.stride(1), stride_wk, stride_wn, s.stride(0), s.stride(1))
        key = ("gemv", w_int4, group_size, x.dtype, w_arg.dtype, device_index) + tuple(
            a.data_ptr() % 16 if isinstance(a, torch.Tensor) else a for a in args
        )
        hit = _LAUNCH_CACHE.get(key)
        if hit is None:
            # Triton 3.1 benchmarks on an independent stream without a dependency
            # on the caller. Finish tensor initialization and prior allocator uses
            # before tuning; cached launches stay on the caller's current stream.
            torch.cuda.synchronize(x.device)
            compiled = _gemv_kernel[lambda meta: (triton.cdiv(N, meta["BN"]),)](
                *args, GROUP_SIZE=group_size, W_INT4=w_int4)
            grid = (triton.cdiv(N, _gemv_kernel.best_config.kwargs["BN"]), 1, 1)
            _LAUNCH_CACHE[key] = (compiled, grid)
        else:
            compiled, grid = hit
            compiled[grid](*args[:5])
        return y.reshape(*x.shape[:-1], N)
    small = M <= SMALL_M
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if small:
        partials = torch.empty((16, M, N), dtype=torch.float32, device=x.device)
        y_arg, kernel = partials, _matmul_small_m
    else:
        y_arg, kernel = _as_kernel_arg(y, legacy_bf16), _matmul_large_m
    x_arg = _as_kernel_arg(x2, legacy_bf16)
    args = (
        x_arg, w_arg, s, z, y_arg,
        M, N, K,
        x2.stride(0), x2.stride(1),
        stride_wk, stride_wn,
        s.stride(0), s.stride(1),
        N, 1,
    )
    # Triton specializes on int values (==1, %16) and pointer alignment, so they are part of the key
    key = (small, w_int4, io_bf16, legacy_bf16, group_size, x_arg.dtype, w_arg.dtype, device_index) + tuple(
        a.data_ptr() % 16 if isinstance(a, torch.Tensor) else a for a in args
    )
    hit = _LAUNCH_CACHE.get(key)
    if hit is None:
        torch.cuda.synchronize(x.device)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), meta["SPLIT_K"])
        compiled = kernel[grid](*args, GROUP_SIZE=group_size, W_INT4=w_int4, IO_BF16=io_bf16,
                                LEGACY_BF16=legacy_bf16, PARTIAL_OUT=small)
        cfg = kernel.best_config.kwargs
        grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]), cfg["SPLIT_K"], 1)
        _LAUNCH_CACHE[key] = (compiled, grid)
    else:
        compiled, grid = hit
        compiled[grid](*args)
    if small:
        _reduce_split_k[(triton.cdiv(M * N, 256),)](
            partials, _as_kernel_arg(y, legacy_bf16), M * N, grid[1], legacy_bf16, 256)
    return y.reshape(*x.shape[:-1], N)


def w4a16_matmul(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """x[..., K] (bf16/fp16) @ W^T, W given as packed int4 (see quantize.py). Returns x.dtype [..., N]."""
    assert x.dtype in (torch.bfloat16, torch.float16)
    K = x.shape[-1]
    assert qweight.shape[0] * 8 == K, (qweight.shape, K)
    group_size = K // scales.shape[0]
    return _matmul(x, qweight, scales, zeros, qweight.shape[1], qweight.stride(0), qweight.stride(1), group_size, True)


def w16a16_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Baseline Triton kernel: x[..., K] @ w[N, K]^T, same structure as the W4 kernel but full-precision weights."""
    assert x.dtype == w.dtype and x.dtype in (torch.bfloat16, torch.float16)
    w_arg = w
    dummy = torch.empty((1, 1), dtype=torch.float16, device=x.device)
    return _matmul(x, w_arg, dummy, dummy, w.shape[0], w_arg.stride(1), w_arg.stride(0), 128, False)
