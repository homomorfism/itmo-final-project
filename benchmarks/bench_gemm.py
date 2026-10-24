"""Benchmark X16 @ W4^T vs X16 @ W16^T on Llama-3.2-1B-Instruct linear-layer shapes.

Usage: uv run python -m benchmarks.bench_gemm [--out results/gemm.json]
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from int4_triton import quantize_int4, w4a16_matmul, w16a16_matmul

# Llama-3.2-1B: hidden 2048, intermediate 8192, 32 q heads / 8 kv heads, head_dim 64 -> W is [out, in]
LLAMA_1B_LINEARS = {
    "q_proj": (2048, 2048),
    "k_proj": (512, 2048),
    "v_proj": (512, 2048),
    "o_proj": (2048, 2048),
    "gate_proj": (8192, 2048),
    "up_proj": (8192, 2048),
    "down_proj": (2048, 8192),
}
TOKENS = [128, 512, 2048]
DECODE_TOKENS = [1, 16]


def bench(fn):
    """GPU time via CUDA graph replay (excludes Python/launch overhead, which is large on a loaded host)."""
    fn()  # autotune + compile outside of the graph
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        return triton.testing.do_bench_cudagraph(fn, rep=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gemm.json")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--tokens", type=int, nargs="+", default=TOKENS + DECODE_TOKENS)
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}  triton {triton.__version__}")
    rows = []
    for M in args.tokens:
        for name, (N, K) in LLAMA_1B_LINEARS.items():
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
            qw, s, z = quantize_int4(w, args.group_size)
            x16, w16 = x.half(), w.half()
            r = {
                "M": M, "layer": name, "N": N, "K": K,
                "torch_bf16_ms": bench(lambda: F.linear(x, w)),
                "torch_fp16_ms": bench(lambda: F.linear(x16, w16)),
                "triton_w16_bf16_ms": bench(lambda: w16a16_matmul(x, w)),
                "triton_w4_bf16_ms": bench(lambda: w4a16_matmul(x, qw, s, z)),
                "weight_bytes_bf16": w.numel() * w.element_size(),
                "weight_bytes_int4": qw.numel() * qw.element_size(),
                "weight_bytes_int4_with_scales": qw.numel() * 4 + s.numel() * 2 + z.numel() * 2,
            }
            rows.append(r)
            print(
                f"M={M:5d} {name:10s} [{N:5d}x{K:5d}]  torch bf16 {r['torch_bf16_ms']:7.3f} ms | "
                f"torch fp16 {r['torch_fp16_ms']:7.3f} | triton W16 {r['triton_w16_bf16_ms']:7.3f} | "
                f"triton W4 {r['triton_w4_bf16_ms']:7.3f}"
            )

    # per-decoder-layer totals (sum of all 7 linears)
    print("\nPer decoder layer (sum of 7 linears), ms:")
    print("| M | torch bf16 | torch fp16 | Triton W16 (bf16) | Triton W4 (bf16) | W4 speedup vs torch bf16 | vs torch fp16 | vs Triton W16 |")
    print("|---|---|---|---|---|---|---|---|")
    totals = []
    for M in args.tokens:
        sel = [r for r in rows if r["M"] == M]
        t = {k: sum(r[k] for r in sel) for k in ("torch_bf16_ms", "torch_fp16_ms", "triton_w16_bf16_ms", "triton_w4_bf16_ms")}
        t["M"] = M
        totals.append(t)
        w4 = t["triton_w4_bf16_ms"]
        print(
            f"| {M} | {t['torch_bf16_ms']:.3f} | {t['torch_fp16_ms']:.3f} | {t['triton_w16_bf16_ms']:.3f} | {w4:.3f} | "
            f"{t['torch_bf16_ms'] / w4:.2f}x | {t['torch_fp16_ms'] / w4:.2f}x | {t['triton_w16_bf16_ms'] / w4:.2f}x |"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "device": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "group_size": args.group_size, "timing": "CUDA graph replay, rep=200 ms",
            "rows": rows, "per_layer_totals": totals,
        }, f, indent=2)


if __name__ == "__main__":
    main()
