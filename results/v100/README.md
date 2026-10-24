# INT4 weight quantization + W4A16 GEMM in Triton (Llama-3.2-1B-Instruct)

Hardware: **Tesla V100-PCIE-16GB (sm_70)**, torch 2.5.1+cu121, Triton 3.1.0, transformers 4.49.

```bash
uv sync
uv run pytest -q tests                                    # correctness (74 tests)
uv run python -m benchmarks.bench_gemm                    # step 3: X16@W4^T vs X16@W16^T
uv run python -m benchmarks.eval_llama --mode bf16        # step 5: baseline
uv run python -m benchmarks.eval_llama --mode w4          # step 5: int4 model
```

## Layout

| File | Content |
|---|---|
| `int4_triton/quantize.py` | Triton kernel: fp16/bf16 `W[N,K]` → int4, packed 8-per-`int32` (step 1) + torch reference |
| `int4_triton/gemm.py` | Triton kernel `Y = X @ W^T`, X bf16/fp16, W int4 (dequantized on the fly) or bf16 (step 2) |
| `int4_triton/qlinear.py` | `Int4Linear` + `quantize_model_linears(model)` (step 4) |
| `int4_triton/utils.py` | bit-level bf16↔fp16/fp32 conversions (see the V100 notes) |
| `benchmarks/bench_gemm.py` | GEMM benchmark on Llama-3.2-1B shapes (step 3) → `results/gemm.json` |
| `benchmarks/eval_llama.py` | wikitext-2 perplexity, prefill/decode speed, memory (step 5) → `results/llama_*.json` |

## 1. Quantization kernel

Asymmetric group-wise int4 quantization (round-to-nearest), group size 128 along K by default:
`scale = (max-min)/15`, `zero = round(-min/scale)`, `q = clamp(round(w/scale)+zero, 0, 15)`.

* `qweight`: `int32 [K/8, N]`, 8 nibbles packed along K (GPTQ-style layout, coalesced loads along N in the GEMM).
* `scales`, `zeros`: `fp16 [K/G, N]`.

Packed weight memory is **exactly 4× smaller** than fp16. This is checked in `test_memory_reduction_and_error`: `N*K*2` bytes become `N*K/2`. Counting scales and zeros (fp16, G=128), it is **3.76×**.
For all Llama decoder linears: 1856 MiB → 464 MiB packed, or 493 MiB with scales and zeros.

## 2. W4A16 GEMM kernel

The kernel is a tiled `tl.dot` GEMM with grouped launch order. Inside the K-loop it loads a `[BK/8, BN]` int32 tile once, unpacks it in registers to `[BK, BN]` nibbles, dequantizes `(q - z) * s` to fp16 and feeds it to the tensor cores. Accumulation is in fp32. The full-precision weight is never written to global memory. Two autotuned paths share the kernel body:
* **M > 64** (prefill): plain tiles, output stored directly in bf16.
* **M ≤ 64** (decode): **split-K**. Partial sums are `atomic_add`ed into an fp32 buffer, because there are too few output tiles to fill 80 SMs.

After the first autotuned call for a given shape, the compiled kernel is cached and launched directly. That bypasses roughly 250 µs per call of Triton 3.1 autotuner/JIT dispatch overhead.

### V100-specific notes (important)
* `tl.dot` on **bf16** operands compiles on sm_70 but **returns wrong results** (max abs error about 850 on a 4096³ test). Converting bf16 with `.to()` fails in ptxas (`cvt.bf16` needs sm_80).
  So bf16 tensors are passed as raw `int16` bits, converted with shifts and bitcasts, and the matmul runs on **fp16 tensor cores** with fp32 accumulation. The output is rounded back to bf16 with round-to-nearest-even.
  bf16→fp16 is exact for |x| < 65504; larger values saturate.
* V100 has **no bf16 tensor cores**, so cuBLAS bf16 (`torch bf16` below) is about 7× slower than cuBLAS fp16. Below I report both the literal baseline (torch bf16) and fairer ones: torch fp16 and Triton W16, which is the same kernel with bf16 weights.

## 3. GEMM benchmark (`results/gemm.json`)

The table shows GPU time in ms, the sum over the 7 linears of one Llama-3.2-1B decoder layer (q 2048×2048, k/v 512×2048, o 2048×2048, gate/up 8192×2048, down 2048×8192). Timings use CUDA-graph replay to exclude Python overhead; the host was heavily loaded.

| tokens M | torch bf16 (X16@W16ᵀ) | torch fp16 | Triton W16 (bf16) | **Triton W4 (bf16)** | W4 vs torch bf16 | W4 vs torch fp16 | W4 vs Triton W16 |
|---|---|---|---|---|---|---|---|
| 128  | 3.614 | 0.408 | 0.636 | **0.651** | 5.55× | 0.63× | 0.98× |
| 512  | 7.436 | 0.962 | 1.470 | **1.662** | 4.47× | 0.58× | 0.88× |
| 2048 | 25.331 | 3.306 | 4.797 | **5.517** | 4.59× | 0.60× | 0.87× |
| 1 (decode)  | 0.176 | 0.177 | 0.276 | 0.218 | 0.81× | 0.81× | 1.27× |
| 16 (decode) | 3.331 | 0.232 | 0.298 | 0.265 | 12.6× | 0.87× | 1.12× |

Per-shape numbers are in `results/logs/bench_gemm.log`.

Takeaways:
* For M ≥ 128 the GEMM is compute-bound, so int4 cannot speed it up. Dequantization costs 0–13% compared with the same kernel on bf16 weights.
* Compared with the literal baseline (`bf16 @ bf16` via cuBLAS), W4 is **4.5–5.5× faster**. That is mostly because the kernel uses fp16 tensor cores, which cuBLAS bf16 cannot on V100.
* Triton on Volta reaches about 60–70% of cuBLAS fp16. Triton 3.x has no `cp.async` pipelining for sm_70.
* For decode-sized M (memory-bound), W4 beats Triton W16 by 1.1–1.3× and beats cuBLAS on the K=8192 shape (M=16: 0.059 ms vs 0.056/0.079 ms). It is still slower than cuBLAS for small matrices, where the split-K zero-init and cast kernels dominate.

## 4–5. Quantized Llama-3.2-1B-Instruct (`results/llama_*.json`)

wikitext-2-raw-v1 test split, contexts of 2048 tokens (141 windows, non-overlapping). The prefill time is a forward pass over one 2048-token window. Decode is single-token steps with a static KV cache and **CUDA graph** (prompt 128, 128 new tokens). The "eager" column is CPU-bound on this shared host (load average around 90), so it is not representative.
By default `lm_head` (tied with embeddings) stays bf16. The `+lm_head` row quantizes it too.

| model | linears | wikitext-2 PPL | prefill ms / 2048 tok | prefill tok/s | decode tok/s (graph) | linear weights | whole model | peak mem |
|---|---|---|---|---|---|---|---|---|
| bf16 (torch, cuBLAS) | bf16 | **13.163** | 700 | 2 924 | 139.6 | 1856 MiB | 2357 MiB | 4.76 GB |
| fp16 (torch, cuBLAS) | fp16 | 13.160 | 94 | 21 801 | 155.1 | 1856 MiB | 2357 MiB | 4.76 GB |
| Triton W16 (bf16) | bf16 | 13.164 | 367 | 5 584 | 108.7 | 1856 MiB | 2357 MiB | 4.76 GB |
| **Triton W4 g128 (bf16 act.)** | int4 | **16.435** | **384** | **5 337** | 125.2 | **493 MiB** | **994 MiB** | 3.43 GB |
| Triton W4 g128 + lm_head | int4 | 16.816 | 297 | 6 895 | 127.9 | 626 MiB* | 1127 MiB | 3.56 GB |
| W4 g128 dequantized → torch bf16 (reference) | bf16 | 16.432 | 700 | 2 926 | 139.9 | – | – | – |
| Triton W4 g64 | int4 | 15.420 | 382 | 5 368 | – | 522 MiB | – | – |
| Triton W4 g32 | int4 | 14.808 | 382 | 5 362 | – | 580 MiB | – | – |

\* Includes the int4 lm_head. The bf16 embedding table is still kept for the input embeddings.

Conclusions:
* **The kernel is exact with respect to the quantization**. The Triton W4 model gives PPL 16.435, against 16.432 for the same int4 weights dequantized and run through torch. The remaining difference is fp16 versus bf16 rounding of activations.
* **PPL degradation (13.16 → 16.44) comes from plain round-to-nearest quantization**. Llama-3 models are known to be sensitive to it. Smaller groups help: g64 gives 15.42 and g32 gives 14.81, for slightly more memory. GPTQ or AWQ would close most of the remaining gap, and the kernel/storage format is compatible with them.
* **Memory**: decoder linear weights take 3.76× less space (1856 → 493 MiB), the whole model 2.4× less (2357 → 994 MiB), and peak memory during evaluation drops from 4.76 to 3.43 GB.
* **Prefill speed**: 1.8× faster than the bf16 baseline (2.3× with int4 lm_head), and on par with Triton W16. On V100 the honest upper bound is cuBLAS fp16. Running the whole model in fp16 is 4× faster still, because attention, lm_head and the other bf16 ops also miss the tensor cores.
* **Decode**: with CUDA graphs W4 is 10% slower than bf16. On a 1B model, decode on V100 is dominated by dozens of small non-GEMM kernels, and the M=1 W4 path launches 3 kernels per linear (zero-init, split-K GEMM, cast). A dedicated GEMV kernel without `tl.dot` or atomics would be the next step.
