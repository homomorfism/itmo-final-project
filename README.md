# INT4 weight quantization + W4A16 GEMM in Triton (Llama-3.2-1B-Instruct)

Measured on **NVIDIA H200 (sm_90)** on 2026-10-24: Python 3.12.3, torch 2.5.1+cu121, Triton 3.1.0, transformers 4.49.0, NVIDIA driver 580.196.07. Native bf16 tensor cores are enabled. Dependencies are unchanged and installed with `uv sync --frozen`.

Kernel measurements and correctness tests use physical GPU 4; model evaluations use physical GPU 5. Both were idle before use; final reported timing runs sharing a GPU are sequential. GPU identities and software versions are recorded in [hardware.json](results/h200/hardware.json). No power limits or clocks were changed.

```bash
uv sync --frozen
CUDA_VISIBLE_DEVICES=4 uv run pytest -q tests
CUDA_VISIBLE_DEVICES=4 uv run python -m benchmarks.bench_gemm --out results/h200/gemm.json
CUDA_VISIBLE_DEVICES=5 uv run python -m benchmarks.eval_llama --mode bf16 --out-dir results/h200
CUDA_VISIBLE_DEVICES=5 uv run python -m benchmarks.eval_llama --mode w4 --out-dir results/h200
# Other comparisons: --mode fp16 / triton-w16 / w4-fake
# Quantize the output projection too: --mode w4 --quantize-lm-head
# Smaller groups: --mode w4 --group-size 64 --out-dir results/h200/groupsize_g64
#                --mode w4 --group-size 32 --out-dir results/h200/groupsize_g32
```

All new measurements are in [results/h200](results/h200). The previous V100 report is preserved in [results/v100/README.md](results/v100/README.md); the historical V100 JSON files and logs remain at their original paths under `results/`. V100 and H200 numbers should not be mixed: the former uses fp16 emulation for bf16 and a different decode timing implementation.

## Layout

| File | Content |
|---|---|
| `int4_triton/quantize.py` | Triton group-wise INT4 quantization, packed eight values per int32, and torch reference (step 1) |
| `int4_triton/gemm.py` | Fused unpack/dequantize + GEMM, split-K reduction, decode GEMV, and W16 baseline (step 2) |
| `int4_triton/qlinear.py` | `Int4Linear` and model conversion (step 4) |
| `int4_triton/utils.py` | bf16 bit conversion helpers retained for the Volta fallback |
| `benchmarks/bench_gemm.py` | Seven decoder-linear shapes, five token counts, CUDA-graph timing (step 3) |
| `benchmarks/eval_llama.py` | WikiText-2 perplexity, prefill/decode timing, memory and sample generation (step 5) |

## 1. Quantization and storage

Asymmetric round-to-nearest INT4 quantization uses groups of 128 weights along K by default. Each group's range includes zero:

`scale = (max - min) / 15`, `zero = floor(-min / scale + 0.5)`, `q = clamp(floor(w / scale + 0.5) + zero, 0, 15)`.

The scale is rounded to fp16 before quantization; an all-zero group uses scale 1. Storage for `W[N,K]`:

* `qweight`: `int32[K/8,N]`, with eight consecutive K values packed into each word;
* `scales`, `zeros`: `fp16[K/G,N]`.

Packed weights take **exactly 4× less memory** than fp16/bf16. Including scales and zero points at G=128 gives **3.76×** compression. Decoder linear weights occupy 1856 MiB before quantization and 493 MiB afterward. These counts exclude the tied embedding/output table unless `--quantize-lm-head` is passed.

## 2. H200 kernel changes

* **Native bf16:** Ampere and newer load bf16 operands directly and use bf16 tensor-core `tl.dot` with fp32 accumulation. W4 dequantization computes `(q - zero) * scale` in fp32 and rounds once to the activation dtype. This avoids an unnecessary fp16 intermediate and matches the updated torch reference. The Volta bit-conversion fallback remains available; it was not revalidated on a V100 in this run.
* **Prefill, M > 64:** grouped tiled GEMM with on-the-fly unpacking; the full dequantized weight matrix is never stored in global memory. Autotuning includes K tiles up to 128 and 2–4 pipeline stages.
* **Small batches, 2 ≤ M ≤ 64:** each split-K block writes a separate fp32 partial result, then one reduction kernel sums and casts it. This replaces zero-fill + atomic GEMM + cast with two kernels and a deterministic reduction.
* **One-token decode, M = 1:** a dedicated GEMV on Ampere and newer writes the output in one launch, without a scratch buffer or atomics. The W4 path loads packed words once and accumulates their eight nibbles in registers.
* **Dispatch and tuning:** compiled launches are cached by shape, strides, dtype, group size, alignment and device. CUDA-graph autotuning measures GPU execution. Triton 3.1's autotuner uses independent streams, so first-time tuning synchronizes pending tensor operations; cached calls do not synchronize.

## 3. GEMM measurements

Each row sums GPU time for the seven linears in one decoder layer: q/o 2048×2048, k/v 512×2048, gate/up 8192×2048, down 2048×8192. All activations in the W4/W16 columns are bf16. Each callable is warmed before CUDA-graph replay (`rep=200` ms, Triton tester's ten timing repetitions); compilation and Python dispatch are excluded.

This is a repeated-weight microbenchmark, with substantial L2 reuse on H200. Summing isolated GEMMs does not measure whole-model latency or guarantee the same bandwidth behavior as streaming all model weights.

| tokens M | torch bf16, ms | torch fp16, ms | Triton W16, ms | Triton W4, ms | W4 speedup vs torch bf16 | W4 speedup vs Triton W16 |
|---|---|---|---|---|---|---|
| 128 | 0.065 | 0.069 | 0.087 | **0.166** | 0.39× | 0.52× |
| 512 | 0.140 | 0.141 | 0.142 | **0.316** | 0.44× | 0.45× |
| 2048 | 0.410 | 0.420 | 0.412 | **1.064** | 0.39× | 0.39× |
| 1 | 0.051 | 0.052 | 0.029 | **0.045** | 1.13× | 0.66× |
| 16 | 0.059 | 0.054 | 0.074 | **0.068** | 0.87× | 1.10× |

The original V100-oriented code was also measured on the H200, before changing the kernels. This compares implementations on the same physical GPU, rather than comparing V100 against H200:

| tokens M | original W4 on H200, ms | optimized W4 on H200, ms | speedup |
|---|---|---|---|
| 128 | 0.247 | 0.166 | 1.49× |
| 512 | 0.389 | 0.316 | 1.23× |
| 2048 | 1.186 | 1.064 | 1.12× |
| 1 | 0.099 | 0.045 | 2.21× |
| 16 | 0.114 | 0.068 | 1.68× |

Raw results: [optimized GEMMs](results/h200/gemm.json), [original kernels on H200](results/h200/gemm_before.json), [per-shape timings](results/h200/logs/bench_gemm.log).

Native bf16 removes the V100 baseline disadvantage: cuBLAS bf16 and fp16 are now comparable. INT4 compression alone does not ensure a speedup; unpacking/dequantization and the chosen tile layout still cost time, especially for prefill. Ratios below 1 in the table mean W4 is slower than that baseline.

## 4–5. Llama-3.2-1B-Instruct evaluation

Checkpoint: `unsloth/Llama-3.2-1B-Instruct`, revision `5a8abab4a5d6f164389b1079fb721cfab8d7126c`. Every row evaluates the WikiText-2-raw-v1 **test** split using all **141 non-overlapping windows of 2048 tokens**, predicting tokens 2–2048 in each window. Lower perplexity is better.

Prefill below is a CUDA-graph forward pass over one 2048-token window, including attention and the full logits projection. The JSON files separately retain eager wall time averaged across all perplexity windows. Decode uses a 128-token prompt, 128 generated tokens, a static KV cache, and an autoregressive CUDA graph that includes token copying and cache-position updates. Its tok/s column measures the complete replay loop; GPU-event and eager timings are also saved. These timing changes make the H200 model rows internally comparable, but not directly interchangeable with the historical V100 timing columns.

`lm_head` stays bf16 by default and shares the embedding table. The `+lm_head` variant unties and quantizes that projection while preserving bf16 input embeddings. It therefore occupies more total memory than default W4: the formerly shared projection now needs a separate packed copy. Peak memory is allocated GPU memory during perplexity evaluation, in GiB; it is not total reserved memory.

| model | WikiText-2 PPL ↓ | prefill graph, ms / 2048 tok | prefill tok/s | decode tok/s, graph | linear weights, MiB | whole model, MiB | peak, GiB |
|---|---|---|---|---|---|---|---|---|
| bf16 / cuBLAS | 13.165 | 14.03 | 145,963 | 379.4 | 1856 | 2357 | 4.84 |
| fp16 / cuBLAS | 13.160 | 14.27 | 143,521 | 375.3 | 1856 | 2357 | 4.84 |
| Triton W16, native bf16 | 13.165 | 14.11 | 145,150 | 432.7 | 1856 | 2357 | 4.81 |
| **Triton W4 g128** | 16.431 | 26.01 | 78,734 | 415.7 | 493 | 994 | 3.48 |
| W4 g128 + lm_head | 16.815 | 28.78 | 71,164 | 419.8 | 626 | 1127 | 3.64 |
| W4 g128 dequantized → torch bf16 | 16.431 | 13.99 | 146,400 | 380.2 | 1856 | 2357 | 4.84 |
| W4 g64 | 15.419 | 26.86 | 76,246 | 414.6 | 522 | 1023 | 3.54 |
| W4 g32 | 14.806 | 31.27 | 65,502 | 410.4 | 580 | 1081 | 3.60 |

* **Kernel quality:** native Triton W4 g128 gives PPL **16.431**, versus **16.431** for the same weights dequantized and evaluated through torch bf16 (absolute difference 0.0000). Compare these two rows to isolate kernel error from quantization error.
* **Quantization quality:** bf16 gives PPL 13.165; plain INT4 g128 gives 16.431. Smaller groups improve it to 15.419 at g64 and 14.806 at g32, with additional metadata storage. This implementation uses plain round-to-nearest quantization, without calibration.
* **Memory:** default W4 reduces decoder-linear weights from 1856 to 493 MiB and the whole model from 2357 to 994 MiB (2.37× smaller).
* **End-to-end speed:** W4/bf16 speedup is 0.54× for prefill and 1.10× for graph decode. The full-model measurements, rather than the compression ratio, determine the actual performance benefit.

Per-mode JSON files and generated samples are in [results/h200](results/h200); complete run logs are in [results/h200/logs](results/h200/logs).

## Correctness

**85 tests passed on H200.** The suite checks packed quantization against torch, compression and quantization error, fp16/bf16 GEMMs on all four distinct Llama projection shapes, non-divisible output dimensions, 3D inputs and bias. Added cases cover groups 32/64 across the decode/prefill boundary, non-contiguous activations, repeated cached launches and deterministic scratch-buffer reduction. Native bf16 inputs above fp16's maximum representable value are checked explicitly to catch accidental fp16 saturation. See [pytest.log](results/h200/logs/pytest.log).
