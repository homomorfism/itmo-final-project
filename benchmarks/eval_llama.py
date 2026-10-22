"""End-to-end evaluation of Llama-3.2-1B-Instruct with int4 Triton linear layers.

Measures wikitext-2 perplexity, prefill speed (forward pass over the perplexity windows),
decode speed (single-token steps, CUDA-graph captured to remove Python launch overhead)
and weight memory.

Usage:
    uv run python -m benchmarks.eval_llama --mode bf16
    uv run python -m benchmarks.eval_llama --mode w4 [--quantize-lm-head]
Modes:
    bf16        torch nn.Linear, bf16 (cuBLAS; no bf16 tensor cores on V100)
    fp16        torch nn.Linear, fp16 (cuBLAS tensor cores)
    triton-w16  bf16 weights, Triton W16A16 kernel (same kernel structure as W4, apples-to-apples)
    w4          int4 weights, Triton W4A16 kernel, bf16 activations
    w4-fake     int4 weights dequantized back to bf16, torch matmul (quality reference for w4)
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from int4_triton import Int4Linear, quantize_model_linears, w16a16_matmul

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"


class TritonW16Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, x):
        y = w16a16_matmul(x, self.weight)
        return y if self.bias is None else y + self.bias


def replace_with_triton_w16(model, skip):
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, TritonW16Linear(child))


def linear_weight_bytes(model, include_lm_head=False):
    total = 0
    for name, m in model.named_modules():
        if "lm_head" in name and not include_lm_head:
            continue
        if isinstance(m, Int4Linear):
            total += sum(b.numel() * b.element_size() for b in (m.qweight, m.scales, m.zeros))
        elif isinstance(m, (nn.Linear, TritonW16Linear)):
            total += m.weight.numel() * m.weight.element_size()
    return total


def model_bytes(model):
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() in seen:  # tied embeddings / lm_head
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def eval_ppl(model, tok, seqlen, max_windows=None):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    n = ids.shape[1] // seqlen
    if max_windows:
        n = min(n, max_windows)
    ids = ids[:, : n * seqlen].view(n, seqlen)

    # warmup (Triton autotuning / compilation happens here)
    for _ in range(2):
        model(ids[:1].cuda(), use_cache=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    nlls, fwd_time = [], 0.0
    for i in range(n):
        batch = ids[i : i + 1].cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(batch, use_cache=False).logits
        torch.cuda.synchronize()
        fwd_time += time.perf_counter() - t0
        loss = nn.functional.cross_entropy(logits[0, :-1].float(), batch[0, 1:], reduction="sum")
        nlls.append(loss.item())
    ppl = math.exp(sum(nlls) / (n * (seqlen - 1)))
    return {
        "ppl": ppl,
        "windows": n,
        "seqlen": seqlen,
        "prefill_total_s": fwd_time,
        "prefill_ms_per_window": fwd_time / n * 1e3,
        "prefill_tokens_per_s": n * seqlen / fwd_time,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


@torch.no_grad()
def bench_decode(model, prompt_len=128, new_tokens=128, use_graph=True):
    dev = "cuda"
    torch.manual_seed(0)
    prompt = torch.randint(1000, 20000, (1, prompt_len), device=dev)
    cache = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=prompt_len + new_tokens + 8, device=dev, dtype=model.dtype
    )
    out = model(input_ids=prompt, cache_position=torch.arange(prompt_len, device=dev), past_key_values=cache, use_cache=True)
    static_ids = out.logits[:, -1:].argmax(-1)
    static_pos = torch.tensor([prompt_len], device=dev)

    def step():
        logits = model(
            input_ids=static_ids,
            cache_position=static_pos,
            position_ids=static_pos[None],
            past_key_values=cache,
            use_cache=True,
        ).logits
        return logits[:, -1:].argmax(-1)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warmup: autotune + populate kernel launch caches
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = step()
        run = lambda: graph.replay() or static_out
    else:
        run = step

    tokens = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(new_tokens):
        nxt = run()
        static_ids.copy_(nxt)
        static_pos += 1
        tokens.append(nxt)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"decode_tokens_per_s": new_tokens / dt, "decode_ms_per_token": dt / new_tokens * 1e3, "cuda_graph": use_graph}


@torch.no_grad()
def sample_generation(model, tok):
    msgs = [{"role": "user", "content": "Explain in two sentences what weight quantization of neural networks is."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["bf16", "fp16", "triton-w16", "w4", "w4-fake"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--quantize-lm-head", action="store_true")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    dtype = torch.float16 if args.mode == "fp16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, attn_implementation="sdpa").cuda().eval()
    skip = () if args.quantize_lm_head else ("lm_head",)
    if args.quantize_lm_head:
        # lm_head is tied to the embedding matrix: untie so the embedding stays full precision
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())

    lin_bytes_before = linear_weight_bytes(model, args.quantize_lm_head)
    t0 = time.perf_counter()
    if args.mode == "w4":
        quantize_model_linears(model, args.group_size, skip=skip)
    elif args.mode == "w4-fake":
        quantize_model_linears(model, args.group_size, skip=skip, fake=True)
    elif args.mode == "triton-w16":
        replace_with_triton_w16(model, skip)
    torch.cuda.synchronize()
    quant_time = time.perf_counter() - t0

    res = {
        "mode": args.mode,
        "group_size": args.group_size if args.mode.startswith("w4") else None,
        "quantize_lm_head": args.quantize_lm_head,
        "device": torch.cuda.get_device_name(),
        "quantize_time_s": quant_time,
        "linear_weight_bytes_before": lin_bytes_before,
        "linear_weight_bytes": linear_weight_bytes(model, args.quantize_lm_head),
        "model_bytes": model_bytes(model),
    }
    print(json.dumps(res, indent=2))
    res.update(eval_ppl(model, tok, args.seqlen, args.max_windows))
    print(f"ppl={res['ppl']:.4f}  prefill {res['prefill_ms_per_window']:.1f} ms/window ({res['prefill_tokens_per_s']:.0f} tok/s)")
    if not args.no_decode:
        try:
            res.update(bench_decode(model, use_graph=True))
        except Exception as e:  # fall back to eager if the graph capture is not possible
            print("CUDA graph decode failed:", repr(e)[:300])
            res.update(bench_decode(model, use_graph=False))
        eager = bench_decode(model, use_graph=False)
        res["decode_eager_tokens_per_s"] = eager["decode_tokens_per_s"]
        print(f"decode {res['decode_tokens_per_s']:.1f} tok/s (graph={res['cuda_graph']}), eager {res['decode_eager_tokens_per_s']:.1f} tok/s")
    res["sample"] = sample_generation(model, tok)
    print("sample:", res["sample"])

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.mode + ("-lmhead" if args.quantize_lm_head else "")
    with open(os.path.join(args.out_dir, f"llama_{name}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
