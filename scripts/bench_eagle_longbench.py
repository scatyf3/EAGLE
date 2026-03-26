#!/usr/bin/env python3
"""
EAGLE benchmark on LongBench long-context data.

Env:  specreason
Usage:
    python scripts/bench_eagle_longbench.py \
        --data outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl \
        --base-model Qwen/Qwen3-4B \
        --ea-model AngelSlim/Qwen3-4B_eagle3 \
        --n-samples 10 \
        --gen-len 200 \
        --output outputs/bench_eagle_results.jsonl

The script runs both EAGLE (SD) and AR baseline on each sample, and writes one
JSON record per sample with timing & acceptance-rate metrics.
"""

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def build_prompt(sample: dict) -> str:
    """Build a model-agnostic plain-text prompt from a LongBench sample."""
    ctx = sample.get("context", "")
    inp = sample.get("input", "")
    if inp:
        prompt = f"{ctx}\n\nQuestion: {inp}\nAnswer:"
    else:
        prompt = f"{ctx}\n\nPlease summarize the text above.\nSummary:"
    return prompt


def warmup(model, tokenizer, prompt: str, gen_len: int, args):
    ids = tokenizer([prompt]).input_ids
    ids = ids[0][:args.max_input_tokens]
    input_tensor = torch.as_tensor([ids]).cuda()
    with torch.inference_mode():
        model.eagenerate(
            input_tensor,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_length=args.max_length,
            max_new_tokens=gen_len,
            log=True,
        )
    torch.cuda.synchronize()


def run_eagle(model, tokenizer, prompt: str, gen_len: int, args):
    ids = tokenizer([prompt]).input_ids
    ids = ids[0][:args.max_input_tokens]
    ctx_tokens = len(ids)
    input_tensor = torch.as_tensor([ids]).cuda()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids, new_token, idx, _pf = model.eagenerate(
            input_tensor,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_length=args.max_length,
            max_new_tokens=gen_len,
            log=True,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    gen_tokens = int(new_token)
    # _pf is the prefill_time (seconds) returned by eagenerate()
    prefill_time = float(_pf)
    decode_wall = max(wall - prefill_time, 1e-9)
    # decode-only tok/s: the metric that reflects SD's actual benefit
    decode_tps = float(gen_len) / decode_wall
    # end-to-end tok/s (kept for reference)
    tps = float(gen_len) / wall if wall > 0 else 0.0
    # idx = number of EAGLE draft steps; acceptance_rate = tokens/step
    accept_rate = gen_tokens / max(int(idx), 1)
    return ctx_tokens, gen_tokens, wall, tps, prefill_time, decode_tps, accept_rate


def run_baseline(model, tokenizer, prompt: str, gen_len: int, args):
    ids = tokenizer([prompt]).input_ids
    ids = ids[0][:args.max_input_tokens]
    ctx_tokens = len(ids)
    input_tensor = torch.as_tensor([ids]).cuda()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids, new_token, idx = model.naivegenerate(
            input_tensor,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_length=args.max_length,
            max_new_tokens=gen_len,
            log=True,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    gen_tokens = int(new_token)
    # Fair end-to-end metric: use fixed requested generation length over wall time.
    tps = float(gen_len) / wall if wall > 0 else 0.0
    return ctx_tokens, gen_tokens, wall, tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--ea-model", default="AngelSlim/Qwen3-1.7B_eagle3")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--max-input-tokens", type=int, default=30000, dest="max_input_tokens")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--output", default="outputs/bench_eagle_results.jsonl")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--use-eagle3", action="store_true")
    args = parser.parse_args()

    # Load model
    from eagle.model.ea_model import EaModel
    print(f"[EAGLE] Loading {args.base_model} + {args.ea_model} ...")
    model = EaModel.from_pretrained(
        base_model_path=args.base_model,
        ea_model_path=args.ea_model,
        total_token=-1,
        depth=6,
        top_k=10,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=args.use_eagle3,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    # Load data
    data_path = os.path.join(ROOT, args.data) if not os.path.isabs(args.data) else args.data
    with open(data_path) as f:
        samples = [json.loads(l) for l in f]
    samples = samples[:args.n_samples]
    print(f"[EAGLE] {len(samples)} samples loaded from {data_path}")

    # Warmup
    print("[EAGLE] Warming up ...")
    warmup(model, tokenizer, build_prompt(samples[0]), args.gen_len, args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "w")
    
    print(f"[EAGLE] Running {len(samples)} samples ...")
    total_sd_tps, total_ar_tps, speedups = [], [], []

    for i, sample in enumerate(samples):
        prompt = build_prompt(sample)

        ctx, gen, wall, tps, prefill_t, decode_tps, accept = run_eagle(
            model, tokenizer, prompt, args.gen_len, args
        )

        ar_tps_val = None
        speedup = None
        if not args.skip_baseline:
            _, ar_gen, ar_wall, ar_tps_val = run_baseline(model, tokenizer, prompt, args.gen_len, args)
            # Speedup vs internal AR baseline (end-to-end, for quick sanity check)
            speedup = tps / ar_tps_val if ar_tps_val > 0 else None
            total_ar_tps.append(ar_tps_val)
            speedups.append(speedup)

        total_sd_tps.append(decode_tps)

        record = {
            "system": "eagle",
            "model": args.base_model,
            "ea_model": args.ea_model,
            "sample_id": i,
            "dataset": sample.get("dataset", ""),
            "ctx_tokens": ctx,
            "gen_tokens": gen,
            "wall_time": round(wall, 4),
            "prefill_time": round(prefill_t, 4),
            "tokens_per_sec": round(tps, 2),        # end-to-end
            "decode_tps": round(decode_tps, 2),     # decode-only (primary)
            "acceptance_rate": round(accept, 3),
            "ar_tokens_per_sec": round(ar_tps_val, 2) if ar_tps_val else None,
            "speedup": round(speedup, 3) if speedup else None,
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        print(f"[{i+1}/{len(samples)}] ctx={ctx} gen={gen} "
              f"EAGLE decode={decode_tps:.1f} total={tps:.1f} tok/s accept={accept:.2f}"
              + (f" AR={ar_tps_val:.1f} tok/s speedup={speedup:.2f}x" if ar_tps_val else ""))

    out_f.close()

    # Summary
    mean_decode = sum(total_sd_tps) / len(total_sd_tps)
    print(f"\n=== EAGLE Summary ===")
    print(f"  Samples: {len(samples)}")
    print(f"  Mean EAGLE decode tok/s: {mean_decode:.2f}  (decode-only, speedup vs. bench_ar_unified)")
    if total_ar_tps:
        mean_ar = sum(total_ar_tps) / len(total_ar_tps)
        mean_sp = sum(speedups) / len(speedups)
        print(f"  Mean AR tok/s (internal, e2e): {mean_ar:.2f}")
        print(f"  Mean speedup (e2e vs internal): {mean_sp:.2f}x")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
