#!/usr/bin/env python3
"""
Unified AR baseline for Yarn-Llama-2-7b-128k using plain transformers.

Measures prefill and decode phases *separately*, providing a neutral reference
for computing speedup of EAGLE / TriForce.

Env:  specreason  (transformers >= 4.48)  OR  QWen_DTD (transformers >= 4.37)

Usage:
    python scripts/bench_ar_unified_longbench.py \
        --model NousResearch/Yarn-Llama-2-7b-128k \
        --data  outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl \
        --n-samples 10 \
        --max-input-tokens 3000 \
        --gen-len 32 \
        --output outputs/bench_ar_unified_llama2.jsonl
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_prompt(sample: dict) -> str:
    """Same prompt template as bench_eagle_longbench / bench_triforce_longbench."""
    ctx = sample.get("context", "")
    inp = sample.get("input", "")
    if inp:
        return f"{ctx}\n\nQuestion: {inp}\nAnswer:"
    return f"{ctx}\n\nPlease summarize the text above.\nSummary:"


def run_ar(model, input_ids, gen_len: int, temperature: float = 0.6, top_p: float = 0.9):
    """
    Run AR generation with explicit prefill / decode phase separation.

    Phase 1 – Prefill: one full forward pass over the context, KV cache built.
    Phase 2 – Decode: gen_len single-token steps using the cached KV.

    Returns:
        prefill_time  (s)
        decode_time   (s)  — covers exactly gen_len steps
        gen_len       (int)
    """
    device = input_ids.device

    # ---------- Prefill ----------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    past = out.past_key_values
    # Sample first decode token from the last prefill logit
    logits = out.logits[:, -1, :]
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
    else:
        next_token = logits.argmax(dim=-1, keepdim=True)

    del out  # free intermediate activations

    # ---------- Decode loop ----------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(gen_len - 1):
        with torch.inference_mode():
            out = model(next_token, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t0

    return prefill_time, decode_time, gen_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Yarn-Llama-2-7b-128k")
    parser.add_argument("--data", default="outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--max-input-tokens", type=int, default=3000,
                        help="Truncate context to this many tokens")
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", default="outputs/bench_ar_unified_llama2.jsonl")
    args = parser.parse_args()

    print(f"[AR] Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda:0"
    ).eval()
    print("[AR] Model loaded.")

    data_path = os.path.join(ROOT, args.data) if not os.path.isabs(args.data) else args.data
    with open(data_path) as f:
        samples = [json.loads(l) for l in f]
    samples = samples[: args.n_samples]
    print(f"[AR] {len(samples)} samples from {data_path}")

    # ---------- Warmup ----------
    print("[AR] Warming up ...")
    ids = (
        tokenizer.encode(build_prompt(samples[0]), return_tensors="pt")[
            :, : args.max_input_tokens
        ].to("cuda:0")
    )
    run_ar(model, ids, args.gen_len, args.temperature, args.top_p)
    print("[AR] Warmup done.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "w")

    decode_tps_list, total_tps_list = [], []

    for i, sample in enumerate(samples):
        ids = (
            tokenizer.encode(build_prompt(sample), return_tensors="pt")[
                :, : args.max_input_tokens
            ].to("cuda:0")
        )
        ctx_tokens = ids.shape[1]

        prefill_time, decode_time, n_gen = run_ar(
            model, ids, args.gen_len, args.temperature, args.top_p
        )

        decode_tps = float(n_gen) / decode_time if decode_time > 0 else 0.0
        total_tps = float(n_gen) / (prefill_time + decode_time)

        decode_tps_list.append(decode_tps)
        total_tps_list.append(total_tps)

        record = {
            "system": "ar_unified",
            "model": args.model,
            "sample_id": i,
            "dataset": sample.get("dataset", ""),
            "ctx_tokens": ctx_tokens,
            "gen_tokens": n_gen,
            "prefill_time": round(prefill_time, 4),
            "decode_time": round(decode_time, 4),
            "decode_tps": round(decode_tps, 2),
            "total_tps": round(total_tps, 2),
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        print(
            f"[{i+1}/{len(samples)}] ctx={ctx_tokens} "
            f"prefill={prefill_time*1000:.0f}ms  decode={decode_time*1000:.0f}ms  "
            f"decode_tps={decode_tps:.1f}  total_tps={total_tps:.1f}"
        )

    out_f.close()

    mean_decode = sum(decode_tps_list) / len(decode_tps_list)
    mean_total = sum(total_tps_list) / len(total_tps_list)
    print(f"\n=== AR Unified Baseline ===")
    print(f"  Model:              {args.model}")
    print(f"  Samples:            {len(samples)}")
    print(f"  Gen len:            {args.gen_len}")
    print(f"  Max ctx (tokens):   {args.max_input_tokens}")
    print(f"  Mean decode tok/s:  {mean_decode:.2f}")
    print(f"  Mean total tok/s:   {mean_total:.2f}")
    print(f"  Output:             {args.output}")


if __name__ == "__main__":
    main()
