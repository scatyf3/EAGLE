#!/usr/bin/env python3
"""
LongSpec benchmark on LongBench long-context data.

Env:  QWen_DTD  (has flash_attn 2.8.x)
Usage:
    python scripts/bench_longspec_longbench.py \
        --data outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl \
        --target-model gradientai/Llama-3-8B-Instruct-262k \
        --draft-model sail/longspec-Llama-3-8B-Instruct-262k \
        --n-samples 10 \
        --gen-len 200 \
        --output outputs/bench_longspec_results.jsonl

Runs both LongSpec (tree_spec_generate) and AR baseline (vanilla_generate) on
each sample, writes one JSON record per sample with timing & acceptance-rate.
"""

import argparse
import json
import os
import sys
import time

import torch

# Add LongSpec test dir to path
LONGSPEC_TEST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "LongSpec", "longspec", "test")
)
sys.path.insert(0, LONGSPEC_TEST)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_prompt(sample: dict, tokenizer) -> torch.Tensor:
    """Format LongBench sample using Llama-3 instruct template, return input_ids."""
    ctx = sample.get("context", "")
    inp = sample.get("input", "")
    dataset = sample.get("dataset", "")

    # Use dataset-specific prompt if known, otherwise generic summarisation
    if dataset in ("qmsum",):
        text = (
            f"You are given a meeting transcript and a query. "
            f"Answer the query in one or more sentences.\n\n"
            f"Transcript:\n{ctx}\n\nQuery: {inp}"
        )
    elif dataset in ("repobench-p",):
        text = f"Please complete the code given below.\n{ctx}\nNow, complete the code given."
    elif dataset in ("gov_report",):
        text = f"You are given a report by a government agency. Write a one-page summary.\n\nReport:\n{ctx}"
    elif dataset in ("multi_news",):
        text = f"You are given several news passages. Write a one-page summary of all news.\n\nNews:\n{ctx}"
    elif dataset in ("lcc",):
        text = f"Please complete the code given below.\n{ctx}\nNow, complete the code given."
    else:
        text = f"Please summarize the following text:\n\n{ctx}"
        if inp:
            text += f"\n\nQuery: {inp}"

    encoded = tokenizer(text, return_tensors="pt", padding=False)
    return encoded["input_ids"].cuda()


def run_tree_spec(model, input_ids, gen_len: int, tree_shape):
    prompt_length = input_ids.size(1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids, count, num, elapsed_time, spec_mask, draft_t, target_t = model.tree_spec_generate(
            input_ids,
            prompt_length=prompt_length,
            tree_shape=tree_shape,
            max_gen_len=gen_len,
            temperature=0.0,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    gen_tokens = int(count + num)  # count=draft accepted, num=LLM calls
    tps = gen_tokens / wall if wall > 0 else 0.0
    # acceptance_rate: tokens accepted per LLM verification step
    accept_rate = (count / num).item() if num > 0 else 0.0
    return prompt_length, gen_tokens, wall, tps, float(accept_rate)


def run_vanilla(model, input_ids, gen_len: int):
    prompt_length = input_ids.size(1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids, tokens, elapsed = model.vanilla_generate(
            input_ids,
            prompt_length=prompt_length,
            max_gen_len=gen_len,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    tps = int(tokens) / wall if wall > 0 else 0.0
    return int(tokens), wall, tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl")
    parser.add_argument("--target-model", default="gradientai/Llama-3-8B-Instruct-262k")
    parser.add_argument("--draft-model", default="sail/longspec-Llama-3-8B-Instruct-262k")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--max-input-tokens", type=int, default=30000, dest="max_input_tokens")
    parser.add_argument("--tree-shape", type=int, nargs="+", default=[4, 16, 16, 16, 16])
    parser.add_argument("--output", default="outputs/bench_longspec_results.jsonl")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoConfig
    from llama_glide import LlamaGlide

    print(f"[LongSpec] Loading {args.target_model} + {args.draft_model} ...")
    config = AutoConfig.from_pretrained(args.target_model)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    # Llama-3 tokens
    if "llama-3" in args.target_model.lower() or "llama3" in args.target_model.lower():
        config.pad_token_id = 128001
        config.eos_token_id = 128009

    model = LlamaGlide(config, args.target_model, args.draft_model)
    model.eval()

    # Load, filter, truncate data
    data_path = os.path.join(ROOT, args.data) if not os.path.isabs(args.data) else args.data
    with open(data_path) as f:
        samples = [json.loads(l) for l in f]
    samples = samples[:args.n_samples]
    print(f"[LongSpec] {len(samples)} samples from {data_path}")

    # Warmup
    print("[LongSpec] Warming up ...")
    with torch.inference_mode():
        w_ids = build_prompt(samples[0], tokenizer)
        w_ids = w_ids[:, :args.max_input_tokens]
        model.tree_spec_generate(
            w_ids, prompt_length=w_ids.size(1),
            tree_shape=args.tree_shape, max_gen_len=args.gen_len, temperature=0.0
        )
    torch.cuda.synchronize()
    print("[LongSpec] Warmup done.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "w")

    total_sd_tps, total_ar_tps, speedups = [], [], []

    for i, sample in enumerate(samples):
        input_ids = build_prompt(sample, tokenizer)
        input_ids = input_ids[:, :args.max_input_tokens]

        ctx, gen, wall, tps, accept = run_tree_spec(model, input_ids, args.gen_len, args.tree_shape)

        ar_tps_val = None
        speedup = None
        if not args.skip_baseline:
            ar_gen, ar_wall, ar_tps_val = run_vanilla(model, input_ids, args.gen_len)
            speedup = tps / ar_tps_val if ar_tps_val > 0 else None
            total_ar_tps.append(ar_tps_val)
            speedups.append(speedup)

        total_sd_tps.append(tps)

        record = {
            "system": "longspec",
            "model": args.target_model,
            "draft_model": args.draft_model,
            "tree_shape": args.tree_shape,
            "sample_id": i,
            "dataset": sample.get("dataset", ""),
            "ctx_tokens": ctx,
            "gen_tokens": gen,
            "wall_time": round(wall, 4),
            "tokens_per_sec": round(tps, 2),
            "acceptance_rate": round(accept, 3),
            "ar_tokens_per_sec": round(ar_tps_val, 2) if ar_tps_val else None,
            "speedup": round(speedup, 3) if speedup else None,
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        print(f"[{i+1}/{len(samples)}] ctx={ctx} gen={gen} "
              f"LongSpec={tps:.1f} tok/s accept={accept:.2f}"
              + (f" AR={ar_tps_val:.1f} tok/s speedup={speedup:.2f}x" if ar_tps_val else ""))

    out_f.close()

    mean_tps = sum(total_sd_tps) / len(total_sd_tps)
    print(f"\n=== LongSpec Summary ===")
    print(f"  Samples: {len(samples)}")
    print(f"  Mean LongSpec tokens/sec: {mean_tps:.2f}")
    if total_ar_tps:
        mean_ar = sum(total_ar_tps) / len(total_ar_tps)
        mean_sp = sum(speedups) / len(speedups)
        print(f"  Mean AR tokens/sec:       {mean_ar:.2f}")
        print(f"  Mean speedup:             {mean_sp:.2f}x")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
