#!/usr/bin/env python3
"""
TriForce benchmark on LongBench long-context data.

Env:  QWen_DTD  (has flash_attn 2.8.x)
Usage:
    python scripts/bench_triforce_longbench.py \
        --data outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl \
        --target-model NousResearch/Yarn-Llama-2-7b-128k \
        --draft-model JackFram/llama-68m \
        --n-samples 10 \
        --prefill 20000 \
        --gen-len 200 \
        --budget 4096 \
        --output outputs/bench_triforce_results.jsonl

TriForce uses CUDA-graph inference compiled for a fixed `prefill` length.
All inputs are truncated to `--prefill` tokens.
"""

import argparse
import json
import os
import sys
import time

import torch

TRIFORCE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "TriForce")
)
sys.path.insert(0, TRIFORCE_ROOT)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_prompt(sample: dict) -> str:
    """Model-agnostic prompt builder aligned with EAGLE benchmark script."""
    ctx = sample.get("context", "")
    inp = sample.get("input", "")
    if inp:
        return f"{ctx}\n\nQuestion: {inp}\nAnswer:"
    return f"{ctx}\n\nPlease summarize the text above.\nSummary:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl")
    parser.add_argument("--target-model", default="NousResearch/Yarn-Llama-2-7b-128k")
    parser.add_argument("--draft-model", default="JackFram/llama-68m")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--prefill", type=int, default=20000,
                        help="Fixed context length (all inputs truncated to this)")
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--budget", type=int, default=4096,
                        help="KV cache retrieval budget (tokens)")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--gamma", type=int, default=6,
                        help="Draft tokens per speculation step")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--output", default="outputs/bench_triforce_results.jsonl")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from models.modeling_llama import LlamaForCausalLM
    from models.modeling_llama_68m import LlamaForCausalLM as LlamaForCausalLM_68M
    from models.cache import FlashSimpleCache, StreamingLLMEvictionCache, RetrievalCache
    from utils.decoding import Autoregressive, TriForce
    from utils.graph_infer import GraphInferenceEngine

    prefill = args.prefill
    gen_len = args.gen_len
    gamma = args.gamma
    top_p = args.top_p
    temperature = args.temp
    max_budget = args.budget
    chunk_size = args.chunk_size
    top_k = -1

    print(f"[TriForce] Loading {args.target_model} ...")
    target = LlamaForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.float16, device_map="cuda:0"
    ).eval()

    print(f"[TriForce] Loading draft {args.draft_model} ...")
    draft = LlamaForCausalLM_68M.from_pretrained(
        args.draft_model, torch_dtype=torch.float16, device_map="cuda:0"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True, legacy=False)

    draft_cache_budget = 256
    recent_size = draft_cache_budget - 16 - gamma

    cache = FlashSimpleCache(target, prefill + gen_len + 16)
    graph_cache = RetrievalCache(target, max_budget=max_budget, prefill=prefill,
                                  gamma=gamma, chunk_size=chunk_size)
    draft_cache = StreamingLLMEvictionCache(draft, start_size=16,
                                             recent_size=recent_size, gamma=gamma)
    graph_engine = GraphInferenceEngine(target, cache, graph_cache, draft, draft_cache)
    graph_engine.initialize_cuda_graph(gamma, probs=True,
                                        temperature=temperature, top_p=top_p)
    print("[TriForce] CUDA graph compiled.")

    # Load LongBench data
    data_path = os.path.join(ROOT, args.data) if not os.path.isabs(args.data) else args.data
    with open(data_path) as f:
        samples = [json.loads(l) for l in f]
    samples = samples[:args.n_samples]
    print(f"[TriForce] {len(samples)} samples from {data_path}")

    # Tokenize all samples with the same prompt template used by EAGLE benchmark.
    def tokenize_sample(sample):
        text = build_prompt(sample)
        ids = tokenizer.encode(text, return_tensors="pt")
        return ids  # shape: (1, seq_len)

    tokenized = [tokenize_sample(s) for s in samples]

    # Warmup: 1 pass of Autoregressive, 3 of TriForce
    print("[TriForce] Warming up ...")
    w_ids = tokenized[0].to(target.device)[:, :prefill]
    Autoregressive(tokenizer, graph_engine, w_ids, max_len=gen_len,
                   top_k=top_k, top_p=top_p, temperature=temperature)
    for _ in range(3):
        TriForce(tokenizer, graph_engine, w_ids, gamma=gamma, max_len=gen_len,
                 top_k=top_k, top_p=top_p, temperature=temperature)
    print("[TriForce] Warmup done.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "w")

    total_sd_tps, total_ar_tps, speedups = [], [], []

    for i, (sample, tok_ids) in enumerate(zip(samples, tokenized)):
        input_ids = tok_ids.to(target.device)[:, :prefill]
        ctx_tokens = input_ids.shape[1]

        # ---- TriForce generation ----
        t0 = time.perf_counter()
        accept_rate, tf_speed = TriForce(
            tokenizer, graph_engine, input_ids,
            gamma=gamma, max_len=gen_len,
            top_k=top_k, top_p=top_p, temperature=temperature,
        )
        wall_tf = time.perf_counter() - t0
        gen_tokens = args.gen_len
        # decode-only tok/s: tf_speed is measured AFTER prefill inside TriForce()
        decode_tps = float(tf_speed)
        # end-to-end tok/s (includes prefill, kept for reference)
        total_tps = float(args.gen_len) / wall_tf if wall_tf > 0 else 0.0

        # ---- AR baseline ----
        ar_decode_tps = None
        speedup = None
        if not args.skip_baseline:
            t0_ar = time.perf_counter()
            ar_speed = Autoregressive(
                tokenizer, graph_engine, input_ids,
                max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature,
            )
            wall_ar = time.perf_counter() - t0_ar
            # ar_speed is decode-only (timer starts after prefill inside Autoregressive())
            ar_decode_tps = float(ar_speed)
            # decode-only speedup: what speculative decoding actually accelerates
            speedup = decode_tps / ar_decode_tps if ar_decode_tps > 0 else None
            total_ar_tps.append(ar_decode_tps)
            speedups.append(speedup)

        total_sd_tps.append(decode_tps)

        record = {
            "system": "triforce",
            "model": args.target_model,
            "draft_model": args.draft_model,
            "budget": args.budget,
            "gamma": gamma,
            "sample_id": i,
            "dataset": sample.get("dataset", ""),
            "ctx_tokens": ctx_tokens,
            "gen_tokens": gen_tokens,
            "wall_time": round(wall_tf, 4),
            "tokens_per_sec": round(total_tps, 2),   # end-to-end (reference)
            "decode_tps": round(decode_tps, 2),       # decode-only (primary)
            "acceptance_rate": round(float(accept_rate), 3),
            "ar_tokens_per_sec": round(ar_decode_tps, 2) if ar_decode_tps else None,  # decode-only
            "speedup": round(speedup, 3) if speedup else None,
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        print(f"[{i+1}/{len(samples)}] ctx={ctx_tokens} gen~{gen_tokens} "
              f"TriForce decode={decode_tps:.1f} total={total_tps:.1f} tok/s accept={accept_rate:.2f}"
              + (f" AR(dec)={ar_decode_tps:.1f} tok/s speedup={speedup:.2f}x" if ar_decode_tps else ""))

    out_f.close()

    mean_decode = sum(total_sd_tps) / len(total_sd_tps)
    print(f"\n=== TriForce Summary ===")
    print(f"  Samples:               {len(samples)}")
    print(f"  Prefill length (toks): {prefill}")
    print(f"  Mean TriForce decode tok/s:  {mean_decode:.2f}  (decode-only, speedup vs. bench_ar_unified)")
    if total_ar_tps:
        mean_ar = sum(total_ar_tps) / len(total_ar_tps)
        mean_sp = sum(speedups) / len(speedups)
        print(f"  Mean AR(decode) tok/s:       {mean_ar:.2f}")
        print(f"  Mean decode speedup:         {mean_sp:.2f}x")
    print(f"  Output:                {args.output}")


if __name__ == "__main__":
    main()
