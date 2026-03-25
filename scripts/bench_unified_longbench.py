#!/usr/bin/env python3
"""
Unified latency benchmark: AR / EAGLE / TriForce on the same LongBench samples.

Metrics per sample:
  total_latency  – wall time from first input token to last output token (s)
  prefill_time   – time to process the context (s)
  decode_time    – time for gen_len decode steps (s)
  prefill_tps    – ctx_tokens / prefill_time
  decode_tps     – gen_len / decode_time
  total_tps      – gen_len / total_latency
  acceptance_rate – tokens accepted per draft step (SD methods only)

Timing approach
  ar      : explicit torch.cuda.synchronize() boundaries around each phase
  eagle   : eagenerate(log=True) returns internal prefill_time; decode=wall-prefill
  triforce: TriForce() / Autoregressive() start their timer AFTER prefill internally,
            returning decode_tps; we derive prefill = wall_external - gen_len/decode_tps

Environments
  --method ar,eagle  → conda run -n specreason ...
  --method triforce  → conda run -n QWen_DTD  ...

Usage
  python scripts/bench_unified_longbench.py \\
      --method eagle \\
      --base-model NousResearch/Yarn-Llama-2-7b-128k \\
      --ea-model   yuhuili/EAGLE-llama2-chat-7B \\
      --data       outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl \\
      --n-samples  10 --gen-len 32 --max-input-tokens 3000 \\
      --output     outputs/bench_unified_eagle.jsonl

  python scripts/bench_unified_longbench.py \\
      --method ar \\
      --base-model NousResearch/Yarn-Llama-2-7b-128k \\
      --output outputs/bench_unified_ar.jsonl

  python scripts/bench_unified_longbench.py \\
      --method triforce \\
      --base-model NousResearch/Yarn-Llama-2-7b-128k \\
      --draft-model JackFram/llama-68m \\
      --prefill 3000 \\
      --output outputs/bench_unified_triforce.jsonl
"""

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(sample: dict) -> str:
    ctx = sample.get("context", "")
    inp = sample.get("input", "")
    if inp:
        return f"{ctx}\n\nQuestion: {inp}\nAnswer:"
    return f"{ctx}\n\nPlease summarize the text above.\nSummary:"


def _print_sample(i, n, method, ctx, gen_len,
                  total_latency, prefill_time, decode_time,
                  prefill_tps, decode_tps, accept=None):
    acc_str = f"  accept={accept:.2f}" if accept is not None else ""
    print(
        f"[{i+1}/{n}] {method}  ctx={ctx}  gen={gen_len}  "
        f"total={total_latency*1000:.0f}ms  "
        f"prefill={prefill_time*1000:.0f}ms({prefill_tps:.0f}t/s)  "
        f"decode={decode_time*1000:.0f}ms({decode_tps:.1f}t/s)"
        + acc_str
    )


def make_record(method, model, draft_model, sample, i,
                ctx_tokens, gen_len,
                total_latency, prefill_time, decode_time,
                prefill_tps, decode_tps, accept=None, **extra):
    return {
        "method": method,
        "model": model,
        "draft_model": draft_model,
        "sample_id": i,
        "dataset": sample.get("dataset", ""),
        "ctx_tokens": ctx_tokens,
        "gen_tokens": gen_len,
        "total_latency": round(total_latency, 4),
        "prefill_time": round(prefill_time, 4),
        "decode_time": round(decode_time, 4),
        "prefill_tps": round(prefill_tps, 1),
        "decode_tps": round(decode_tps, 2),
        "total_tps": round(gen_len / total_latency, 2) if total_latency > 0 else 0.0,
        "acceptance_rate": round(accept, 3) if accept is not None else None,
        **extra,
    }


# ──────────────────────────────────────────────────────────────────────────────
# AR – plain transformers
# ──────────────────────────────────────────────────────────────────────────────

def run_ar_method(args, samples, out_f):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[AR] Loading {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="cuda:0"
    ).eval()
    print("[AR] Model loaded.")

    def _run_one(ids):
        """Returns (prefill_time, decode_time, gen_len)."""
        device = ids.device
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(ids, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        pf = time.perf_counter() - t0

        past = out.past_key_values
        logits = out.logits[:, -1, :]
        if args.temperature > 0:
            probs = torch.softmax(logits / args.temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        del out

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(args.gen_len - 1):
            with torch.inference_mode():
                out = model(next_tok, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            if args.temperature > 0:
                probs = torch.softmax(logits / args.temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        dec = time.perf_counter() - t1
        return pf, dec

    def tokenize(sample):
        return (
            tokenizer.encode(build_prompt(sample), return_tensors="pt")
            [:, : args.max_input_tokens].to("cuda:0")
        )

    # Warmup
    print("[AR] Warming up ...")
    w_ids = tokenize(samples[0])
    _run_one(w_ids)
    print("[AR] Warmup done.")

    decode_tps_list, prefill_tps_list, total_tps_list = [], [], []

    for i, sample in enumerate(samples):
        ids = tokenize(sample)
        ctx_tokens = ids.shape[1]

        pf, dec = _run_one(ids)
        total = pf + dec
        dec_tps = args.gen_len / dec if dec > 0 else 0.0
        pf_tps = ctx_tokens / pf if pf > 0 else 0.0

        decode_tps_list.append(dec_tps)
        prefill_tps_list.append(pf_tps)
        total_tps_list.append(args.gen_len / total)

        rec = make_record("ar", args.base_model, None, sample, i,
                          ctx_tokens, args.gen_len,
                          total, pf, dec, pf_tps, dec_tps)
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        _print_sample(i, len(samples), "AR", ctx_tokens, args.gen_len,
                      total, pf, dec, pf_tps, dec_tps)

    _print_final("AR", args.base_model, len(samples),
                 decode_tps_list, prefill_tps_list, total_tps_list)


# ──────────────────────────────────────────────────────────────────────────────
# EAGLE
# ──────────────────────────────────────────────────────────────────────────────

def run_eagle_method(args, samples, out_f):
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
    ).eval()
    tokenizer = model.get_tokenizer()
    print("[EAGLE] Model loaded.")

    def _run_one(ids):
        """Returns (wall, prefill_time, new_token, idx, accept_rate)."""
        input_tensor = torch.as_tensor([ids]).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids, new_token, idx, pf = model.eagenerate(
                input_tensor,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_length=args.max_length,
                max_new_tokens=args.gen_len,
                log=True,
            )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        accept = int(new_token) / max(int(idx), 1)
        return wall, float(pf), int(new_token), accept

    def tokenize(sample):
        return tokenizer([build_prompt(sample)]).input_ids[0][: args.max_input_tokens]

    # Warmup
    print("[EAGLE] Warming up ...")
    _run_one(tokenize(samples[0]))
    print("[EAGLE] Warmup done.")

    decode_tps_list, prefill_tps_list, total_tps_list = [], [], []

    for i, sample in enumerate(samples):
        ids = tokenize(sample)
        ctx_tokens = len(ids)

        wall, pf, gen_tokens, accept = _run_one(ids)
        dec = max(wall - pf, 1e-9)
        dec_tps = args.gen_len / dec
        pf_tps = ctx_tokens / pf if pf > 0 else 0.0

        decode_tps_list.append(dec_tps)
        prefill_tps_list.append(pf_tps)
        total_tps_list.append(args.gen_len / wall)

        rec = make_record("eagle", args.base_model, args.ea_model, sample, i,
                          ctx_tokens, args.gen_len,
                          wall, pf, dec, pf_tps, dec_tps, accept)
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        _print_sample(i, len(samples), "EAGLE", ctx_tokens, args.gen_len,
                      wall, pf, dec, pf_tps, dec_tps, accept)

    _print_final("EAGLE", args.base_model, len(samples),
                 decode_tps_list, prefill_tps_list, total_tps_list)


# ──────────────────────────────────────────────────────────────────────────────
# TriForce
# ──────────────────────────────────────────────────────────────────────────────

def run_triforce_method(args, samples, out_f):
    TRIFORCE_ROOT = os.path.join(ROOT, "TriForce")
    sys.path.insert(0, TRIFORCE_ROOT)

    from transformers import AutoTokenizer
    from models.modeling_llama import LlamaForCausalLM
    from models.modeling_llama_68m import LlamaForCausalLM as LlamaForCausalLM_68M
    from models.cache import FlashSimpleCache, StreamingLLMEvictionCache, RetrievalCache
    from utils.decoding import Autoregressive, TriForce
    from utils.graph_infer import GraphInferenceEngine

    prefill = args.prefill
    gamma = args.gamma
    top_k = -1

    print(f"[TriForce] Loading {args.base_model} ...")
    target = LlamaForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="cuda:0"
    ).eval()
    print(f"[TriForce] Loading draft {args.draft_model} ...")
    draft = LlamaForCausalLM_68M.from_pretrained(
        args.draft_model, torch_dtype=torch.float16, device_map="cuda:0"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, legacy=False)

    draft_cache_budget = 256
    recent_size = draft_cache_budget - 16 - gamma
    cache = FlashSimpleCache(target, prefill + args.gen_len + 16)
    graph_cache = RetrievalCache(target, max_budget=args.budget, prefill=prefill,
                                 gamma=gamma, chunk_size=args.chunk_size)
    draft_cache = StreamingLLMEvictionCache(draft, start_size=16,
                                            recent_size=recent_size, gamma=gamma)
    graph_engine = GraphInferenceEngine(target, cache, graph_cache, draft, draft_cache)
    graph_engine.initialize_cuda_graph(gamma, probs=True,
                                       temperature=args.temperature, top_p=args.top_p)
    print("[TriForce] CUDA graph compiled.")

    def tokenize(sample):
        return (
            tokenizer.encode(build_prompt(sample), return_tensors="pt")
            .to(target.device)[:, :prefill]
        )

    # ── timing helpers ──────────────────────────────────────────────────────
    # TriForce() now supports explicit phase timing boundaries:
    # prefill_time and decode_time are measured internally and returned.
    # ────────────────────────────────────────────────────────────────────────
    def _run_tf(ids):
        accept_rate, decode_tps, pf, dec, total, stage_stats = TriForce(
            tokenizer, graph_engine, ids,
            gamma=gamma, max_len=args.gen_len,
            top_k=top_k, top_p=args.top_p, temperature=args.temperature,
            return_timing=True,
            use_draft=args.use_draft,
        )
        return float(total), float(pf), float(dec), float(accept_rate), float(decode_tps), stage_stats

    def _run_ar(ids):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ar_speed = Autoregressive(
            tokenizer, graph_engine, ids,
            max_len=args.gen_len, top_k=top_k,
            top_p=args.top_p, temperature=args.temperature,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        dec = args.gen_len / float(ar_speed) if ar_speed > 0 else wall
        pf = max(wall - dec, 1e-9)
        return wall, pf, dec

    # Warmup
    print("[TriForce] Warming up ...")
    w_ids = tokenize(samples[0])
    _run_ar(w_ids)
    for _ in range(3):
        _run_tf(w_ids)
    print("[TriForce] Warmup done.")

    decode_tps_list, prefill_tps_list, total_tps_list = [], [], []

    for i, sample in enumerate(samples):
        ids = tokenize(sample)
        ctx_tokens = ids.shape[1]

        wall, pf, dec, accept, dec_tps, stage_stats = _run_tf(ids)
        pf_tps = ctx_tokens / pf if pf > 0 else 0.0

        decode_tps_list.append(dec_tps)
        prefill_tps_list.append(pf_tps)
        total_tps_list.append(args.gen_len / wall)

        rec = make_record("triforce", args.base_model, args.draft_model, sample, i,
                          ctx_tokens, args.gen_len,
                          wall, pf, dec, pf_tps, dec_tps, accept,
                          budget=args.budget, gamma=gamma,
                          stage1_acceptance_rate=stage_stats.get("stage1_acceptance_rate"),
                          stage1_accepted_len=stage_stats.get("stage1_accepted_len"),
                          stage1_proposed_len=stage_stats.get("stage1_proposed_len"),
                          stage2_acceptance_rate=stage_stats.get("stage2_acceptance_rate"),
                          stage2_accepted_len=stage_stats.get("stage2_accepted_len"),
                          stage2_proposed_len=stage_stats.get("stage2_proposed_len"))
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        _print_sample(i, len(samples), "TriForce", ctx_tokens, args.gen_len,
                      wall, pf, dec, pf_tps, dec_tps, accept)

    _print_final("TriForce", args.base_model, len(samples),
                 decode_tps_list, prefill_tps_list, total_tps_list)


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def _mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def _print_final(method, model, n, decode_tps_list, prefill_tps_list, total_tps_list):
    print(f"\n{'='*60}")
    print(f"  {method} Summary")
    print(f"  Model:             {model}")
    print(f"  Samples:           {n}")
    print(f"  Mean decode tok/s: {_mean(decode_tps_list):.2f}")
    print(f"  Mean prefill tok/s:{_mean(prefill_tps_list):.1f}")
    print(f"  Mean total  tok/s: {_mean(total_tps_list):.2f}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified AR / EAGLE / TriForce latency benchmark."
    )
    parser.add_argument("--method", required=True, choices=["ar", "eagle", "triforce"])
    # Data
    parser.add_argument("--data", default="outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl")
    parser.add_argument("--n-samples", type=int, default=10)
    # Models
    parser.add_argument("--base-model", default="NousResearch/Yarn-Llama-2-7b-128k")
    parser.add_argument("--ea-model",   default="yuhuili/EAGLE-llama2-chat-7B",
                        help="EAGLE draft model path (method=eagle)")
    parser.add_argument("--draft-model", default="JackFram/llama-68m",
                        help="TriForce draft model path (method=triforce)")
    # Generation
    parser.add_argument("--gen-len",            type=int,   default=32)
    parser.add_argument("--max-input-tokens",   type=int,   default=3000)
    parser.add_argument("--temperature",        type=float, default=0.6)
    parser.add_argument("--top-p",              type=float, default=0.9)
    # EAGLE-specific
    parser.add_argument("--top-k",     type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Max total sequence length for EAGLE KV cache")
    parser.add_argument("--use-eagle3", action="store_true")
    # TriForce-specific
    parser.add_argument("--prefill",    type=int, default=3000,
                        help="Fixed context length for TriForce CUDA graph")
    parser.add_argument("--budget",     type=int, default=2048,
                        help="TriForce retrieval KV budget")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--gamma",      type=int, default=6,
                        help="TriForce draft tokens per step")
    parser.add_argument("--use-draft",  action="store_true", default=True,
                        help="Use llama-68m draft in TriForce (default True; set --no-use-draft to disable)")
    parser.add_argument("--no-use-draft", action="store_false", dest="use_draft",
                        help="Disable llama-68m draft, use retrieval cache only")
    # Output
    parser.add_argument("--output", default="outputs/bench_unified_result.jsonl")
    args = parser.parse_args()

    # Sync max-input-tokens / prefill for TriForce
    if args.method == "triforce":
        args.max_input_tokens = args.prefill

    data_path = os.path.join(ROOT, args.data) if not os.path.isabs(args.data) else args.data
    with open(data_path) as f:
        samples = [json.loads(l) for l in f]
    samples = samples[: args.n_samples]
    print(f"[bench] method={args.method}  {len(samples)} samples from {data_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as out_f:
        if args.method == "ar":
            run_ar_method(args, samples, out_f)
        elif args.method == "eagle":
            run_eagle_method(args, samples, out_f)
        elif args.method == "triforce":
            run_triforce_method(args, samples, out_f)

    print(f"\n[bench] Results written to {args.output}")


if __name__ == "__main__":
    main()
