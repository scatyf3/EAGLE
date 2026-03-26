#!/usr/bin/env python3

import argparse
import json
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template

from eagle.model.ea_model import EaModel
import eagle.model.ea_model as ea_model_module
import eagle.model.cnets as draft_cnets
import eagle.model.modeling_llama_kv as llama_kv
import eagle.model.modeling_qwen3_kv as qwen3_kv


BUCKET_ORDER = ["lt256", "256-512", "512-1k", "1k-2k", "2k-4k", "4k-8k"]


def truncate_input_ids(input_ids, max_input_tokens):
    if not max_input_tokens:
        return input_ids
    if len(input_ids[0]) <= max_input_tokens:
        return input_ids
    half = max_input_tokens // 2
    input_ids[0] = input_ids[0][:half] + input_ids[0][-half:]
    return input_ids


def set_flash_mode(enabled, availability):
    llama_kv._HAS_FLASH_ATTN = bool(enabled and availability["llama_flash"])
    llama_kv._HAS_TRITON_TREE_ATTN = bool(enabled and availability["llama_triton"])
    qwen3_kv._HAS_FLASH_ATTN = bool(enabled and availability["qwen3_flash"])
    qwen3_kv._HAS_TRITON_TREE_ATTN = bool(enabled and availability["qwen3_triton"])
    draft_cnets._HAS_FLASH_ATTN_DRAFT = bool(enabled and availability["draft_flash"])


def build_prompt(question, use_all_turns):
    turns = question.get("turns") or []
    turns = turns if use_all_turns else turns[:1]
    conv = get_conversation_template("qwen3")
    for turn in turns:
        conv.append_message(conv.roles[0], turn)
        conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), len(turns)


def mean(values):
    return sum(values) / len(values) if values else 0.0


@contextmanager
def instrument_decode_timers():
    """
    Time decode-loop components inside eagenerate:
    - target path: tree_decoding + evaluate_posterior
    - draft path:  update_inference_inputs (includes draft refresh)
    """
    stats = {
        "tree_decoding_s": 0.0,
        "evaluate_posterior_s": 0.0,
        "update_inference_inputs_s": 0.0,
        "tree_calls": 0,
        "eval_calls": 0,
        "update_calls": 0,
    }

    orig_tree = ea_model_module.tree_decoding
    orig_eval = ea_model_module.evaluate_posterior
    orig_update = ea_model_module.update_inference_inputs

    def _wrap_timed(fn_name, fn):
        def _wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if fn_name == "tree":
                stats["tree_decoding_s"] += dt
                stats["tree_calls"] += 1
            elif fn_name == "eval":
                stats["evaluate_posterior_s"] += dt
                stats["eval_calls"] += 1
            elif fn_name == "update":
                stats["update_inference_inputs_s"] += dt
                stats["update_calls"] += 1
            return out

        return _wrapped

    ea_model_module.tree_decoding = _wrap_timed("tree", orig_tree)
    ea_model_module.evaluate_posterior = _wrap_timed("eval", orig_eval)
    ea_model_module.update_inference_inputs = _wrap_timed("update", orig_update)
    try:
        yield stats
    finally:
        ea_model_module.tree_decoding = orig_tree
        ea_model_module.evaluate_posterior = orig_eval
        ea_model_module.update_inference_inputs = orig_update


def sample_by_bucket(questions, per_bucket, seed):
    random.seed(seed)
    by_bucket = defaultdict(list)
    for q in questions:
        b = q.get("context_bucket")
        if b:
            by_bucket[b].append(q)

    sampled = []
    for b in BUCKET_ORDER:
        pool = by_bucket.get(b, [])
        if not pool:
            continue
        n = min(per_bucket, len(pool))
        sampled.extend(random.sample(pool, n))
    return sampled


def run_one(model, tokenizer, question, max_input_tokens, max_new_tokens, max_length, all_turns):
    prompt, turns_used = build_prompt(question, all_turns)
    input_ids = tokenizer([prompt]).input_ids
    input_ids = truncate_input_ids(input_ids, max_input_tokens)
    input_ids_cuda = torch.as_tensor(input_ids).cuda()

    with instrument_decode_timers() as tstats:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _output_ids, new_tokens, decode_steps, prefill_s = model.eagenerate(
            input_ids_cuda,
            temperature=0.0,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            log=True,
        )
        torch.cuda.synchronize()
        total_s = time.perf_counter() - t0

    decode_s = max(total_s - float(prefill_s), 1e-9)
    target_s = tstats["tree_decoding_s"] + tstats["evaluate_posterior_s"]
    draft_s = tstats["update_inference_inputs_s"]
    accounted_s = target_s + draft_s
    other_decode_s = max(decode_s - accounted_s, 0.0)

    return {
        "context_bucket": question.get("context_bucket"),
        "context_tokens": question.get("context_tokens"),
        "source_dataset": question.get("source_dataset"),
        "question_id": question.get("question_id"),
        "sample_id": question.get("sample_id"),
        "turns_used": turns_used,
        "prompt_tokens": len(input_ids[0]),
        "new_tokens": int(new_tokens),
        "decode_steps": int(decode_steps),
        "prefill_s": float(prefill_s),
        "total_s": float(total_s),
        "decode_s": float(decode_s),
        "target_s": float(target_s),
        "draft_s": float(draft_s),
        "other_decode_s": float(other_decode_s),
        "target_vs_decode_ratio": float(target_s / decode_s),
        "draft_vs_decode_ratio": float(draft_s / decode_s),
        "target_vs_target_draft_ratio": float(target_s / max(target_s + draft_s, 1e-9)),
        "draft_vs_target_draft_ratio": float(draft_s / max(target_s + draft_s, 1e-9)),
        "tree_calls": int(tstats["tree_calls"]),
        "eval_calls": int(tstats["eval_calls"]),
        "update_calls": int(tstats["update_calls"]),
    }


def aggregate(records):
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[r["context_bucket"]].append(r)

    out = []
    for b in BUCKET_ORDER:
        items = by_bucket.get(b, [])
        if not items:
            continue
        out.append(
            {
                "context_bucket": b,
                "samples": len(items),
                "mean_context_tokens": mean([x["context_tokens"] for x in items]),
                "mean_decode_s": mean([x["decode_s"] for x in items]),
                "mean_target_s": mean([x["target_s"] for x in items]),
                "mean_draft_s": mean([x["draft_s"] for x in items]),
                "mean_other_decode_s": mean([x["other_decode_s"] for x in items]),
                "mean_target_vs_decode_ratio": mean([x["target_vs_decode_ratio"] for x in items]),
                "mean_draft_vs_decode_ratio": mean([x["draft_vs_decode_ratio"] for x in items]),
                "mean_target_vs_target_draft_ratio": mean([x["target_vs_target_draft_ratio"] for x in items]),
                "mean_draft_vs_target_draft_ratio": mean([x["draft_vs_target_draft_ratio"] for x in items]),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/mnt/hdd/yxy/HF/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/mnt/hdd/yxy/HF/hub/models--AngelSlim--Qwen3-1.7B_eagle3/snapshots/94441b48acc5804677ae12259617c83323b543a9",
    )
    parser.add_argument("--question-file", type=str, default="outputs/context_datapoints_0_8k_4datasets.jsonl")
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--all-turns", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=9000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=19000)
    parser.add_argument("--enable-flash", action="store_true")
    parser.add_argument("--enable-triton", action="store_true")
    parser.add_argument("--disable-qwen3-triton", action="store_true")
    parser.add_argument("--out-jsonl", type=str, default="outputs/eagle_draft_target_ratio_per_sample.jsonl")
    parser.add_argument("--out-summary", type=str, default="outputs/eagle_draft_target_ratio_summary.json")
    args = parser.parse_args()

    questions = load_questions(args.question_file, args.question_begin, args.question_end)
    picked = sample_by_bucket(questions, args.per_bucket, args.seed)

    availability = {
        "llama_flash": bool(getattr(llama_kv, "_HAS_FLASH_ATTN", False)),
        "llama_triton": bool(getattr(llama_kv, "_HAS_TRITON_TREE_ATTN", False)),
        "qwen3_flash": bool(getattr(qwen3_kv, "_HAS_FLASH_ATTN", False)),
        "qwen3_triton": bool(getattr(qwen3_kv, "_HAS_TRITON_TREE_ATTN", False)),
        "draft_flash": bool(getattr(draft_cnets, "_HAS_FLASH_ATTN_DRAFT", False)),
    }
    if not args.enable_flash:
        availability["llama_flash"] = False
        availability["qwen3_flash"] = False
        availability["draft_flash"] = False
    if not args.enable_triton:
        availability["llama_triton"] = False
        availability["qwen3_triton"] = False
    if args.disable_qwen3_triton:
        availability["qwen3_triton"] = False

    print("Loading model...")
    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=60,
        depth=7,
        top_k=10,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=True,
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    print("Flash/Triton effective:", availability)
    set_flash_mode(True, availability)

    # Warmup with first picked sample.
    if picked:
        q0 = picked[0]
        for i in range(args.warmup_runs):
            prompt, _ = build_prompt(q0, args.all_turns)
            input_ids = tokenizer([prompt]).input_ids
            input_ids = truncate_input_ids(input_ids, args.max_input_tokens)
            _ = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=0.0,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                log=True,
            )
            torch.cuda.synchronize()
            print(f"warmup {i + 1}/{args.warmup_runs} done")

    records = []
    for i, q in enumerate(picked):
        rec = run_one(
            model=model,
            tokenizer=tokenizer,
            question=q,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            all_turns=args.all_turns,
        )
        records.append(rec)
        print(
            f"[{i + 1}/{len(picked)}] bucket={rec['context_bucket']} ctx={rec['context_tokens']} "
            f"target/decode={rec['target_vs_decode_ratio']:.3f} "
            f"draft/decode={rec['draft_vs_decode_ratio']:.3f}"
        )

    summary_by_bucket = aggregate(records)
    summary = {
        "config": {
            "question_file": args.question_file,
            "question_begin": args.question_begin,
            "question_end": args.question_end,
            "per_bucket": args.per_bucket,
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "all_turns": bool(args.all_turns),
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "enable_flash": bool(args.enable_flash),
            "enable_triton": bool(args.enable_triton),
            "disable_qwen3_triton": bool(args.disable_qwen3_triton),
        },
        "overall": {
            "samples": len(records),
            "mean_target_vs_decode_ratio": mean([x["target_vs_decode_ratio"] for x in records]),
            "mean_draft_vs_decode_ratio": mean([x["draft_vs_decode_ratio"] for x in records]),
            "mean_target_vs_target_draft_ratio": mean([x["target_vs_target_draft_ratio"] for x in records]),
            "mean_draft_vs_target_draft_ratio": mean([x["draft_vs_target_draft_ratio"] for x in records]),
        },
        "by_context_bucket": summary_by_bucket,
    }

    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Draft vs Target Time Ratio Summary ===")
    print(f"Samples: {summary['overall']['samples']}")
    print(f"Mean target/decode: {summary['overall']['mean_target_vs_decode_ratio']:.3f}")
    print(f"Mean draft/decode:  {summary['overall']['mean_draft_vs_decode_ratio']:.3f}")
    print(f"Mean target/(target+draft): {summary['overall']['mean_target_vs_target_draft_ratio']:.3f}")
    print(f"Mean draft/(target+draft):  {summary['overall']['mean_draft_vs_target_draft_ratio']:.3f}")
    print(f"Per-sample saved: {out_jsonl}")
    print(f"Summary saved: {out_summary}")


if __name__ == "__main__":
    main()
