#!/usr/bin/env python3

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer


def truncate_input_ids(input_ids, max_input_tokens):
    if not max_input_tokens:
        return input_ids
    if len(input_ids[0]) <= max_input_tokens:
        return input_ids
    half = max_input_tokens // 2
    input_ids[0] = input_ids[0][:half] + input_ids[0][-half:]
    return input_ids


def build_turns(question, use_all_turns):
    turns = question.get("turns") or []
    return turns if use_all_turns else turns[:1]


def warmup(model, tokenizer, questions, warmup, max_input_tokens, max_new_tokens, use_all_turns, temperature):
    if not questions:
        return
    q0 = questions[0]
    turns0 = build_turns(q0, use_all_turns)
    for _ in range(warmup):
        conv = get_conversation_template("qwen3")
        for turn in turns0:
            conv.append_message(conv.roles[0], turn)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids
            input_ids = truncate_input_ids(input_ids, max_input_tokens)
            ids = torch.as_tensor(input_ids, device=model.device)
            _run_one_step(model, ids, max_new_tokens, temperature)


def _sample_next(logits, temperature):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return logits.argmax(dim=-1, keepdim=True)


def _run_one_step(model, ids, gen_len, temperature):
    """Return (prefill_time_s, decode_time_s, new_tokens, generated_ids_1d)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(ids, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    past = out.past_key_values
    logits = out.logits[:, -1, :]
    next_tok = _sample_next(logits, temperature)
    del out

    # If gen_len <= 1, we still count exactly one generated token.
    target_gen = max(int(gen_len), 1)

    generated = [int(next_tok.item())]

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(target_gen - 1):
        with torch.inference_mode():
            out = model(next_tok, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_tok = _sample_next(logits, temperature)
        generated.append(int(next_tok.item()))
        del out
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t1

    return prefill_time, decode_time, target_gen, generated


def run_one_question(model, tokenizer, question, max_input_tokens, max_new_tokens, use_all_turns, temperature):
    conv = get_conversation_template("qwen3")
    turns = build_turns(question, use_all_turns)

    total_prefill_s = 0.0
    total_decode_s = 0.0
    total_prompt_tokens = 0
    total_new_tokens = 0

    for turn in turns:
        conv.append_message(conv.roles[0], turn)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        input_ids = truncate_input_ids(input_ids, max_input_tokens)
        ids = torch.as_tensor(input_ids, device=model.device)

        prefill_s, decode_s, new_tok, generated_ids = _run_one_step(
            model,
            ids,
            max_new_tokens,
            temperature,
        )

        total_prefill_s += float(prefill_s)
        total_decode_s += float(decode_s)
        total_prompt_tokens += int(ids.shape[1])
        total_new_tokens += int(new_tok)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True, spaces_between_special_tokens=False).strip()
        conv.update_last_message(text)

    total_time_s = total_prefill_s + total_decode_s
    decode_time_s = max(total_decode_s, 1e-9)
    total_tps = total_new_tokens / max(total_time_s, 1e-9)
    prefill_tps = total_prompt_tokens / max(total_prefill_s, 1e-9)
    decode_tps = total_new_tokens / decode_time_s

    return {
        "total_time_s": total_time_s,
        "prefill_time_s": total_prefill_s,
        "decode_time_s": total_decode_s,
        "prompt_tokens": total_prompt_tokens,
        "new_tokens": total_new_tokens,
        "total_tps": total_tps,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "num_turns": len(turns),
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/mnt/hdd/yxy/HF/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=9000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--all-turns", action="store_true")
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--out-summary", type=str, required=True)
    args = parser.parse_args()

    questions = load_questions(args.question_file, args.question_begin, args.question_end)

    print("Loading AR model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    ).eval()

    print(f"Loaded {len(questions)} questions from {args.question_file}")
    print("Warming up...")
    warmup(
        model,
        tokenizer,
        questions,
        args.warmup,
        args.max_input_tokens,
        args.max_new_tokens,
        args.all_turns,
        args.temperature,
    )

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)

    records = []
    per_question = defaultdict(list)

    for question_idx, question in enumerate(questions):
        for repeat_idx in range(args.repeats):
            metrics = run_one_question(
                model,
                tokenizer,
                question,
                args.max_input_tokens,
                args.max_new_tokens,
                args.all_turns,
                args.temperature,
            )
            record = {
                "question_idx": question_idx,
                "repeat_idx": repeat_idx,
                "question_id": question.get("question_id"),
                "sample_id": question.get("sample_id"),
                "source_dataset": question.get("source_dataset"),
                "source_category": question.get("source_category"),
                "context_bucket": question.get("context_bucket"),
                "context_tokens": question.get("context_tokens"),
                **metrics,
            }
            records.append(record)
            per_question[question_idx].append(record)

            print(
                f"[{question_idx + 1}/{len(questions)}][run {repeat_idx + 1}/{args.repeats}] "
                f"ctx={record.get('context_tokens')} turns={metrics['num_turns']} "
                f"total={metrics['total_tps']:.2f} prefill={metrics['prefill_tps']:.2f} decode={metrics['decode_tps']:.2f} tok/s"
            )

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    per_question_summary = []
    for question_idx, items in per_question.items():
        base = items[0]
        per_question_summary.append(
            {
                "question_idx": question_idx,
                "question_id": base.get("question_id"),
                "sample_id": base.get("sample_id"),
                "source_dataset": base.get("source_dataset"),
                "source_category": base.get("source_category"),
                "context_bucket": base.get("context_bucket"),
                "context_tokens": base.get("context_tokens"),
                "repeats": len(items),
                "mean_total_tps": mean([x["total_tps"] for x in items]),
                "mean_prefill_tps": mean([x["prefill_tps"] for x in items]),
                "mean_decode_tps": mean([x["decode_tps"] for x in items]),
                "mean_total_time_s": mean([x["total_time_s"] for x in items]),
                "mean_prefill_time_s": mean([x["prefill_time_s"] for x in items]),
                "mean_decode_time_s": mean([x["decode_time_s"] for x in items]),
            }
        )

    summary = {
        "config": {
            "base_model_path": args.base_model_path,
            "question_file": args.question_file,
            "question_begin": args.question_begin,
            "question_end": args.question_end,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "all_turns": bool(args.all_turns),
        },
        "overall": {
            "questions": len(questions),
            "runs": len(records),
            "mean_total_tps": mean([x["total_tps"] for x in records]),
            "mean_prefill_tps": mean([x["prefill_tps"] for x in records]),
            "mean_decode_tps": mean([x["decode_tps"] for x in records]),
            "mean_total_time_s": mean([x["total_time_s"] for x in records]),
            "mean_prefill_time_s": mean([x["prefill_time_s"] for x in records]),
            "mean_decode_time_s": mean([x["decode_time_s"] for x in records]),
        },
        "per_question": per_question_summary,
    }

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== AR Repeated Benchmark Summary ===")
    print(f"Questions: {summary['overall']['questions']}")
    print(f"Runs: {summary['overall']['runs']}")
    print(f"Mean total tok/s:   {summary['overall']['mean_total_tps']:.2f}")
    print(f"Mean prefill tok/s: {summary['overall']['mean_prefill_tps']:.2f}")
    print(f"Mean decode tok/s:  {summary['overall']['mean_decode_tps']:.2f}")
    print(f"JSONL saved to: {args.out_jsonl}")
    print(f"Summary saved to: {args.out_summary}")


if __name__ == "__main__":
    main()
