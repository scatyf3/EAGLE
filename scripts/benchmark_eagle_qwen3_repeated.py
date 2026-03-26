import argparse
import json
import os
import time
from collections import defaultdict

import torch
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template

from eagle.model.ea_model import EaModel
import eagle.model.cnets as draft_cnets
import eagle.model.modeling_llama_kv as llama_kv
import eagle.model.modeling_qwen3_kv as qwen3_kv


def truncate_input_ids(input_ids, max_input_tokens):
    if not max_input_tokens:
        return input_ids
    if len(input_ids[0]) <= max_input_tokens:
        return input_ids
    half = max_input_tokens // 2
    input_ids[0] = input_ids[0][:half] + input_ids[0][-half:]
    return input_ids


def decode_output(tokenizer, conv, output_ids):
    output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
    if conv.stop_str and output.find(conv.stop_str) > 0:
        output = output[: output.find(conv.stop_str)]
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                output = output.replace(special_tok, "")
        else:
            output = output.replace(special_token, "")
    return output.strip()


def build_turns(question, use_all_turns):
    turns = question.get("turns") or []
    return turns if use_all_turns else turns[:1]


def set_flash_mode(enabled, availability):
    llama_kv._HAS_FLASH_ATTN = bool(enabled and availability["llama_flash"])
    llama_kv._HAS_TRITON_TREE_ATTN = bool(enabled and availability["llama_triton"])
    qwen3_kv._HAS_FLASH_ATTN = bool(enabled and availability["qwen3_flash"])
    qwen3_kv._HAS_TRITON_TREE_ATTN = bool(enabled and availability["qwen3_triton"])
    draft_cnets._HAS_FLASH_ATTN_DRAFT = bool(enabled and availability["draft_flash"])


def warmup(model, tokenizer, questions, warmup, max_input_tokens, max_new_tokens, max_length, use_all_turns):
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
            model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=0.0,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                log=True,
            )


def run_one_question(model, tokenizer, question, max_input_tokens, max_new_tokens, max_length, use_all_turns):
    conv = get_conversation_template("qwen3")
    turns = build_turns(question, use_all_turns)

    total_time_s = 0.0
    total_prefill_s = 0.0
    total_prompt_tokens = 0
    total_new_tokens = 0
    total_draft_steps = 0

    for turn in turns:
        conv.append_message(conv.roles[0], turn)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        input_ids = truncate_input_ids(input_ids, max_input_tokens)

        prompt_tokens = len(input_ids[0])

        torch.cuda.synchronize()
        start = time.perf_counter()
        output_ids, new_token, idx, prefill_s = model.eagenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=0.0,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            log=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated = output_ids[0][len(input_ids[0]):]
        text = decode_output(tokenizer, conv, generated)
        conv.update_last_message(text)

        total_time_s += float(elapsed)
        total_prefill_s += float(prefill_s)
        total_prompt_tokens += int(prompt_tokens)
        total_new_tokens += int(new_token)
        total_draft_steps += int(idx)

    decode_time_s = max(total_time_s - total_prefill_s, 1e-9)
    total_tps = total_new_tokens / max(total_time_s, 1e-9)
    prefill_tps = total_prompt_tokens / max(total_prefill_s, 1e-9)
    decode_tps = total_new_tokens / decode_time_s
    acceptance_rate = total_new_tokens / max(total_draft_steps, 1)

    return {
        "total_time_s": total_time_s,
        "prefill_time_s": total_prefill_s,
        "decode_time_s": decode_time_s,
        "prompt_tokens": total_prompt_tokens,
        "new_tokens": total_new_tokens,
        "total_tps": total_tps,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "acceptance_rate": acceptance_rate,
        "draft_steps": total_draft_steps,
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
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/mnt/hdd/yxy/HF/hub/models--AngelSlim--Qwen3-1.7B_eagle3/snapshots/94441b48acc5804677ae12259617c83323b543a9",
    )
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=9000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=19000)
    parser.add_argument("--all-turns", action="store_true")
    parser.add_argument("--enable-flash", action="store_true")
    parser.add_argument("--enable-triton", action="store_true")
    parser.add_argument("--disable-qwen3-triton", action="store_true")
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--out-summary", type=str, required=True)
    args = parser.parse_args()

    questions = load_questions(args.question_file, args.question_begin, args.question_end)

    availability = {
        "llama_flash": bool(getattr(llama_kv, "_HAS_FLASH_ATTN", False)),
        "llama_triton": bool(getattr(llama_kv, "_HAS_TRITON_TREE_ATTN", False)),
        "qwen3_flash": bool(getattr(qwen3_kv, "_HAS_FLASH_ATTN", False)),
        "qwen3_triton": bool(getattr(qwen3_kv, "_HAS_TRITON_TREE_ATTN", False)),
        "draft_flash": bool(getattr(draft_cnets, "_HAS_FLASH_ATTN_DRAFT", False)),
    }
    # Default policy: disable flash/triton unless explicitly enabled.
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

    print(f"Loaded {len(questions)} questions from {args.question_file}")
    print("Warming up...")
    warmup(
        model,
        tokenizer,
        questions,
        args.warmup,
        args.max_input_tokens,
        args.max_new_tokens,
        args.max_length,
        args.all_turns,
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
                args.max_length,
                args.all_turns,
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
                "mean_acceptance_rate": mean([x["acceptance_rate"] for x in items]),
                "mean_total_time_s": mean([x["total_time_s"] for x in items]),
                "mean_prefill_time_s": mean([x["prefill_time_s"] for x in items]),
                "mean_decode_time_s": mean([x["decode_time_s"] for x in items]),
            }
        )

    summary = {
        "config": {
            "base_model_path": args.base_model_path,
            "ea_model_path": args.ea_model_path,
            "question_file": args.question_file,
            "question_begin": args.question_begin,
            "question_end": args.question_end,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "all_turns": bool(args.all_turns),
        },
        "overall": {
            "questions": len(questions),
            "runs": len(records),
            "mean_total_tps": mean([x["total_tps"] for x in records]),
            "mean_prefill_tps": mean([x["prefill_tps"] for x in records]),
            "mean_decode_tps": mean([x["decode_tps"] for x in records]),
            "mean_acceptance_rate": mean([x["acceptance_rate"] for x in records]),
            "mean_total_time_s": mean([x["total_time_s"] for x in records]),
            "mean_prefill_time_s": mean([x["prefill_time_s"] for x in records]),
            "mean_decode_time_s": mean([x["decode_time_s"] for x in records]),
        },
        "per_question": per_question_summary,
    }

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Repeated Benchmark Summary ===")
    print(f"Questions: {summary['overall']['questions']}")
    print(f"Runs: {summary['overall']['runs']}")
    print(f"Mean total tok/s:   {summary['overall']['mean_total_tps']:.2f}")
    print(f"Mean prefill tok/s: {summary['overall']['mean_prefill_tps']:.2f}")
    print(f"Mean decode tok/s:  {summary['overall']['mean_decode_tps']:.2f}")
    print(f"Mean acceptance:    {summary['overall']['mean_acceptance_rate']:.3f}")
    print(f"JSONL saved to: {args.out_jsonl}")
    print(f"Summary saved to: {args.out_summary}")


if __name__ == "__main__":
    main()