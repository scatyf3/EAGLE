import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template

from eagle.model.ea_model import EaModel
import eagle.model.cnets as draft_cnets
import eagle.model.modeling_llama_kv as llama_kv
import eagle.model.modeling_qwen3_kv as qwen3_kv


@dataclass
class ModeResult:
    name: str
    samples: int
    total_time_s: float
    total_prompt_tokens: int
    total_new_tokens: int
    avg_time_s: float
    tok_per_s: float
    prefill_s: float


def truncate_input_ids(input_ids: List[List[int]], max_input_tokens: int) -> List[List[int]]:
    if not max_input_tokens:
        return input_ids
    if len(input_ids[0]) <= max_input_tokens:
        return input_ids
    half = max_input_tokens // 2
    input_ids[0] = input_ids[0][:half] + input_ids[0][-half:]
    return input_ids


def set_flash_mode(enabled: bool, availability: Dict[str, bool]) -> None:
    llama_kv._HAS_FLASH_ATTN = bool(enabled and availability["llama_flash"])
    llama_kv._HAS_TRITON_TREE_ATTN = bool(enabled and availability["llama_triton"])
    qwen3_kv._HAS_FLASH_ATTN = bool(enabled and availability["qwen3_flash"])
    qwen3_kv._HAS_TRITON_TREE_ATTN = bool(enabled and availability["qwen3_triton"])
    draft_cnets._HAS_FLASH_ATTN_DRAFT = bool(enabled and availability["draft_flash"])


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


def run_mode(
    model,
    tokenizer,
    questions,
    mode_name,
    warmup,
    max_input_tokens,
    max_new_tokens,
    max_length,
    use_all_turns,
):
    # Warmup on first question to stabilize kernels and cache behavior.
    if questions:
        q0 = questions[0]
        turns0 = q0["turns"] if use_all_turns else q0["turns"][:1]
        for _ in range(warmup):
            conv = get_conversation_template("qwen3")
            for qs in turns0:
                conv.append_message(conv.roles[0], qs)
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

    total_time_s = 0.0
    total_prefill_s = 0.0
    total_prompt_tokens = 0
    total_new_tokens = 0
    samples = 0

    for q in questions:
        conv = get_conversation_template("qwen3")
        turns = q["turns"] if use_all_turns else q["turns"][:1]
        for qs in turns:
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids
            input_ids = truncate_input_ids(input_ids, max_input_tokens)

            torch.cuda.synchronize()
            start = time.perf_counter()
            output_ids, new_token, _idx, prefill_s = model.eagenerate(
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

            samples += 1
            total_time_s += float(elapsed)
            total_prefill_s += float(prefill_s)
            total_prompt_tokens += int(len(input_ids[0]))
            total_new_tokens += int(new_token)

    avg_time_s = total_time_s / max(samples, 1)
    tok_per_s = total_new_tokens / max(total_time_s, 1e-8)
    return ModeResult(
        name=mode_name,
        samples=samples,
        total_time_s=total_time_s,
        total_prompt_tokens=total_prompt_tokens,
        total_new_tokens=total_new_tokens,
        avg_time_s=avg_time_s,
        tok_per_s=tok_per_s,
        prefill_s=total_prefill_s,
    )


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
    parser.add_argument("--bench-name", type=str, default="hotpotqa_e_prompt")
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=9000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=19000)
    parser.add_argument("--all-turns", action="store_true")
    parser.add_argument("--disable-qwen3-triton", action="store_true")
    parser.add_argument("--out-json", type=str, default="outputs/eagle_flash_qwen3_benchmark.json")
    args = parser.parse_args()

    question_file = os.path.join("eagle", "data", args.bench_name, "question.jsonl")
    questions = load_questions(question_file, args.question_begin, args.question_end)

    availability = {
        "llama_flash": bool(getattr(llama_kv, "_HAS_FLASH_ATTN", False)),
        "llama_triton": bool(getattr(llama_kv, "_HAS_TRITON_TREE_ATTN", False)),
        "qwen3_flash": bool(getattr(qwen3_kv, "_HAS_FLASH_ATTN", False)),
        "qwen3_triton": bool(getattr(qwen3_kv, "_HAS_TRITON_TREE_ATTN", False)),
        "draft_flash": bool(getattr(draft_cnets, "_HAS_FLASH_ATTN_DRAFT", False)),
    }
    if args.disable_qwen3_triton:
        availability["qwen3_triton"] = False

    print("Flash/Triton availability:", availability)
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

    set_flash_mode(False, availability)
    baseline = run_mode(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        mode_name="baseline_no_flash",
        warmup=args.warmup,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        use_all_turns=args.all_turns,
    )

    set_flash_mode(True, availability)
    optimized = run_mode(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        mode_name="optimized_flash_triton",
        warmup=args.warmup,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        use_all_turns=args.all_turns,
    )

    speedup = baseline.total_time_s / max(optimized.total_time_s, 1e-8)
    delta_pct = (baseline.total_time_s - optimized.total_time_s) / max(baseline.total_time_s, 1e-8) * 100.0

    report = {
        "config": {
            "base_model_path": args.base_model_path,
            "ea_model_path": args.ea_model_path,
            "bench_name": args.bench_name,
            "question_begin": args.question_begin,
            "question_end": args.question_end,
            "warmup": args.warmup,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "all_turns": bool(args.all_turns),
        },
        "availability": availability,
        "baseline": baseline.__dict__,
        "optimized": optimized.__dict__,
        "comparison": {
            "speedup_x": speedup,
            "time_reduction_percent": delta_pct,
        },
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    print("\n=== Benchmark Summary ===")
    print(f"Samples: {baseline.samples}")
    print(f"Baseline total:  {baseline.total_time_s:.3f}s, tok/s={baseline.tok_per_s:.3f}, prefill_s={baseline.prefill_s:.3f}")
    print(f"Optimized total: {optimized.total_time_s:.3f}s, tok/s={optimized.tok_per_s:.3f}, prefill_s={optimized.prefill_s:.3f}")
    print(f"Speedup: {speedup:.4f}x, Time reduction: {delta_pct:.2f}%")
    print(f"Report saved to: {args.out_json}")


if __name__ == "__main__":
    main()
