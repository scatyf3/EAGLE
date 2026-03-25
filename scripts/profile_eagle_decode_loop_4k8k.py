import argparse
import json
import os
import time
from statistics import mean

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


def select_question_4k8k(questions, sample_index):
    pool = [
        q for q in questions
        if (q.get("context_tokens") is not None)
        and (4096 <= int(q.get("context_tokens")) < 8192)
    ]
    if not pool:
        raise ValueError("No samples in 4k-8k context range were found.")
    if sample_index < 0 or sample_index >= len(pool):
        raise ValueError(f"sample-index out of range: {sample_index}, available [0, {len(pool)-1}]")
    return pool[sample_index], len(pool)


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
    parser.add_argument(
        "--question-file",
        type=str,
        default="outputs/context_datapoints_0_8k_4datasets.jsonl",
    )
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--all-turns", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=9000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=19000)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--profile-runs", type=int, default=5)
    parser.add_argument(
        "--profile-capture-runs",
        type=int,
        default=1,
        help="How many decode iterations are recorded into torch profiler trace.",
    )
    parser.add_argument("--enable-flash", action="store_true")
    parser.add_argument("--enable-triton", action="store_true")
    parser.add_argument("--disable-qwen3-triton", action="store_true")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="outputs/eagle_qwen3_decode_profile_4k8k",
    )
    args = parser.parse_args()

    questions = load_questions(args.question_file, args.question_begin, args.question_end)
    question, pool_size = select_question_4k8k(questions, args.sample_index)
    prompt, used_turns = build_prompt(question, args.all_turns)

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

    input_ids = tokenizer([prompt]).input_ids
    input_ids = truncate_input_ids(input_ids, args.max_input_tokens)
    input_ids_cuda = torch.as_tensor(input_ids).cuda()

    print(
        f"Using sample_id={question.get('sample_id')} dataset={question.get('source_dataset')} "
        f"ctx={question.get('context_tokens')} bucket={question.get('context_bucket')} turns={used_turns} "
        f"pool_4k8k={pool_size}"
    )
    print(f"Prompt tokens after truncation: {len(input_ids[0])}")

    print(f"Warmup runs: {args.warmup_runs}")
    with torch.inference_mode():
        for i in range(args.warmup_runs):
            _ = model.eagenerate(
                input_ids_cuda,
                temperature=0.0,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                log=True,
            )
            torch.cuda.synchronize()
            print(f"  warmup {i + 1}/{args.warmup_runs} done")

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    total_time_list = []
    prefill_time_list = []
    decode_time_list = []
    new_tokens_list = []
    draft_steps_list = []
    decode_tps_list = []

    print(f"Timing runs (for averages): {args.profile_runs}")
    with torch.inference_mode():
        for i in range(args.profile_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _output_ids, new_token, idx, prefill_s = model.eagenerate(
                input_ids_cuda,
                temperature=0.0,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                log=True,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            decode_s = max(float(elapsed) - float(prefill_s), 1e-9)
            decode_tps = float(new_token) / decode_s

            total_time_list.append(float(elapsed))
            prefill_time_list.append(float(prefill_s))
            decode_time_list.append(float(decode_s))
            new_tokens_list.append(int(new_token))
            draft_steps_list.append(int(idx))
            decode_tps_list.append(float(decode_tps))

            print(
                f"  timing {i + 1}/{args.profile_runs}: "
                f"new={int(new_token)} draft_steps={int(idx)} "
                f"prefill={float(prefill_s):.4f}s decode={decode_s:.4f}s decode_tps={decode_tps:.2f}"
            )

    capture_runs = max(1, min(args.profile_capture_runs, args.profile_runs))
    print(f"Profiler capture runs: {capture_runs}")
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        with torch.inference_mode():
            for i in range(capture_runs):
                with torch.profiler.record_function("eagle_decode_loop_iter"):
                    _ = model.eagenerate(
                        input_ids_cuda,
                        temperature=0.0,
                        max_length=args.max_length,
                        max_new_tokens=args.max_new_tokens,
                        log=True,
                    )
                prof.step()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    trace_path = f"{args.out_prefix}.trace.json"
    table_path = f"{args.out_prefix}.ops.txt"
    summary_path = f"{args.out_prefix}.summary.json"

    prof.export_chrome_trace(trace_path)
    table_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
    table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=50)
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("=== Top CUDA ops ===\n")
        f.write(table_cuda)
        f.write("\n\n=== Top CPU ops ===\n")
        f.write(table_cpu)

    summary = {
        "config": {
            "question_file": args.question_file,
            "question_begin": args.question_begin,
            "question_end": args.question_end,
            "sample_index": args.sample_index,
            "all_turns": bool(args.all_turns),
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "warmup_runs": args.warmup_runs,
            "profile_runs": args.profile_runs,
            "profile_capture_runs": capture_runs,
            "enable_flash": bool(args.enable_flash),
            "enable_triton": bool(args.enable_triton),
            "disable_qwen3_triton": bool(args.disable_qwen3_triton),
        },
        "sample": {
            "sample_id": question.get("sample_id"),
            "question_id": question.get("question_id"),
            "source_dataset": question.get("source_dataset"),
            "source_category": question.get("source_category"),
            "context_bucket": question.get("context_bucket"),
            "context_tokens": question.get("context_tokens"),
            "turns_used": used_turns,
            "prompt_tokens_after_trunc": len(input_ids[0]),
            "pool_4k8k_size": pool_size,
        },
        "averages": {
            "mean_total_time_s": mean(total_time_list),
            "mean_prefill_time_s": mean(prefill_time_list),
            "mean_decode_time_s": mean(decode_time_list),
            "mean_new_tokens": mean(new_tokens_list),
            "mean_draft_steps": mean(draft_steps_list),
            "mean_decode_tps": mean(decode_tps_list),
            "mean_acceptance_rate": (sum(new_tokens_list) / max(sum(draft_steps_list), 1)),
        },
        "per_run": [
            {
                "run_idx": i,
                "total_time_s": total_time_list[i],
                "prefill_time_s": prefill_time_list[i],
                "decode_time_s": decode_time_list[i],
                "new_tokens": new_tokens_list[i],
                "draft_steps": draft_steps_list[i],
                "decode_tps": decode_tps_list[i],
            }
            for i in range(len(total_time_list))
        ],
        "artifacts": {
            "trace_json": trace_path,
            "ops_table": table_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Profile Summary ===")
    print(f"Mean decode tps: {summary['averages']['mean_decode_tps']:.2f}")
    print(f"Mean decode time: {summary['averages']['mean_decode_time_s']:.4f}s")
    print(f"Mean acceptance: {summary['averages']['mean_acceptance_rate']:.3f}")
    print(f"Trace saved: {trace_path}")
    print(f"Ops table saved: {table_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
