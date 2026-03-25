#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

import torch
from fastchat.llm_judge.common import load_questions

from eagle.model.ea_model import EaModel
from eagle.model.kv_cache import initialize_past_key_values
from eagle.model.utils import (
    prepare_logits_processor,
    reset_tree_mode,
    initialize_tree,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
)
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


def select_question_4k8k(questions, sample_index):
    pool = [
        q
        for q in questions
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
    parser.add_argument("--question-file", type=str, default="outputs/context_datapoints_0_8k_4datasets.jsonl")
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=9000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=19000)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--enable-flash", action="store_true")
    parser.add_argument("--enable-triton", action="store_true")
    parser.add_argument("--disable-qwen3-triton", action="store_true")
    parser.add_argument("--out-prefix", type=str, default="outputs/eagle_decode_only_profile_4k8k")
    args = parser.parse_args()

    questions = load_questions(args.question_file, args.question_begin, args.question_end)
    question, pool_size = select_question_4k8k(questions, args.sample_index)
    prompt = (question.get("turns") or [""])[0]

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
    set_flash_mode(True, availability)
    print("Flash/Triton effective:", availability)

    input_ids = tokenizer([prompt]).input_ids
    input_ids = truncate_input_ids(input_ids, args.max_input_tokens)
    input_ids = torch.as_tensor(input_ids).cuda()
    print(
        f"Using sample_id={question.get('sample_id')} dataset={question.get('source_dataset')} "
        f"ctx={question.get('context_tokens')} bucket={question.get('context_bucket')} "
        f"pool_4k8k={pool_size}"
    )

    logits_processor = None
    model.ea_layer.reset_kv()
    model.draft_input_ids = input_ids.clone()
    reset_tree_mode(model)
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.base_model, max_length=args.max_length)

    # Prefill (initialize_tree) is intentionally outside profiler.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
        input_ids,
        model,
        past_key_values,
        logits_processor,
    )
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    print(f"Prefill done (excluded from trace): {prefill_s:.4f}s")

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    padding = (torch.zeros(1, 1, dtype=torch.long, device=input_ids.device) - 1)
    decode_steps = max(1, int(args.decode_steps))
    new_token = 0

    torch.cuda.synchronize()
    d0 = time.perf_counter()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        with torch.inference_mode():
            for step in range(decode_steps):
                model.base_model.model.tree_mask = tree_mask
                draft_tokens = draft_tokens.to(input_ids.device)

                with torch.profiler.record_function("decode_only_step"):
                    logits_step, hidden_state_new, _ = tree_decoding(
                        model,
                        draft_tokens,
                        past_key_values,
                        tree_position_ids,
                        input_ids,
                        retrieve_indices,
                    )

                    draft_tokens_pad = torch.cat((draft_tokens, padding), dim=1)
                    candidates = draft_tokens_pad[0, retrieve_indices]
                    best_candidate, accept_length, sample_p = evaluate_posterior(
                        logits_step,
                        candidates,
                        logits_processor,
                    )

                    input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                        input_ids=input_ids,
                        candidates=candidates,
                        best_candidate=best_candidate,
                        accept_length=accept_length,
                        retrieve_indices=retrieve_indices,
                        logits_processor=logits_processor,
                        new_token=new_token,
                        past_key_values_data_list=past_key_values_data,
                        current_length_data=current_length_data,
                        model=model,
                        hidden_state_new=hidden_state_new,
                        sample_p=sample_p,
                    )
                prof.step()

                if new_token >= args.max_new_tokens:
                    break

    torch.cuda.synchronize()
    decode_profiled_s = time.perf_counter() - d0

    trace_path = f"{args.out_prefix}.trace.json"
    ops_path = f"{args.out_prefix}.ops.txt"
    summary_path = f"{args.out_prefix}.summary.json"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    prof.export_chrome_trace(trace_path)
    table_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
    table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=50)
    with open(ops_path, "w", encoding="utf-8") as f:
        f.write("=== Top CUDA ops ===\n")
        f.write(table_cuda)
        f.write("\n\n=== Top CPU ops ===\n")
        f.write(table_cpu)

    summary = {
        "config": {
            "question_file": args.question_file,
            "sample_index": args.sample_index,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "decode_steps_requested": decode_steps,
            "enable_flash": bool(args.enable_flash),
            "enable_triton": bool(args.enable_triton),
            "disable_qwen3_triton": bool(args.disable_qwen3_triton),
        },
        "sample": {
            "sample_id": question.get("sample_id"),
            "question_id": question.get("question_id"),
            "source_dataset": question.get("source_dataset"),
            "context_bucket": question.get("context_bucket"),
            "context_tokens": question.get("context_tokens"),
            "prompt_tokens_after_trunc": int(input_ids.shape[1]),
        },
        "timing": {
            "prefill_outside_profiler_s": prefill_s,
            "decode_profiled_total_s": decode_profiled_s,
            "new_tokens_after_profiled_steps": int(new_token),
        },
        "artifacts": {
            "trace_json": trace_path,
            "ops_table": ops_path,
        },
        "note": "initialize_tree is excluded from profiler by design in this script.",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Decode-only Profile Summary ===")
    print(f"Trace saved: {trace_path}")
    print(f"Ops table saved: {ops_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
