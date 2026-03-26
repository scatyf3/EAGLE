"""
Profile exactly one EAGLE speculative step (target verify + draft refresh)
under a fair context setting (ctx=3000).

This script keeps trace size small by profiling only one decode iteration,
not the whole generation loop.
"""

import json
import time
from pathlib import Path
import sys

import torch

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def load_prompt() -> str:
    jsonl_path = Path(__file__).parent.parent / "outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl"
    with open(jsonl_path, "r") as f:
        sample = json.loads(next(f))
    return sample.get("prompt", sample.get("context", ""))


def main():
    # Fair-setting knobs
    ctx_len = 3000
    max_length = 8192
    temperature = 0.6
    top_p = 0.9
    top_k = -1

    print("[*] Loading EAGLE model...")
    model = EaModel.from_pretrained(
        base_model_path="NousResearch/Yarn-Llama-2-7b-128k",
        ea_model_path="yuhuili/EAGLE-llama2-chat-7B",
        total_token=-1,
        depth=6,
        top_k=10,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=False,
    ).eval()

    tokenizer = model.get_tokenizer()
    prompt = load_prompt()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[:, :ctx_len].to("cuda:0")
    print(f"[*] Input shape: {input_ids.shape} (ctx={ctx_len})")

    logits_processor = prepare_logits_processor(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # Reset decode state and allocate KV cache with large max_length.
    model.ea_layer.reset_kv()
    model.draft_input_ids = input_ids.clone()
    reset_tree_mode(model)
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.base_model, max_length=max_length)

    # ---------- Prefill (not profiled) ----------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids,
            model,
            past_key_values,
            logits_processor,
        )
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0
    print(f"[*] Prefill done: {prefill_time * 1000:.2f} ms")

    # ---------- Profile exactly one speculative step ----------
    # This single step contains:
    # 1) target tree_decoding
    # 2) posterior evaluation
    # 3) draft refresh via update_inference_inputs (topK_genrate)
    padding = (torch.zeros(1, 1, dtype=torch.long, device=input_ids.device) - 1)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        with torch.inference_mode():
            model.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            step_logits, hidden_state_new, _ = tree_decoding(
                model,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            draft_tokens_with_pad = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens_with_pad[0, retrieve_indices]

            best_candidate, accept_length, sample_p = evaluate_posterior(
                step_logits,
                candidates,
                logits_processor,
            )

            _ = update_inference_inputs(
                input_ids=input_ids,
                candidates=candidates,
                best_candidate=best_candidate,
                accept_length=accept_length,
                retrieve_indices=retrieve_indices,
                logits_processor=logits_processor,
                new_token=0,
                past_key_values_data_list=past_key_values_data,
                current_length_data=current_length_data,
                model=model,
                hidden_state_new=hidden_state_new,
                sample_p=sample_p,
            )
    torch.cuda.synchronize()
    spec_step_time = time.perf_counter() - t1

    out_trace = "outputs/eagle_one_spec_step_ctx3000_no_flash_triton.json"
    prof.export_chrome_trace(out_trace)

    # Quick operator summary
    print("\n[*] One-step speculative profile complete")
    print(f"    accept_length = {int(accept_length)}")
    print(f"    speculative step wall time = {spec_step_time * 1000:.2f} ms")
    print(f"    trace = {out_trace}")
    print("\n[*] Top CUDA ops (single step):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
