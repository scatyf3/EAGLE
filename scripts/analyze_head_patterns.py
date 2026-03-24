#!/usr/bin/env python3
"""Analyze per-head attention patterns (echo / induction) from attn_debug traces.

Supports two modes selected with --mode:

  target (default): reads accepted_token_attn_trace from gen_baseline_answer output.
    Each step: {step_idx, accepted_token_id,
                attn_to_previous_tokens: {layer_xxx: tensor[num_heads, prev_len]}}

  draft:            reads draft_accepted_attn_trace from gen_ea_answer output.
    Each decode step: {decode_step_idx, accepted_token_ids: [int,...],
                       draft_topk_trace: [{draft_step, attn_to_previous_tokens: tensor[num_heads, prev_len], kv_len},...]}
    The i-th accepted token corresponds to draft_topk_trace[i]; attention is a flat tensor
    (no layer dim, single-layer draft model). The layer key is reported as "draft_layer_000".
    Attention positions beyond cur_pos are draft-tree expansion artifacts and are ignored.

Outputs:
  - CSV with one row per (layer, head) and pattern scores
  - JSON summary with top-K heads per pattern
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch


def _safe_div(x, y):
    return 0.0 if y == 0 else float(x) / float(y)


def _iter_target_steps(trace, prompt_ids, generated_ids):
    """Yield (cur_token, cur_pos, {layer: tensor[num_heads, prev_len]}) for target mode."""
    all_ids = list(prompt_ids) + list(generated_ids)
    prompt_len = len(prompt_ids)
    for step in trace:
        step_idx = int(step["step_idx"])
        cur_pos = prompt_len + step_idx
        if cur_pos >= len(all_ids):
            continue
        yield int(step["accepted_token_id"]), cur_pos, step["attn_to_previous_tokens"]


def _iter_draft_steps(trace, prompt_ids, generated_ids):
    """Yield (cur_token, cur_pos, {layer: tensor[num_heads, prev_len]}) for draft mode.

    For each decode step, accepted_token_ids[i] maps to draft_topk_trace[i].
    The attention is wrapped in a single-key dict {"draft_layer_000": tensor} for
    uniformity with the target format. Attention columns beyond cur_pos are clipped
    to exclude draft-tree KV expansion artifacts.
    """
    all_ids = list(prompt_ids) + list(generated_ids)
    prompt_len = len(prompt_ids)
    gen_offset = 0  # tokens generated before the current decode step

    for step in trace:
        accepted = step["accepted_token_ids"]
        topk = step.get("draft_topk_trace") or []
        for i, tok_id in enumerate(accepted):
            cur_pos = prompt_len + gen_offset + i
            if cur_pos >= len(all_ids):
                continue
            if i >= len(topk):
                continue
            attn_tensor = topk[i]["attn_to_previous_tokens"]  # [num_heads, prev_len_raw]
            # Clip to the actual number of previous main-sequence tokens (cur_pos).
            clipped = attn_tensor[:, :cur_pos]
            yield int(tok_id), cur_pos, {"draft_layer_000": clipped}
        gen_offset += len(accepted)


def main():
    parser = argparse.ArgumentParser(description="Analyze per-head echo and induction attention patterns.")
    parser.add_argument("--input-pt", type=str, required=True, help="Path to attn debug .pt file")
    parser.add_argument(
        "--mode",
        type=str,
        default="target",
        choices=["target", "draft"],
        help="Trace format: 'target' (accepted_token_attn_trace) or 'draft' (draft_accepted_attn_trace).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output CSV path. Default: <input>_head_patterns.csv",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Output JSON summary path. Default: <input>_head_patterns_summary.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top heads to keep in JSON summary.",
    )

    args = parser.parse_args()

    data = torch.load(args.input_pt, map_location="cpu")
    prompt_ids = data.get("prompt_token_ids", [])
    generated_ids = data.get("generated_token_ids", [])
    all_ids = list(prompt_ids) + list(generated_ids)
    prompt_len = len(prompt_ids)
    gen_len = len(generated_ids)

    if args.mode == "draft":
        trace_raw = data.get("draft_accepted_attn_trace")
        if not trace_raw:
            raise ValueError("draft_accepted_attn_trace is missing or empty.")
        step_iter = _iter_draft_steps(trace_raw, prompt_ids, generated_ids)
    else:
        trace_raw = data.get("accepted_token_attn_trace")
        if not trace_raw:
            raise ValueError("accepted_token_attn_trace is missing or empty.")
        step_iter = _iter_target_steps(trace_raw, prompt_ids, generated_ids)

    # Infer layer/head layout from first step.
    first_token, first_pos, first_per_layer = next(iter(
        _iter_draft_steps(trace_raw, prompt_ids, generated_ids)
        if args.mode == "draft"
        else _iter_target_steps(trace_raw, prompt_ids, generated_ids)
    ))
    layer_keys = sorted(first_per_layer.keys())
    if not layer_keys:
        raise ValueError("No layers found in attn_to_previous_tokens.")
    num_heads = int(first_per_layer[layer_keys[0]].shape[0])

    # Accumulators per (layer, head).
    acc = {}
    for layer in layer_keys:
        for head in range(num_heads):
            acc[(layer, head)] = {
                "echo_mass_sum": 0.0,
                "echo_hit_steps": 0,
                "echo_candidate_steps": 0,
                "ind_mass_sum": 0.0,
                "ind_hit_steps": 0,
                "ind_candidate_steps": 0,
                "steps_seen": 0,
            }

    # Traverse each accepted token in generation order.
    for cur_token, cur_pos, per_layer in step_iter:
        prev_ids = all_ids[:cur_pos]

        # Echo positions: previous positions with same token id as current token.
        echo_positions = [i for i, tid in enumerate(prev_ids) if tid == cur_token]

        # Induction positions: j+1 for every match at j, if j+1 < cur_pos.
        ind_positions = []
        for j in echo_positions:
            nxt = j + 1
            if nxt < cur_pos:
                ind_positions.append(nxt)
        if ind_positions:
            ind_positions = sorted(set(ind_positions))

        for layer in layer_keys:
            layer_scores = per_layer[layer]  # [num_heads, prev_len]
            prev_len = int(layer_scores.shape[1])

            valid_echo = [p for p in echo_positions if p < prev_len]
            valid_ind = [p for p in ind_positions if p < prev_len]

            for head in range(num_heads):
                row = layer_scores[head].float()
                state = acc[(layer, head)]
                state["steps_seen"] += 1

                if valid_echo:
                    mass = float(row[valid_echo].sum().item())
                    state["echo_mass_sum"] += mass
                    state["echo_candidate_steps"] += 1
                    if mass > 0:
                        state["echo_hit_steps"] += 1

                if valid_ind:
                    mass = float(row[valid_ind].sum().item())
                    state["ind_mass_sum"] += mass
                    state["ind_candidate_steps"] += 1
                    if mass > 0:
                        state["ind_hit_steps"] += 1

    rows = []
    for (layer, head), v in acc.items():
        echo_mass_avg = _safe_div(v["echo_mass_sum"], v["echo_candidate_steps"])
        ind_mass_avg = _safe_div(v["ind_mass_sum"], v["ind_candidate_steps"])
        echo_hit_rate = _safe_div(v["echo_hit_steps"], v["echo_candidate_steps"])
        ind_hit_rate = _safe_div(v["ind_hit_steps"], v["ind_candidate_steps"])

        rows.append(
            {
                "layer": layer,
                "head": head,
                "steps_seen": v["steps_seen"],
                "echo_candidate_steps": v["echo_candidate_steps"],
                "echo_mass_sum": v["echo_mass_sum"],
                "echo_mass_avg": echo_mass_avg,
                "echo_hit_rate": echo_hit_rate,
                "ind_candidate_steps": v["ind_candidate_steps"],
                "ind_mass_sum": v["ind_mass_sum"],
                "ind_mass_avg": ind_mass_avg,
                "ind_hit_rate": ind_hit_rate,
                # Simple combined ranking scores.
                "echo_score": echo_mass_avg * echo_hit_rate,
                "ind_score": ind_mass_avg * ind_hit_rate,
            }
        )

    rows.sort(key=lambda r: (r["layer"], r["head"]))

    input_path = Path(args.input_pt)
    out_csv = args.out_csv or str(input_path.with_name(input_path.stem + "_head_patterns.csv"))
    out_json = args.out_json or str(input_path.with_name(input_path.stem + "_head_patterns_summary.json"))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = [
        "layer",
        "head",
        "steps_seen",
        "echo_candidate_steps",
        "echo_mass_sum",
        "echo_mass_avg",
        "echo_hit_rate",
        "ind_candidate_steps",
        "ind_mass_sum",
        "ind_mass_avg",
        "ind_hit_rate",
        "echo_score",
        "ind_score",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    top_k = max(1, int(args.top_k))
    top_echo = sorted(rows, key=lambda r: r["echo_score"], reverse=True)[:top_k]
    top_ind = sorted(rows, key=lambda r: r["ind_score"], reverse=True)[:top_k]

    summary = {
        "meta": {
            "input_pt": args.input_pt,
            "mode": args.mode,
            "prompt_len": prompt_len,
            "generated_len": gen_len,
            "num_layers": len(layer_keys),
            "num_heads": num_heads,
            "top_k": top_k,
        },
        "top_echo_heads": top_echo,
        "top_induction_heads": top_ind,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved summary: {out_json}")
    if top_echo:
        t = top_echo[0]
        print(f"Top echo head: {t['layer']}/h{t['head']} score={t['echo_score']:.6f}")
    if top_ind:
        t = top_ind[0]
        print(f"Top induction head: {t['layer']}/h{t['head']} score={t['ind_score']:.6f}")


if __name__ == "__main__":
    main()
