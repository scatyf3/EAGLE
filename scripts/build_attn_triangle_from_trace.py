#!/usr/bin/env python3
"""Build a large lower-triangular attention matrix from accepted-token traces.

Input is a .pt file produced by attn_debug, containing:
  accepted_token_attn_trace: list of steps

Each step stores per-layer attention scores with shape [num_heads, prev_len], where
prev_len grows as accepted tokens are appended into the main sequence.
"""

import argparse
import os
from pathlib import Path

import torch


def _parse_layer_key(layer_name: str) -> str:
    if layer_name.startswith("layer_"):
        return layer_name
    return f"layer_{int(layer_name):03d}"


def _aggregate_row(attn_map, layer_key, head_mode):
    if layer_key not in attn_map:
        raise KeyError(f"Layer key {layer_key} not found in attention map.")
    layer_scores = attn_map[layer_key]  # [num_heads, prev_len]

    if head_mode == "mean":
        return layer_scores.float().mean(dim=0)
    head_idx = int(head_mode)
    if head_idx < 0 or head_idx >= layer_scores.shape[0]:
        raise IndexError(
            f"Head index {head_idx} out of range for {layer_key}, num_heads={layer_scores.shape[0]}"
        )
    return layer_scores[head_idx].float()


def main():
    parser = argparse.ArgumentParser(description="Merge accepted-token attention trace into a large lower-triangular matrix.")
    parser.add_argument("--input-pt", type=str, required=True, help="Path to attn debug .pt file.")
    parser.add_argument(
        "--layer",
        type=str,
        default="last",
        help="Layer to use: 'last' or integer index (e.g., 0, 27).",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="mean",
        help="Head aggregation: 'mean' or integer index (e.g., 0, 7).",
    )
    parser.add_argument(
        "--fill",
        type=float,
        default=float("nan"),
        help="Fill value for unavailable entries (prompt rows, upper triangle).",
    )
    parser.add_argument(
        "--out-pt",
        type=str,
        default=None,
        help="Output path for merged matrix (.pt). Defaults to input basename + _triangle.pt.",
    )

    args = parser.parse_args()

    src = torch.load(args.input_pt, map_location="cpu")
    trace = src.get("accepted_token_attn_trace")
    if not trace:
        raise ValueError("accepted_token_attn_trace is empty or missing.")

    prompt_ids = src.get("prompt_token_ids", [])
    prompt_len = len(prompt_ids)
    gen_len = len(trace)
    total_len = prompt_len + gen_len

    first_attn = trace[0]["attn_to_previous_tokens"]
    layer_keys = sorted(first_attn.keys())
    if not layer_keys:
        raise ValueError("No layer attention scores found in trace.")

    if args.layer == "last":
        layer_key = layer_keys[-1]
    else:
        layer_key = _parse_layer_key(args.layer)

    matrix = torch.full((total_len, total_len), float(args.fill), dtype=torch.float32)

    # Fill each accepted token row at its absolute sequence position.
    # Row index: prompt_len + step_idx
    # Available scores cover cols [0, row_idx-1]
    for step in trace:
        step_idx = int(step["step_idx"])
        row_idx = prompt_len + step_idx
        attn_map = step["attn_to_previous_tokens"]
        row_scores = _aggregate_row(attn_map, layer_key, args.head)
        prev_len = row_scores.shape[0]
        matrix[row_idx, :prev_len] = row_scores

    out_path = args.out_pt
    if out_path is None:
        in_path = Path(args.input_pt)
        out_path = str(in_path.with_name(in_path.stem + "_triangle.pt"))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_payload = {
        "meta": {
            "source": args.input_pt,
            "layer_key": layer_key,
            "head": args.head,
            "prompt_len": prompt_len,
            "generated_len": gen_len,
            "total_len": total_len,
            "fill": args.fill,
        },
        "triangle_attn": matrix,
    }
    torch.save(out_payload, out_path)

    print(f"Saved: {out_path}")
    print(f"matrix_shape={tuple(matrix.shape)} layer={layer_key} head={args.head}")


if __name__ == "__main__":
    main()
