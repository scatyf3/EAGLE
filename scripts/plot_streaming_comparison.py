#!/usr/bin/env python
"""Plot timing and memory comparison between baseline and streaming LLM versions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_rows(jsonl_path: Path) -> list[dict]:
    """Load JSONL results file."""
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_bytes_per_token(rows: list[dict], max_length: int) -> int:
    """Infer bytes per token from KV cache capacity."""
    capacity = rows[0]["choices"][0]["memory_profile"]["totals"]["kv_cache_capacity_bytes"]
    return int(capacity) // int(max_length)


def build_frame(rows: list[dict], gpu_count: int, max_length: int) -> pd.DataFrame:
    """Build DataFrame from JSONL rows."""
    bytes_per_token = infer_bytes_per_token(rows, max_length=max_length)
    records = []

    for row in rows:
        choice = row["choices"][0]
        memory_profile = choice["memory_profile"]
        if len(memory_profile["per_device"]) != gpu_count:
            continue

        timing = row["linear_timing"]
        used_tokens = int(memory_profile["totals"]["kv_cache_used_bytes"] // bytes_per_token)
        wall_time = float(choice["wall_time"][0])
        target_self_attn = float(timing["target"]["self_attn_s"])
        target_other = float(timing["target"]["other_linear_s"])
        draft_self_attn = float(timing["draft"]["self_attn_s"])
        draft_other = float(timing["draft"]["other_linear_s"])
        target_total = target_self_attn + target_other
        draft_total = draft_self_attn + draft_other
        peak_allocated = float(memory_profile["totals"]["peak_allocated_bytes"])
        kv_used = float(memory_profile["totals"]["kv_cache_used_bytes"])

        records.append(
            {
                "question_id": int(row["question_id"]),
                "used_tokens": used_tokens,
                "context_k": used_tokens / 1000.0,
                "wall_time_s": wall_time,
                "prefill_time_s": float(choice["prefill_time"][0]),
                "target_self_attn_s": target_self_attn,
                "target_other_linear_s": target_other,
                "target_total_s": target_total,
                "draft_self_attn_s": draft_self_attn,
                "draft_other_linear_s": draft_other,
                "draft_total_s": draft_total,
                "target_self_attn_wall_pct": 100.0 * target_self_attn / wall_time if wall_time else 0.0,
                "draft_self_attn_wall_pct": 100.0 * draft_self_attn / wall_time if wall_time else 0.0,
                "target_kv_cache_peak_pct": 100.0 * kv_used / peak_allocated if peak_allocated else 0.0,
                "kv_cache_used_bytes": kv_used,
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No rows found for gpu_count={gpu_count}")
    return frame.sort_values(["used_tokens", "question_id"]).reset_index(drop=True)


def filter_context_range(frame: pd.DataFrame, min_tokens: int, max_tokens: int) -> pd.DataFrame:
    """Filter frame to context range."""
    return frame[(frame["used_tokens"] >= min_tokens) & (frame["used_tokens"] <= max_tokens)].copy()


def _bucket_label(context_k: float) -> str:
    """Generate bucket label."""
    rounded = max(1, int(round(context_k)))
    return f"~{rounded}k"


def aggregate_every_n(frame: pd.DataFrame, points_per_bucket: int) -> pd.DataFrame:
    """Aggregate frame by bucketing."""
    if frame.empty:
        return frame.copy()

    frame = frame.sort_values(["used_tokens", "question_id"]).reset_index(drop=True).copy()
    frame["bucket_id"] = frame.index // int(points_per_bucket)

    value_cols = [
        "used_tokens",
        "context_k",
        "wall_time_s",
        "prefill_time_s",
        "target_self_attn_s",
        "target_other_linear_s",
        "target_total_s",
        "draft_self_attn_s",
        "draft_other_linear_s",
        "draft_total_s",
        "target_self_attn_wall_pct",
        "draft_self_attn_wall_pct",
        "target_kv_cache_peak_pct",
        "kv_cache_used_bytes",
    ]

    grouped = frame.groupby("bucket_id", as_index=False)[value_cols].mean(numeric_only=True)
    grouped["bucket_label"] = grouped["context_k"].map(_bucket_label)

    denom = grouped["target_total_s"] + grouped["draft_total_s"]
    grouped["target_time_share_pct"] = (100.0 * grouped["target_total_s"] / denom).fillna(0.0)
    grouped["draft_time_share_pct"] = (100.0 * grouped["draft_total_s"] / denom).fillna(0.0)
    return grouped


def save_comparison_plot(baseline_frame: pd.DataFrame, streaming_frame: pd.DataFrame, out_path: Path) -> None:
    """Save comparison plot between baseline and streaming."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline
    y_pos = list(range(len(baseline_frame)))
    ax1.barh(
        y_pos,
        baseline_frame["wall_time_s"],
        color="#2563eb",
        alpha=0.8,
        label="Target time",
    )
    ax1.barh(
        y_pos,
        baseline_frame["draft_total_s"],
        left=baseline_frame["target_total_s"],
        color="#f59e0b",
        alpha=0.8,
        label="Draft time",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(baseline_frame["bucket_label"].tolist())
    ax1.set_xlabel("Total Time (seconds)")
    ax1.set_ylabel("Context bucket")
    ax1.set_title("Baseline (No KV Pruning)")
    ax1.grid(axis="x", alpha=0.3)
    ax1.legend()
    
    # Streaming
    y_pos = list(range(len(streaming_frame)))
    ax2.barh(
        y_pos,
        streaming_frame["wall_time_s"],
        color="#2563eb",
        alpha=0.8,
        label="Target time",
    )
    ax2.barh(
        y_pos,
        streaming_frame["draft_total_s"],
        left=streaming_frame["target_total_s"],
        color="#f59e0b",
        alpha=0.8,
        label="Draft time",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(streaming_frame["bucket_label"].tolist())
    ax2.set_xlabel("Total Time (seconds)")
    ax2.set_title("Streaming LLM (KV Pruned)")
    ax2.grid(axis="x", alpha=0.3)
    ax2.legend()
    
    fig.suptitle("Performance Comparison: Baseline vs Streaming LLM", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_memory_comparison_plot(baseline_frame: pd.DataFrame, streaming_frame: pd.DataFrame, out_path: Path) -> None:
    """Save memory usage comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = list(range(len(baseline_frame)))
    width = 0.35
    
    ax.bar(
        [p - width/2 for p in x_pos],
        baseline_frame["kv_cache_used_bytes"] / 1e9,
        width,
        label="Baseline KV Cache",
        color="#10b981",
        alpha=0.8,
    )
    ax.bar(
        [p + width/2 for p in x_pos],
        streaming_frame["kv_cache_used_bytes"] / 1e9,
        width,
        label="Streaming LLM KV Cache",
        color="#ef4444",
        alpha=0.8,
    )
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(baseline_frame["bucket_label"].tolist())
    ax.set_ylabel("KV Cache Used (GB)")
    ax.set_xlabel("Context bucket")
    ax.set_title("KV Cache Memory Usage Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_speedup_plot(baseline_frame: pd.DataFrame, streaming_frame: pd.DataFrame, out_path: Path) -> None:
    """Save speedup ratio plot."""
    # Merge frames on context bucket
    merged = baseline_frame[["bucket_label", "wall_time_s"]].copy()
    merged.columns = ["bucket_label", "baseline_wall_time_s"]
    merged["streaming_wall_time_s"] = streaming_frame["wall_time_s"].values
    merged["speedup"] = merged["baseline_wall_time_s"] / merged["streaming_wall_time_s"]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    colors = ["#10b981" if x > 1 else "#ef4444" for x in merged["speedup"]]
    ax.bar(
        range(len(merged)),
        merged["speedup"],
        color=colors,
        alpha=0.8,
    )
    
    ax.set_xticks(range(len(merged)))
    ax.set_xticklabels(merged["bucket_label"].tolist())
    ax.set_ylabel("Speedup Ratio (Baseline / Streaming)")
    ax.set_xlabel("Context bucket")
    ax.set_title("Speedup from Streaming LLM (>1 = faster)")
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and streaming LLM performance."
    )
    parser.add_argument(
        "--baseline-file",
        default="outputs/qwen3_1.7b_eagle3_1k_128k.jsonl",
        help="Path to baseline JSONL results file.",
    )
    parser.add_argument(
        "--streaming-file",
        default="outputs/qwen3_1.7b_eagle3_1k_128k_streaming.jsonl",
        help="Path to streaming JSONL results file.",
    )
    parser.add_argument(
        "--output-dir",
        default="figs/analysis/qwen3_1_7b_eagle3_1k_128k_comparison",
        help="Directory to write plots.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=3,
        help="Filter rows by number of visible CUDA devices.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=131072,
        help="Configured KV cache max length.",
    )
    parser.add_argument(
        "--range-min-tokens",
        type=int,
        default=1000,
        help="Lower bound for context range plots.",
    )
    parser.add_argument(
        "--range-max-tokens",
        type=int,
        default=8000,
        help="Upper bound for context range plots.",
    )
    parser.add_argument(
        "--points-per-bucket",
        type=int,
        default=5,
        help="Number of datapoints per averaging bucket.",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_file)
    streaming_path = Path(args.streaming_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline from {baseline_path}...")
    baseline_rows = load_rows(baseline_path)
    baseline_frame = build_frame(baseline_rows, gpu_count=args.gpu_count, max_length=args.max_length)
    baseline_range = filter_context_range(baseline_frame, min_tokens=args.range_min_tokens, max_tokens=args.range_max_tokens)
    baseline_bucketed = aggregate_every_n(baseline_range, points_per_bucket=args.points_per_bucket)

    print(f"Loading streaming from {streaming_path}...")
    streaming_rows = load_rows(streaming_path)
    streaming_frame = build_frame(streaming_rows, gpu_count=args.gpu_count, max_length=args.max_length)
    streaming_range = filter_context_range(streaming_frame, min_tokens=args.range_min_tokens, max_tokens=args.range_max_tokens)
    streaming_bucketed = aggregate_every_n(streaming_range, points_per_bucket=args.points_per_bucket)

    # Save comparison plots
    print("Generating comparison plots...")
    save_comparison_plot(baseline_bucketed, streaming_bucketed, output_dir / "performance_comparison.png")
    save_memory_comparison_plot(baseline_bucketed, streaming_bucketed, output_dir / "memory_comparison.png")
    save_speedup_plot(baseline_bucketed, streaming_bucketed, output_dir / "speedup_ratio.png")

    # Save summary CSVs
    baseline_bucketed.to_csv(output_dir / "baseline_summary.csv", index=False)
    streaming_bucketed.to_csv(output_dir / "streaming_summary.csv", index=False)

    print(f"Wrote comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
