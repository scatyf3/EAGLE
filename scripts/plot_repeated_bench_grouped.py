#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else 0.0


def plot_context_bucket(rows, out_path: Path):
    labels = [row["context_bucket"] for row in rows]
    total = [to_float(row, "mean_total_tps") for row in rows]
    prefill = [to_float(row, "mean_prefill_tps") for row in rows]
    decode = [to_float(row, "mean_decode_tps") for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(labels, total, marker="o", linewidth=2, label="Total tok/s")
    ax.plot(labels, decode, marker="o", linewidth=2, label="Decode tok/s")
    ax.plot(labels, prefill, marker="o", linewidth=2, label="Prefill tok/s")
    ax.set_title("EAGLE Qwen3 Repeated Benchmark by Context Bucket")
    ax.set_xlabel("Context Bucket")
    ax.set_ylabel("Tokens / second")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_source_dataset(rows, out_path: Path):
    labels = [row["source_dataset"] for row in rows]
    total = [to_float(row, "mean_total_tps") for row in rows]
    decode = [to_float(row, "mean_decode_tps") for row in rows]
    prefill = [to_float(row, "mean_prefill_tps") for row in rows]

    x = list(range(len(labels)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width for i in x], total, width=width, label="Total tok/s")
    ax.bar(x, decode, width=width, label="Decode tok/s")
    ax.bar([i + width for i in x], prefill, width=width, label="Prefill tok/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("EAGLE Qwen3 Repeated Benchmark by Source Dataset")
    ax.set_xlabel("Source Dataset")
    ax.set_ylabel("Tokens / second")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-csv", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--out-context-png", required=True)
    parser.add_argument("--out-dataset-png", required=True)
    args = parser.parse_args()

    context_rows = load_csv(Path(args.context_csv))
    dataset_rows = load_csv(Path(args.dataset_csv))

    out_context = Path(args.out_context_png)
    out_dataset = Path(args.out_dataset_png)
    out_context.parent.mkdir(parents=True, exist_ok=True)
    out_dataset.parent.mkdir(parents=True, exist_ok=True)

    plot_context_bucket(context_rows, out_context)
    plot_source_dataset(dataset_rows, out_dataset)

    print(f"Saved: {out_context}")
    print(f"Saved: {out_dataset}")


if __name__ == "__main__":
    main()