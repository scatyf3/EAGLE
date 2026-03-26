#!/usr/bin/env python3

import json
from pathlib import Path
import statistics as st

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "context_runs"
BUCKETS = ["lt256", "256-512", "512-1k", "1k-2k", "2k-4k", "4k-8k"]


def mean_metric(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return st.mean(values) if values else float("nan")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    eagle_accept = []
    eagle_decode = []
    eagle_latency = []

    triforce_s2_rate = []
    triforce_s2_avg_len = []
    triforce_decode = []
    triforce_latency = []
    triforce_budget = []

    for bucket in BUCKETS:
        eagle_rows = load_jsonl(RUN_DIR / f"eagle_llama2_{bucket}.jsonl")
        triforce_rows = load_jsonl(RUN_DIR / f"triforce_llama2_{bucket}.jsonl")

        eagle_accept.append(mean_metric(eagle_rows, "acceptance_rate"))
        eagle_decode.append(mean_metric(eagle_rows, "decode_tps"))
        eagle_latency.append(mean_metric(eagle_rows, "total_latency"))

        triforce_s2_rate.append(mean_metric(triforce_rows, "s2_rate"))
        triforce_s2_avg_len.append(mean_metric(triforce_rows, "s2_avg_acc_len"))
        triforce_decode.append(mean_metric(triforce_rows, "decode_tps"))
        triforce_latency.append(mean_metric(triforce_rows, "total_latency"))
        triforce_budget.append(triforce_rows[0].get("budget", "?"))

    x = list(range(len(BUCKETS)))
    xlabels = [f"{bucket}\nB={budget}" for bucket, budget in zip(BUCKETS, triforce_budget)]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)

    ax = axes[0]
    ax.plot(x, eagle_latency, marker="o", linewidth=2.2, color="#0f766e", label="EAGLE latency")
    ax.plot(x, triforce_latency, marker="s", linewidth=2.2, color="#b45309", label="TriForce latency")
    ax.set_title("Llama2-7B: EAGLE vs TriForce Across Context Buckets")
    ax.set_ylabel("Latency (s)")
    ax.set_xticks(x, xlabels)
    ax.legend(frameon=True)

    ax = axes[1]
    ax.plot(x, eagle_decode, marker="o", linewidth=2.2, color="#0f766e", label="EAGLE decode tok/s")
    ax.plot(x, triforce_decode, marker="s", linewidth=2.2, color="#b45309", label="TriForce decode tok/s")
    ax.set_ylabel("Decode tok/s")
    ax.set_xticks(x, xlabels)
    ax.legend(frameon=True)

    ax = axes[2]
    ax.plot(x, eagle_accept, marker="o", linewidth=2.2, color="#0f766e", label="EAGLE accept/token-step")
    ax.plot(x, triforce_s2_avg_len, marker="s", linewidth=2.2, color="#b45309", label="TriForce S2 avg acc len")
    ax.set_ylabel("Accepted Tokens / Step")
    ax.set_xticks(x, xlabels)

    ax2 = ax.twinx()
    ax2.plot(x, triforce_s2_rate, marker="^", linestyle="--", linewidth=1.8, color="#7c3aed", label="TriForce S2 rate")
    ax2.set_ylabel("TriForce S2 Rate")
    ax2.set_ylim(0, 1.0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    note = (
        "Acceptance metrics are not identical: EAGLE uses accepted tokens per tree step; "
        "TriForce S2 rate is token-level acceptance, and S2 avg acc len is rate * gamma."
    )
    fig.text(0.01, 0.005, note, fontsize=9)

    png_path = RUN_DIR / "llama2_eagle_vs_triforce_buckets.png"
    svg_path = RUN_DIR / "llama2_eagle_vs_triforce_buckets.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()