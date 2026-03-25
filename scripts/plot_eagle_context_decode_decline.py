#!/usr/bin/env python3
"""
Plot EAGLE decode throughput decline as context length grows.

Input format (TSV): outputs/benchmark_datapoints.tsv
Required columns:
  - dataset
  - avg_context_length
  - Decode only latency ms/tok (preferred)
Fallback column:
  - latency ms/tok
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def _to_float(x: str | None) -> float | None:
    if x is None:
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_points(tsv_path: Path) -> list[dict]:
    points: list[dict] = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            ctx = _to_float(row.get("avg_context_length"))
            decode_lat = _to_float(row.get("Decode only latency ms/tok"))
            src = "decode_only"
            if decode_lat is None:
                decode_lat = _to_float(row.get("latency ms/tok"))
                src = "end_to_end_fallback"

            if not dataset or ctx is None or decode_lat is None or decode_lat <= 0:
                continue

            decode_tps = 1000.0 / decode_lat
            points.append(
                {
                    "dataset": dataset,
                    "avg_context_length": ctx,
                    "decode_latency_ms_per_tok": decode_lat,
                    "decode_tps": decode_tps,
                    "source": src,
                }
            )

    points.sort(key=lambda r: r["avg_context_length"])
    return points


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def save_csv(points: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "avg_context_length",
                "decode_latency_ms_per_tok",
                "decode_tps",
                "source",
            ],
        )
        writer.writeheader()
        for p in points:
            writer.writerow(p)


def plot(points: list[dict], out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    xs = [p["avg_context_length"] for p in points]
    ys = [p["decode_tps"] for p in points]
    labels = [p["dataset"] for p in points]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Average Context Length (tokens, log scale)")
    plt.ylabel("Decode Throughput (tokens/s)")
    plt.title(title)

    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot EAGLE context length vs decode throughput")
    parser.add_argument(
        "--input",
        default="outputs/benchmark_datapoints.tsv",
        help="Input TSV path",
    )
    parser.add_argument(
        "--out-png",
        default="outputs/eagle_context_vs_decode_throughput.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/eagle_context_vs_decode_throughput.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    points = load_points(in_path)
    if not points:
        raise RuntimeError("No valid points loaded from input TSV.")

    save_csv(points, Path(args.out_csv))
    plot(points, Path(args.out_png), "EAGLE: Decode Throughput vs Context Length")

    xs = [p["avg_context_length"] for p in points]
    ys = [p["decode_tps"] for p in points]
    r = pearson([math.log10(x) for x in xs], ys)
    print(f"Saved PNG: {args.out_png}")
    print(f"Saved CSV: {args.out_csv}")
    print(f"Points: {len(points)}")
    if r is not None:
        print(f"Pearson r(log10(ctx), decode_tps) = {r:.4f}")


if __name__ == "__main__":
    main()
