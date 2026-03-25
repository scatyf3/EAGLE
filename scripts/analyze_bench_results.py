#!/usr/bin/env python3
"""
Analyze and display the three-system speculative decoding comparison.

Legacy mode (old separate scripts):
    python scripts/analyze_bench_results.py \
        --eagle   outputs/bench_eagle_results.jsonl \
        --triforce outputs/bench_triforce_results.jsonl \
        --ar-unified outputs/bench_ar_unified.jsonl

Unified mode (new bench_unified_longbench.py output):
    python scripts/analyze_bench_results.py \
        --unified outputs/bench_unified_ar.jsonl \
                  outputs/bench_unified_eagle.jsonl \
                  outputs/bench_unified_triforce.jsonl

Unified mode shows: total_latency, prefill_tps, decode_tps, speedup vs AR.
"""

import argparse
import json
import os
import statistics
from pathlib import Path


def load(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 2),
        "median": round(statistics.median(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
    }


def summarize(records: list[dict], label: str, ar_unified: list[dict] | None = None) -> dict | None:
    if not records:
        return None

    r0 = records[0]
    model = r0.get("model", "?")
    draft = r0.get("ea_model") or r0.get("draft_model") or "?"

    ctx_lens = [r["ctx_tokens"] for r in records if "ctx_tokens" in r]
    # decode_tps is the primary metric (decode-only); fall back to tokens_per_sec for old files
    sd_dec   = [r["decode_tps"] for r in records if r.get("decode_tps")]
    sd_e2e   = [r["tokens_per_sec"] for r in records if r.get("tokens_per_sec")]
    sd_tps   = sd_dec if sd_dec else sd_e2e   # prefer decode-only
    accepts  = [r["acceptance_rate"] for r in records if r.get("acceptance_rate")]

    # Speedup: prefer unified AR baseline (decode-only vs decode-only)
    if ar_unified:
        # Match by sample_id
        ar_map = {r["sample_id"]: r for r in ar_unified}
        speedups = []
        ar_tps_vals = []
        for r in records:
            ar_r = ar_map.get(r["sample_id"])
            if ar_r is None:
                continue
            ar_dec = ar_r.get("decode_tps")
            my_dec = r.get("decode_tps") or r.get("tokens_per_sec")
            if ar_dec and my_dec and ar_dec > 0:
                speedups.append(my_dec / ar_dec)
                ar_tps_vals.append(ar_dec)
        ar_tps = ar_tps_vals
    else:
        ar_tps   = [r["ar_tokens_per_sec"] for r in records if r.get("ar_tokens_per_sec")]
        speedups = [r["speedup"] for r in records if r.get("speedup")]

    return {
        "system": label,
        "model": model,
        "draft": draft,
        "n_samples": len(records),
        "ctx_tokens": stats(ctx_lens),
        "sd_tps": stats(sd_tps),
        "ar_tps": stats(ar_tps),
        "acceptance_rate": stats(accepts),
        "speedup": stats(speedups),
        "metric": "decode-only" if sd_dec else "end-to-end",
        "ar_source": "unified" if ar_unified else "internal",
    }


def print_summary(s: dict):
    print(f"\n{'─'*60}")
    print(f"  System:  {s['system']}")
    print(f"  Model:   {s['model']}")
    print(f"  Draft:   {s['draft']}")
    print(f"  Samples: {s['n_samples']}")
    print(f"  Metric:  {s.get('metric','?')}  |  AR source: {s.get('ar_source','?')}")
    ctx = s["ctx_tokens"]
    print(f"  Ctx len (tokens): mean={ctx['mean']} min={ctx['min']} max={ctx['max']}")
    sd = s["sd_tps"]
    ar = s["ar_tps"]
    sp = s["speedup"]
    ac = s["acceptance_rate"]
    print(f"  SD  tok/s:        mean={sd['mean']} median={sd['median']}")
    if ar["n"]:
        print(f"  AR  tok/s:        mean={ar['mean']} median={ar['median']}")
    if sp["n"]:
        print(f"  Speedup vs AR:    mean={sp['mean']}x  median={sp['median']}x  max={sp['max']}x")
    if ac["n"]:
        print(f"  Accept rate:      mean={ac['mean']} (tokens per draft step)")


def print_comparison_table(summaries: list[dict]):
    valid = [s for s in summaries if s]
    if not valid:
        print("No results to compare.")
        return

    print(f"\n{'='*70}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")

    # Header
    col_w = 22
    header = f"{'Metric':<24}" + "".join(f"{s['system']:>{col_w}}" for s in valid)
    print(header)
    print("─" * (24 + col_w * len(valid)))

    def row(label, fn):
        vals = [fn(s) for s in valid]
        line = f"  {label:<22}" + "".join(
            f"{(str(v) if v is not None else 'N/A'):>{col_w}}" for v in vals
        )
        print(line)

    row("SD metric",         lambda s: s.get("metric", "?"))
    row("AR source",         lambda s: s.get("ar_source", "?"))
    row("Target model", lambda s: s["model"].split("/")[-1][:20])
    row("Draft model",  lambda s: s["draft"].split("/")[-1][:20])
    row("# samples",    lambda s: s["n_samples"])
    row("Ctx len mean (tok)", lambda s: s["ctx_tokens"]["mean"])
    row("SD mean tok/s",      lambda s: s["sd_tps"]["mean"])
    row("SD median tok/s",    lambda s: s["sd_tps"]["median"])
    row("AR mean tok/s",      lambda s: s["ar_tps"]["mean"])
    row("Speedup mean",       lambda s: f"{s['speedup']['mean']}x" if s["speedup"]["mean"] else "N/A")
    row("Speedup median",     lambda s: f"{s['speedup']['median']}x" if s["speedup"]["median"] else "N/A")
    row("Speedup max",        lambda s: f"{s['speedup']['max']}x" if s["speedup"]["max"] else "N/A")
    row("Accept rate mean",   lambda s: s["acceptance_rate"]["mean"])

    print(f"{'─'*70}")
    unified = any(s.get("ar_source") == "unified" for s in valid)
    note = "AR from unified neutral baseline; speedup = decode_tps / ar_decode_tps." if unified else \
           "Speedup is each system vs its own internal AR baseline."
    print(f"  Note: {note}")


# ──────────────────────────────────────────────────────────────────────────────
# Unified-format analysis (bench_unified_longbench.py output)
# ──────────────────────────────────────────────────────────────────────────────

def summarize_unified(records: list[dict]) -> dict | None:
    """Summarize records from bench_unified_longbench.py (has prefill_tps, decode_tps, total_tps)."""
    if not records:
        return None
    r0 = records[0]
    method = r0.get("method", "?")
    model  = r0.get("model", "?")
    draft  = r0.get("draft_model") or "—"
    return {
        "method":       method,
        "model":        model,
        "draft":        draft,
        "n_samples":    len(records),
        "ctx_tokens":   stats([r["ctx_tokens"]   for r in records if r.get("ctx_tokens")]),
        "total_latency":stats([r["total_latency"] for r in records if r.get("total_latency")]),
        "prefill_tps":  stats([r["prefill_tps"]   for r in records if r.get("prefill_tps")]),
        "decode_tps":   stats([r["decode_tps"]    for r in records if r.get("decode_tps")]),
        "total_tps":    stats([r["total_tps"]     for r in records if r.get("total_tps")]),
        "accept_rate":  stats([r["acceptance_rate"] for r in records
                               if r.get("acceptance_rate") is not None]),
    }


def print_unified_table(summaries: list[dict]):
    """
    Print a comparison table for unified-format summaries.
    AR baseline is used to compute decode speedup for SD methods.
    """
    valid = [s for s in summaries if s]
    if not valid:
        print("No results.")
        return

    # Locate AR baseline
    ar = next((s for s in valid if s["method"] == "ar"), None)
    ar_decode_mean = ar["decode_tps"]["mean"] if ar else None
    ar_latency_mean = ar["total_latency"]["mean"] if ar else None

    print(f"\n{'='*76}")
    print("  UNIFIED BENCHMARK — total_latency / prefill_tps / decode_tps")
    print(f"{'='*76}")

    col_w = 18
    methods = [s["method"].upper() for s in valid]
    header = f"  {'Metric':<26}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("  " + "─" * (26 + col_w * len(valid)))

    def row(label, fn):
        vals = [fn(s) for s in valid]
        line = f"  {label:<26}" + "".join(
            f"{(str(v) if v is not None else 'N/A'):>{col_w}}" for v in vals
        )
        print(line)

    row("Model",            lambda s: s["model"].split("/")[-1][:16])
    row("Draft",            lambda s: s["draft"].split("/")[-1][:16])
    row("# samples",        lambda s: s["n_samples"])
    row("Ctx len mean (tok)",lambda s: s["ctx_tokens"]["mean"])
    print("  " + "─" * (26 + col_w * len(valid)))
    row("Total latency (s) mean",  lambda s: s["total_latency"]["mean"])
    row("Total latency (s) median",lambda s: s["total_latency"]["median"])
    print("  " + "─" * (26 + col_w * len(valid)))
    row("Prefill tok/s mean", lambda s: s["prefill_tps"]["mean"])
    row("Decode  tok/s mean", lambda s: s["decode_tps"]["mean"])
    row("Total   tok/s mean", lambda s: s["total_tps"]["mean"])
    print("  " + "─" * (26 + col_w * len(valid)))

    def speedup_str(s):
        if s["method"] == "ar" or ar_decode_mean is None:
            return "baseline"
        d = s["decode_tps"]["mean"]
        if d is None:
            return "N/A"
        return f"{d/ar_decode_mean:.2f}x"

    def latency_speedup_str(s):
        if s["method"] == "ar" or ar_latency_mean is None:
            return "baseline"
        l = s["total_latency"]["mean"]
        if l is None:
            return "N/A"
        # lower latency = better; ratio = ar / sd
        return f"{ar_latency_mean/l:.2f}x"

    row("Decode speedup vs AR", speedup_str)
    row("Latency speedup vs AR", latency_speedup_str)
    row("Accept rate mean",  lambda s: s["accept_rate"]["mean"] if s["accept_rate"]["n"] else "—")
    print(f"  {'='*76}")
    if ar:
        print(f"  Speedup = SD decode_tps / AR decode_tps  |  AR: {ar['model'].split('/')[-1]}")
    print()


def main():
    parser = argparse.ArgumentParser()
    # Unified mode
    parser.add_argument("--unified", nargs="+", default=None, metavar="FILE",
                        help="One or more bench_unified_longbench.py output files. "
                             "Include the AR file first for speedup computation.")
    # Legacy mode
    parser.add_argument("--eagle",      default=None)
    parser.add_argument("--longspec",   default=None)
    parser.add_argument("--triforce",   default=None)
    parser.add_argument("--ar-unified", default=None,
                        help="Unified AR baseline (legacy). "
                             "Speedup = decode_tps / ar_unified decode_tps.")
    args = parser.parse_args()

    # ── Unified mode ────────────────────────────────────────────────────────
    if args.unified:
        all_summaries = []
        for path in args.unified:
            recs = load(path)
            if not recs:
                print(f"[analyze] WARNING: no records in {path}")
                continue
            s = summarize_unified(recs)
            if s:
                all_summaries.append(s)
                print(f"[analyze] Loaded {s['n_samples']} {s['method'].upper()} records "
                      f"from {path}")
        print_unified_table(all_summaries)
        return

    # ── Legacy mode ─────────────────────────────────────────────────────────
    eagle_recs    = load(args.eagle)
    longspec_recs = load(args.longspec)
    triforce_recs = load(args.triforce)
    ar_unified    = load(args.ar_unified) if args.ar_unified else None

    if ar_unified:
        print(f"[analyze] Using unified AR baseline: {args.ar_unified} ({len(ar_unified)} records)")

    s_eagle    = summarize(eagle_recs,    "EAGLE",    ar_unified)
    s_longspec = summarize(longspec_recs, "LongSpec", ar_unified)
    s_triforce = summarize(triforce_recs, "TriForce", ar_unified)

    for s in [s_eagle, s_longspec, s_triforce]:
        if s:
            print_summary(s)

    print_comparison_table([s_eagle, s_longspec, s_triforce])


if __name__ == "__main__":
    main()
