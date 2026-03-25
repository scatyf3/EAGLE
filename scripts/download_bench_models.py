#!/usr/bin/env python3
"""
Download all model weights needed for the benchmark comparison.

Usage:
    conda run -n specreason python scripts/download_bench_models.py

Downloads to HuggingFace cache (default: ~/.cache/huggingface/hub).
"""

import os
import sys

def dl(repo_id: str, model_type: str = "model"):
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"{'='*60}")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=repo_id)
        print(f"  → {path}")
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    # ── EAGLE ──────────────────────────────────────────────────────
    # Base: Qwen/Qwen3-4B is usually already in cache
    # dl("Qwen/Qwen3-4B")  # uncomment if needed
    dl("AngelSlim/Qwen3-4B_eagle3")

    # ── LongSpec ───────────────────────────────────────────────────
    dl("gradientai/Llama-3-8B-Instruct-262k")
    dl("sail/longspec-Llama-3-8B-Instruct-262k")

    # ── TriForce ───────────────────────────────────────────────────
    dl("NousResearch/Yarn-Llama-2-7b-128k")
    dl("JackFram/llama-68m")

    print("\n✓ All downloads complete.")
