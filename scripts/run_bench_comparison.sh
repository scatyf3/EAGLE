#!/bin/bash
# run_bench_comparison.sh
# Runs all three speculative decoding systems on LongBench and shows comparison.
#
# Prerequisites:
#   1. Models downloaded (see scripts/download_bench_models.sh)
#   2. conda envs: specreason (EAGLE), QWen_DTD (LongSpec + TriForce)
#
# Usage:
#   bash scripts/run_bench_comparison.sh [N_SAMPLES] [GEN_LEN]
#
# Defaults: 10 samples, 200 output tokens each.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

N_SAMPLES="${1:-10}"
GEN_LEN="${2:-200}"
DATA="$ROOT/outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl"

GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "======================================================"
echo " LongBench Speculative Decoding Comparison"
echo "  N_SAMPLES  = $N_SAMPLES"
echo "  GEN_LEN    = $GEN_LEN"
echo "  DATA       = $DATA"
echo "  GPU        = $GPU"
echo "======================================================"

# ── 1. EAGLE ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/3] Running EAGLE (Qwen3-4B + eagle3 draft) ..."
CUDA_VISIBLE_DEVICES="$GPU" conda run -n specreason --no-capture-output \
    python "$SCRIPT_DIR/bench_eagle_longbench.py" \
        --data "$DATA" \
        --base-model Qwen/Qwen3-4B \
        --ea-model AngelSlim/Qwen3-4B_eagle3 \
        --n-samples "$N_SAMPLES" \
        --gen-len "$GEN_LEN" \
        --max-input-tokens 30000 \
        --output "$ROOT/outputs/bench_eagle_results.jsonl"

echo ">>> EAGLE done."

# ── 2. LongSpec ───────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/3] Running LongSpec (Llama-3-8B-262K + longspec draft) ..."
CUDA_VISIBLE_DEVICES="$GPU" conda run -n QWen_DTD --no-capture-output \
    python "$SCRIPT_DIR/bench_longspec_longbench.py" \
        --data "$DATA" \
        --target-model gradientai/Llama-3-8B-Instruct-262k \
        --draft-model sail/longspec-Llama-3-8B-Instruct-262k \
        --n-samples "$N_SAMPLES" \
        --gen-len "$GEN_LEN" \
        --max-input-tokens 30000 \
        --tree-shape 4 16 16 16 16 \
        --output "$ROOT/outputs/bench_longspec_results.jsonl"

echo ">>> LongSpec done."

# ── 3. TriForce ───────────────────────────────────────────────────────────────
echo ""
echo ">>> [3/3] Running TriForce (Yarn-Llama-2-7B-128K + 68M draft) ..."
CUDA_VISIBLE_DEVICES="$GPU" conda run -n QWen_DTD --no-capture-output \
    python "$SCRIPT_DIR/bench_triforce_longbench.py" \
        --data "$DATA" \
        --target-model NousResearch/Yarn-Llama-2-7b-128k \
        --draft-model JackFram/llama-68m \
        --n-samples "$N_SAMPLES" \
        --prefill 20000 \
        --gen-len "$GEN_LEN" \
        --budget 4096 \
        --gamma 6 \
        --output "$ROOT/outputs/bench_triforce_results.jsonl"

echo ">>> TriForce done."

# ── Analysis ──────────────────────────────────────────────────────────────────
echo ""
echo ">>> Generating comparison table ..."
python "$SCRIPT_DIR/analyze_bench_results.py" \
    --eagle   "$ROOT/outputs/bench_eagle_results.jsonl" \
    --longspec "$ROOT/outputs/bench_longspec_results.jsonl" \
    --triforce "$ROOT/outputs/bench_triforce_results.jsonl"

echo ""
echo "======================================================"
echo " Done. Results in outputs/bench_*_results.jsonl"
echo "======================================================"
