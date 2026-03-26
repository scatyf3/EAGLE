#!/bin/bash
# Minimal benchmark: Qwen3-1.7B EAGLE vs AR baseline.
#
# Usage:
#   bash scripts/run_bench_qwen17_simple.sh [N_SAMPLES] [GEN_LEN]
#
# Defaults: N_SAMPLES=6, GEN_LEN=128

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

N_SAMPLES="${1:-6}"
GEN_LEN="${2:-128}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

DATA="$ROOT/outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl"
OUT="$ROOT/outputs/bench_eagle_qwen17_simple.jsonl"

echo "Running simple benchmark: EAGLE(Qwen3-1.7B) vs AR"
echo "  samples: $N_SAMPLES"
echo "  gen_len: $GEN_LEN"
echo "  data:    $DATA"
echo "  output:  $OUT"

CUDA_VISIBLE_DEVICES="$GPU" conda run -n specreason --no-capture-output \
  python "$SCRIPT_DIR/bench_eagle_longbench.py" \
  --data "$DATA" \
  --base-model Qwen/Qwen3-1.7B \
  --ea-model AngelSlim/Qwen3-1.7B_eagle3 \
  --n-samples "$N_SAMPLES" \
  --gen-len "$GEN_LEN" \
  --max-input-tokens 9000 \
  --max-length 19000 \
  --output "$OUT"

echo ""
echo "Done. Result file: $OUT"
python "$SCRIPT_DIR/analyze_bench_results.py" --eagle "$OUT"
