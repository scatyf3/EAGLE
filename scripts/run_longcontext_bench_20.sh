#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

GROUP="${1:-all}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
N_SAMPLES="${N_SAMPLES:-20}"
GEN_LEN="${GEN_LEN:-32}"
BUCKET_DIR="${BUCKET_DIR:-eagle/data/longbench_ctx_subsampled_20}"

run_8k_plus() {
  cd "$ROOT"

  for method in ar eagle eagle-small eagle-linear triforce; do
    out="outputs/context_runs/${method//-/_}_llama2_8k-plus.jsonl"
    echo ">>> [${method}] 8k+ -> ${out}"

    if [ "$method" = "triforce" ]; then
      conda run -n QWen_DTD env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
        python scripts/bench_unified_longbench.py \
          --method triforce \
          --base-model NousResearch/Yarn-Llama-2-7b-128k \
          --draft-model JackFram/llama-68m \
          --data "$BUCKET_DIR/8k-plus.jsonl" \
          --n-samples "$N_SAMPLES" \
          --gen-len "$GEN_LEN" \
          --prefill 16384 \
          --gamma 6 \
          --output "$out"
    else
      conda run -n specreason env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
        python scripts/bench_unified_longbench.py \
          --method "$method" \
          --base-model NousResearch/Yarn-Llama-2-7b-128k \
          --ea-model yuhuili/EAGLE-llama2-chat-7B \
          --data "$BUCKET_DIR/8k-plus.jsonl" \
          --n-samples "$N_SAMPLES" \
          --gen-len "$GEN_LEN" \
          --max-input-tokens 8300 \
          --max-length 16384 \
          --gamma 6 \
          --output "$out"
    fi
  done
}

run_high_ranges() {
  cd "$ROOT"

  for b in 8k-16k 16k-32k 32k-64k; do
    case "$b" in
      8k-16k)
        max_in=16384
        max_len=32768
        ;;
      16k-32k)
        max_in=32768
        max_len=49152
        ;;
      32k-64k)
        max_in=65536
        max_len=98304
        ;;
      *)
        echo "Unsupported bucket: $b" >&2
        exit 1
        ;;
    esac

    for method in ar eagle eagle-small eagle-linear triforce; do
      out="outputs/context_runs/${method//-/_}_llama2_${b}.jsonl"
      echo ">>> [${method}] ${b} -> ${out}"

      if [ "$method" = "triforce" ]; then
        conda run -n QWen_DTD env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
          python scripts/bench_unified_longbench.py \
            --method triforce \
            --base-model NousResearch/Yarn-Llama-2-7b-128k \
            --draft-model JackFram/llama-68m \
            --data "$BUCKET_DIR/${b}.jsonl" \
            --n-samples "$N_SAMPLES" \
            --gen-len "$GEN_LEN" \
            --prefill "$max_in" \
            --gamma 6 \
            --output "$out"
      else
        conda run -n specreason env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
          python scripts/bench_unified_longbench.py \
            --method "$method" \
            --base-model NousResearch/Yarn-Llama-2-7b-128k \
            --ea-model yuhuili/EAGLE-llama2-chat-7B \
            --data "$BUCKET_DIR/${b}.jsonl" \
            --n-samples "$N_SAMPLES" \
            --gen-len "$GEN_LEN" \
            --max-input-tokens "$max_in" \
            --max-length "$max_len" \
            --gamma 6 \
            --output "$out"
      fi
    done
  done
}

run_standard() {
  cd "$ROOT"

  for b in lt256 256-512 512-1k 1k-2k 2k-4k 4k-8k; do
    case "$b" in
      lt256)   max_in=128 ;;
      256-512) max_in=384 ;;
      512-1k)  max_in=768 ;;
      1k-2k)   max_in=1536 ;;
      2k-4k)   max_in=3072 ;;
      4k-8k)   max_in=8192 ;;
      *)
        echo "Unsupported bucket: $b" >&2
        exit 1
        ;;
    esac

    for method in ar eagle eagle-small eagle-linear triforce; do
      out="outputs/context_runs/${method//-/_}_llama2_${b}.jsonl"
      echo ">>> [${method}] ${b} -> ${out}"

      if [ "$method" = "triforce" ]; then
        conda run -n QWen_DTD env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
          python scripts/bench_unified_longbench.py \
            --method triforce \
            --base-model NousResearch/Yarn-Llama-2-7b-128k \
            --draft-model JackFram/llama-68m \
            --data "$BUCKET_DIR/${b}.jsonl" \
            --n-samples "$N_SAMPLES" \
            --gen-len "$GEN_LEN" \
            --prefill "$max_in" \
            --gamma 6 \
            --output "$out"
      else
        conda run -n specreason env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
          python scripts/bench_unified_longbench.py \
            --method "$method" \
            --base-model NousResearch/Yarn-Llama-2-7b-128k \
            --ea-model yuhuili/EAGLE-llama2-chat-7B \
            --data "$BUCKET_DIR/${b}.jsonl" \
            --n-samples "$N_SAMPLES" \
            --gen-len "$GEN_LEN" \
            --max-input-tokens "$max_in" \
            --max-length 16384 \
            --gamma 6 \
            --output "$out"
      fi
    done
  done
}

case "$GROUP" in
  standard)
    run_standard
    ;;
  8k-plus)
    run_8k_plus
    ;;
  high-ranges)
    run_high_ranges
    ;;
  all)
    run_standard
    run_8k_plus
    run_high_ranges
    ;;
  *)
    echo "Usage: bash scripts/run_longcontext_bench_20.sh [standard|8k-plus|high-ranges|all]" >&2
    echo "Optional env vars: CUDA_VISIBLE_DEVICES, N_SAMPLES, GEN_LEN, BUCKET_DIR" >&2
    exit 1
    ;;
esac