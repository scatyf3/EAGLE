# Long-Context Benchmark README

This README documents the long-context benchmark workflow used in this repo, including:

1. Dataset preparation
2. Run commands (AR / EAGLE / EAGLE-small / EAGLE-linear / TriForce)

All paths below are relative to repo root.

## 1) Dataset

### 1.1 Bucket files used for testing

Prepared bucket files are under:

- `eagle/data/longbench_ctx_subsampled_20/lt256.jsonl`
- `eagle/data/longbench_ctx_subsampled_20/256-512.jsonl`
- `eagle/data/longbench_ctx_subsampled_20/512-1k.jsonl`
- `eagle/data/longbench_ctx_subsampled_20/1k-2k.jsonl`
- `eagle/data/longbench_ctx_subsampled_20/2k-4k.jsonl`
- `eagle/data/longbench_ctx_subsampled_20/4k-8k.jsonl`
- `eagle/data/longbench_ctx_subsampled_20/8k-plus.jsonl`
- `eagle/data/longbench_ctx_subsampled_20/8k-16k.jsonl`
- `eagle/data/longbench_ctx_subsampled_20/16k-32k.jsonl`
- `eagle/data/longbench_ctx_subsampled_20/32k-64k.jsonl`

Default sampling target is 20 rows per bucket file.

Each sample is a JSON object with fields like:

- `sample_id`
- `dataset`
- `context_bucket`
- `context_tokens`
- `context`
- `input`

### 1.2 Rebuild bucket datasets (optional)

If you want to regenerate all benchmark bucket files with 20 rows per bucket from raw LongBench source plus supplemental short-context sets:

```bash
cd /mnt/hdd/yf/EAGLE
python scripts/build_context_buckets_for_bench.py --n-per-bucket 20 --seed 42
```

The generated subsample dataset is stored inside the repo under:

- `eagle/data/longbench_ctx_subsampled_20/`

## 2) Run methods

Unified entry script:

- `scripts/bench_unified_longbench.py`

Supported methods:

- `ar`
- `eagle`
- `eagle-small`
- `eagle-linear`
- `triforce`

### 2.1 Environment notes

- AR/EAGLE/EAGLE-small/EAGLE-linear: use conda env `specreason`
- TriForce: use conda env `QWen_DTD`
- Recommended to set `PYTHONPATH=.`

### 2.2 Five-method run on 8k-plus

Standalone script:

```bash
cd /mnt/hdd/yf/EAGLE
bash scripts/run_longcontext_bench_20.sh 8k-plus
```

```bash
cd /mnt/hdd/yf/EAGLE

for method in ar eagle eagle-small eagle-linear triforce; do
  out="outputs/context_runs/${method//-/_}_llama2_8k-plus.jsonl"
  echo ">>> [${method}] 8k+ -> ${out}"

  if [ "$method" = "triforce" ]; then
    conda run -n QWen_DTD env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
      python scripts/bench_unified_longbench.py \
        --method triforce \
        --base-model NousResearch/Yarn-Llama-2-7b-128k \
        --draft-model JackFram/llama-68m \
        --data eagle/data/longbench_ctx_subsampled_20/8k-plus.jsonl \
        --n-samples 20 \
        --gen-len 32 \
        --prefill 16384 \
        --gamma 6 \
        --output "$out"
  else
    conda run -n specreason env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
      python scripts/bench_unified_longbench.py \
        --method "$method" \
        --base-model NousResearch/Yarn-Llama-2-7b-128k \
        --ea-model yuhuili/EAGLE-llama2-chat-7B \
        --data eagle/data/longbench_ctx_subsampled_20/8k-plus.jsonl \
        --n-samples 20 \
        --gen-len 32 \
        --max-input-tokens 8300 \
        --max-length 16384 \
        --gamma 6 \
        --output "$out"
  fi
done
```

Note:

- For EAGLE-family methods on long context, GPU memory pressure can be high. If OOM occurs, lower `--max-input-tokens` while keeping `8k+` requirement.

### 2.3 Run all six standard buckets (<=8k)

Standalone script:

```bash
cd /mnt/hdd/yf/EAGLE
bash scripts/run_longcontext_bench_20.sh standard
```

```bash
cd /mnt/hdd/yf/EAGLE

for b in lt256 256-512 512-1k 1k-2k 2k-4k 4k-8k; do
  case "$b" in
    lt256)   max_in=128 ;;
    256-512) max_in=384 ;;
    512-1k)  max_in=768 ;;
    1k-2k)   max_in=1536 ;;
    2k-4k)   max_in=3072 ;;
    4k-8k)   max_in=8192 ;;
  esac

  for method in ar eagle eagle-small eagle-linear triforce; do
    out="outputs/context_runs/${method//-/_}_llama2_${b}.jsonl"
    echo ">>> [${method}] ${b} -> ${out}"

    if [ "$method" = "triforce" ]; then
      conda run -n QWen_DTD env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python scripts/bench_unified_longbench.py \
          --method triforce \
          --base-model NousResearch/Yarn-Llama-2-7b-128k \
          --draft-model JackFram/llama-68m \
          --data eagle/data/longbench_ctx_subsampled_20/${b}.jsonl \
          --n-samples 20 \
          --gen-len 32 \
          --prefill ${max_in} \
          --gamma 6 \
          --output "$out"
    else
      conda run -n specreason env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python scripts/bench_unified_longbench.py \
          --method "$method" \
          --base-model NousResearch/Yarn-Llama-2-7b-128k \
          --ea-model yuhuili/EAGLE-llama2-chat-7B \
          --data eagle/data/longbench_ctx_subsampled_20/${b}.jsonl \
          --n-samples 20 \
          --gen-len 32 \
          --max-input-tokens ${max_in} \
          --max-length 16384 \
          --gamma 6 \
          --output "$out"
    fi
  done
done
```

### 2.4 Run three higher context buckets

Standalone script:

```bash
cd /mnt/hdd/yf/EAGLE
bash scripts/run_longcontext_bench_20.sh high-ranges
```

```bash
cd /mnt/hdd/yf/EAGLE

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
  esac

  for method in ar eagle eagle-small eagle-linear triforce; do
    out="outputs/context_runs/${method//-/_}_llama2_${b}.jsonl"
    echo ">>> [${method}] ${b} -> ${out}"

    if [ "$method" = "triforce" ]; then
      conda run -n QWen_DTD env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python scripts/bench_unified_longbench.py \
          --method triforce \
          --base-model NousResearch/Yarn-Llama-2-7b-128k \
          --draft-model JackFram/llama-68m \
          --data eagle/data/longbench_ctx_subsampled_20/${b}.jsonl \
          --n-samples 20 \
          --gen-len 32 \
          --prefill ${max_in} \
          --gamma 6 \
          --output "$out"
    else
      conda run -n specreason env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python scripts/bench_unified_longbench.py \
          --method "$method" \
          --base-model NousResearch/Yarn-Llama-2-7b-128k \
          --ea-model yuhuili/EAGLE-llama2-chat-7B \
          --data eagle/data/longbench_ctx_subsampled_20/${b}.jsonl \
          --n-samples 20 \
          --gen-len 32 \
          --max-input-tokens ${max_in} \
          --max-length ${max_len} \
          --gamma 6 \
          --output "$out"
    fi
  done
done
```

Note:

- These three buckets are much heavier than `8k-plus`; EAGLE-family methods may OOM depending on GPU memory.
- If needed, lower `--max-input-tokens` or use `BUCKET_DIR=... bash scripts/run_longcontext_bench_20.sh high-ranges` with custom data.

Run both groups in one shot:

```bash
cd /mnt/hdd/yf/EAGLE
bash scripts/run_longcontext_bench_20.sh all
```

## 3) Outputs and summary files

Per-run JSONL outputs are saved in:

- `outputs/context_runs/`

Main aggregate tables/figures used in this repo:

- `outputs/context_runs/llama2_eagle_vs_triforce_buckets.tsv`
- `outputs/context_runs/llama2_eagle_vs_triforce_buckets_transposed.tsv`
- `outputs/context_runs/llama2_eagle_vs_triforce_buckets.png`
- `outputs/context_runs/llama2_eagle_vs_triforce_buckets.svg`

## 4) Quick sanity checks

Check line counts for 8k-plus outputs:

```bash
cd /mnt/hdd/yf/EAGLE
wc -l outputs/context_runs/*_llama2_8k-plus.jsonl
```

Expected: 20 lines per method if all runs complete.
