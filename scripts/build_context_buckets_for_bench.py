"""Build benchmark-ready LongBench context bucket files.

This script samples prompts from raw LongBench data and writes JSONL files in
the schema consumed by scripts/bench_unified_longbench.py.

Example:
    python scripts/build_context_buckets_for_bench.py \
        --n-per-bucket 20 \
        --seed 42
"""

import argparse
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROMPT_CFG = ROOT / "LongBench-hip/config/dataset2prompt.json"
ZIP_PATH = ROOT / "data/longbench_raw/data.zip"
DEFAULT_OUTPUT_DIR = ROOT / "eagle/data/longbench_ctx_subsampled_20"
GSM8K_FILE = ROOT / "eagle/data/gsm8k/question.jsonl"
MTBENCH_FILE = ROOT / "eagle/data/mt_bench/question.jsonl"
HUMANEVAL_FILE = ROOT / "eagle/data/humaneval/question.jsonl"


ENGLISH_TASKS = [
    "narrativeqa",
    "qasper_e",
    "multifieldqa_en_e",
    "hotpotqa_e",
    "2wikimqa_e",
    "gov_report_e",
    "triviaqa_e",
    "repobench-p_e",
    "lcc_e",
    "passage_retrieval_en_e",
]


TASK_TEMPLATE_KEY = {
    "qasper_e": "qasper",
    "multifieldqa_en_e": "multifieldqa_en",
    "hotpotqa_e": "hotpotqa",
    "2wikimqa_e": "2wikimqa",
    "gov_report_e": "gov_report",
    "triviaqa_e": "triviaqa",
    "repobench-p_e": "repobench-p",
    "lcc_e": "lcc",
    "passage_retrieval_en_e": "passage_retrieval_en",
    "narrativeqa": "narrativeqa",
}


BUCKET_SPECS = [
    ("lt256", 0, 256),
    ("256-512", 256, 512),
    ("512-1k", 512, 1024),
    ("1k-2k", 1024, 2048),
    ("2k-4k", 2048, 4096),
    ("4k-8k", 4096, 8192),
    ("8k-plus", 8192, 32768),
    # These three labels were previously used as practical proxies for the
    # LongBench 8k / 16k / 32k sampled source buckets rather than strict cuts.
    ("8k-16k", 5500, 11000),
    ("16k-32k", 11000, 22000),
    ("32k-64k", 22000, 40000),
]


def load_prompt_templates():
    return json.loads(PROMPT_CFG.read_text(encoding="utf-8"))


def approx_token_len(text):
    return len(re.findall(r"\w+|[^\w\s]", text))


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_items_from_zip(prompt_templates):
    items = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for task in ENGLISH_TASKS:
            name = f"data/{task}.jsonl"
            if name not in zf.namelist():
                continue
            template_key = TASK_TEMPLATE_KEY.get(task, task)
            template = prompt_templates.get(template_key)
            if template is None:
                continue

            with zf.open(name) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    example = json.loads(line)
                    try:
                        prompt = template.format(**example)
                    except KeyError:
                        continue
                    items.append(
                        {
                            "task": task,
                            "dataset": "longbench",
                            "length": example.get("length", 0),
                            "prompt": prompt,
                        }
                    )
    return items


def load_plain_dataset(path, dataset_name):
    items = []
    for example in read_jsonl(path):
        turns = example.get("turns") or []
        prompt = "\n".join(turns)
        items.append(
            {
                "task": dataset_name,
                "dataset": dataset_name,
                "length": approx_token_len(prompt),
                "prompt": prompt,
            }
        )
    return items


def select_diverse_items(pool, n_per_bucket, seed, lower, upper):
    if len(pool) <= n_per_bucket:
        return list(pool)

    midpoint = (lower + upper) / 2
    by_task = defaultdict(list)
    for item in pool:
        by_task[item["task"]].append(item)

    for task_items in by_task.values():
        task_items.sort(key=lambda item: abs(item["length"] - midpoint))

    rng = random.Random(seed)
    task_names = sorted(by_task)
    rng.shuffle(task_names)
    selected = []
    offsets = {task: 0 for task in task_names}

    while len(selected) < n_per_bucket:
        progressed = False
        for task in list(task_names):
            idx = offsets[task]
            task_items = by_task[task]
            if idx >= len(task_items):
                continue
            selected.append(task_items[idx])
            offsets[task] += 1
            progressed = True
            if len(selected) >= n_per_bucket:
                break
        if not progressed:
            break
        task_names = [task for task in task_names if offsets[task] < len(by_task[task])]
        rng.shuffle(task_names)

    if len(selected) < n_per_bucket:
        seen = {id(item) for item in selected}
        remainder = sorted(pool, key=lambda item: abs(item["length"] - midpoint))
        for item in remainder:
            if id(item) in seen:
                continue
            selected.append(item)
            if len(selected) >= n_per_bucket:
                break

    return selected[:n_per_bucket]


def truncate_prompt_to_tokens(prompt, target_tokens):
    if approx_token_len(prompt) <= target_tokens:
        return prompt.strip()

    lo, hi = 1, len(prompt)
    best = prompt
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = prompt[:mid]
        tokens = approx_token_len(candidate)
        if tokens <= target_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    if " " in best:
        best = best.rsplit(" ", 1)[0]
    return best.strip()


def synthesize_shorter_items(items, existing_count, target_count, lower, upper, seed):
    if existing_count >= target_count:
        return []

    donors = [item for item in items if item["length"] > upper]
    donors = select_diverse_items(donors, target_count - existing_count, seed, upper, upper * 2)
    rng = random.Random(seed)
    synthetic = []
    for donor in donors:
        target_tokens = rng.randint(max(lower + 1, (lower + upper) // 2), upper)
        prompt = truncate_prompt_to_tokens(donor["prompt"], target_tokens)
        length = approx_token_len(prompt)
        if lower < length <= upper:
            synthetic.append(
                {
                    "task": donor["task"],
                    "dataset": donor["dataset"],
                    "length": length,
                    "prompt": prompt,
                }
            )
    return synthetic


def build_bucket_file(label, lower, upper, items, output_dir, n_per_bucket, seed):
    pool = [item for item in items if lower < item["length"] <= upper]
    chosen = select_diverse_items(pool, n_per_bucket, seed, lower, upper)
    if len(chosen) < n_per_bucket:
        chosen.extend(
            synthesize_shorter_items(
                items=items,
                existing_count=len(chosen),
                target_count=n_per_bucket,
                lower=lower,
                upper=upper,
                seed=seed + len(chosen),
            )
        )
        chosen = chosen[:n_per_bucket]
    out_path = output_dir / f"{label}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for sample_id, item in enumerate(chosen):
            row = {
                "sample_id": sample_id,
                "dataset": item["dataset"],
                "context_bucket": label,
                "context_tokens": item["length"],
                "context": item["prompt"],
                "input": "",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_templates = load_prompt_templates()
    items = load_items_from_zip(prompt_templates)
    items.extend(load_plain_dataset(GSM8K_FILE, "gsm8k"))
    items.extend(load_plain_dataset(MTBENCH_FILE, "mtbench"))
    items.extend(load_plain_dataset(HUMANEVAL_FILE, "humaneval"))

    print(f"Loaded {len(items)} prompts from LongBench and supplemental short-context sets")
    for label, lower, upper in BUCKET_SPECS:
        count = build_bucket_file(
            label=label,
            lower=lower,
            upper=upper,
            items=items,
            output_dir=output_dir,
            n_per_bucket=args.n_per_bucket,
            seed=args.seed,
        )
        print(f"{label}: wrote {count} rows")


if __name__ == "__main__":
    main()