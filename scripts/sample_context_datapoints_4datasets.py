#!/usr/bin/env python3
"""
Sample context-length datapoints from LongBench / GSM8K / MTBench / HumanEval
and aggregate them into one JSONL file.

Notes:
- LongBench uses the dataset-provided `length` field (token count).
- Other datasets estimate token count from text using a lightweight regex splitter.
"""

import argparse
import json
import random
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

LONG_ZIP = ROOT / "data/longbench_raw/data.zip"
LONG_PROMPT_CFG = ROOT / "LongBench-hip/config/dataset2prompt.json"

GSM8K_FILE = ROOT / "eagle/data/gsm8k/question.jsonl"
MTBENCH_FILE = ROOT / "eagle/data/mt_bench/question.jsonl"
HUMANEVAL_FILE = ROOT / "eagle/data/humaneval/question.jsonl"

LONG_ENGLISH_TASKS = [
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

BUCKETS = [
    ("lt256", 0, 256),
    ("256-512", 256, 512),
    ("512-1k", 512, 1024),
    ("1k-2k", 1024, 2048),
    ("2k-4k", 2048, 4096),
    ("4k-8k", 4096, 8192),
]


def _approx_token_len(text: str) -> int:
    # Simple tokenizer-like split on words/punctuation for a stable approximation.
    return len(re.findall(r"\w+|[^\w\s]", text))


def _bucket_of(length: int):
    for label, lo, hi in BUCKETS:
        if lo <= length < hi:
            return label
    return None


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_longbench():
    prompt_templates = json.loads(LONG_PROMPT_CFG.read_text(encoding="utf-8"))
    out = []
    with zipfile.ZipFile(LONG_ZIP) as zf:
        names = set(zf.namelist())
        for task in LONG_ENGLISH_TASKS:
            name = f"data/{task}.jsonl"
            if name not in names:
                continue
            tpl_key = TASK_TEMPLATE_KEY.get(task, task)
            tpl = prompt_templates.get(tpl_key)
            if tpl is None:
                continue
            with zf.open(name) as f:
                for raw in f:
                    if not raw.strip():
                        continue
                    ex = json.loads(raw)
                    try:
                        prompt = tpl.format(**ex)
                    except KeyError:
                        continue
                    length = int(ex.get("length", 0))
                    out.append(
                        {
                            "source_dataset": "longbench",
                            "source_category": task,
                            "question_id": ex.get("_id", ex.get("id", None)),
                            "context_tokens": length,
                            "turns": [prompt],
                            "reference": ex.get("answers", []),
                        }
                    )
    return out


def load_plain_dataset(path: Path, dataset: str):
    out = []
    for ex in _read_jsonl(path):
        turns = ex.get("turns") or []
        text = "\n".join(turns)
        length = _approx_token_len(text)
        out.append(
            {
                "source_dataset": dataset,
                "source_category": ex.get("category", ""),
                "question_id": ex.get("question_id", None),
                "context_tokens": length,
                "turns": turns,
                "reference": ex.get("reference", []),
            }
        )
    return out


def sample_points(candidates, per_bucket, min_non_longbench):
    random.shuffle(candidates)

    # Group by bucket first.
    by_bucket = {label: [] for label, _, _ in BUCKETS}
    for item in candidates:
        label = _bucket_of(item["context_tokens"])
        if label:
            by_bucket[label].append(item)

    selected = []
    for label, _, _ in BUCKETS:
        pool = by_bucket[label]
        if not pool:
            continue

        # First, take some non-longbench samples when available.
        picked = []
        non_long = [x for x in pool if x["source_dataset"] != "longbench"]
        random.shuffle(non_long)
        for item in non_long[:min_non_longbench]:
            picked.append(item)

        # Fill remaining with closest-to-midpoint samples for stable coverage.
        lo = next(v[1] for v in BUCKETS if v[0] == label)
        hi = next(v[2] for v in BUCKETS if v[0] == label)
        mid = (lo + hi) / 2.0
        remain = [x for x in pool if x not in picked]
        remain.sort(key=lambda x: abs(x["context_tokens"] - mid))
        picked.extend(remain[: max(0, per_bucket - len(picked))])

        selected.extend(picked[:per_bucket])

    # Deterministic ordering in output.
    selected.sort(key=lambda x: (x["context_tokens"], x["source_dataset"], str(x["question_id"])))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/context_datapoints_256_8k_4datasets.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-bucket", type=int, default=16)
    parser.add_argument(
        "--min-non-longbench-per-bucket",
        type=int,
        default=2,
        help="Try to include this many non-LongBench samples per bucket when available.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    all_points = []
    all_points.extend(load_longbench())
    all_points.extend(load_plain_dataset(GSM8K_FILE, "gsm8k"))
    all_points.extend(load_plain_dataset(MTBENCH_FILE, "mtbench"))
    all_points.extend(load_plain_dataset(HUMANEVAL_FILE, "humaneval"))

    # Keep only the requested range [0, 8192), including the short-context bucket.
    filtered = [x for x in all_points if 0 <= x["context_tokens"] < 8192]
    sampled = sample_points(
        filtered,
        per_bucket=args.per_bucket,
        min_non_longbench=args.min_non_longbench_per_bucket,
    )

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(sampled):
            row = {
                "sample_id": idx,
                "source_dataset": item["source_dataset"],
                "source_category": item["source_category"],
                "question_id": item["question_id"],
                "context_tokens": item["context_tokens"],
                "context_bucket": _bucket_of(item["context_tokens"]),
                "turns": item["turns"],
                "reference": item["reference"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Console summary.
    by_ds = {}
    by_bucket = {label: 0 for label, _, _ in BUCKETS}
    for item in sampled:
        ds = item["source_dataset"]
        by_ds[ds] = by_ds.get(ds, 0) + 1
        b = _bucket_of(item["context_tokens"])
        by_bucket[b] += 1

    print(f"Wrote {len(sampled)} samples to {out_path}")
    print("By dataset:")
    for ds in sorted(by_ds):
        print(f"  {ds:10s}: {by_ds[ds]}")
    print("By bucket:")
    for label, _, _ in BUCKETS:
        print(f"  {label:8s}: {by_bucket[label]}")


if __name__ == "__main__":
    main()
