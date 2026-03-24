"""
Sample LongBench prompts at target context-length buckets and output
a question.jsonl compatible with gen_ea_answer_qwen3.py.

Usage:
    python scripts/sample_longbench_by_length.py \
        --output eagle/data/longbench_ctx_sampled/question.jsonl \
        --n-per-bucket 5 \
        --seed 42

Output format per line:
    {"question_id": int, "category": str, "turns": [prompt], "reference": [...]}
"""
import argparse
import json
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Prompt templates from LongBench-hip
PROMPT_CFG = Path("/mnt/hdd/yf/LongBench-hip/config/dataset2prompt.json")
ZIP_PATH = ROOT / "data/longbench_raw/data.zip"

# Only English tasks; skip Chinese / few-shot tasks that don't have a `context` field
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

# For tasks where the prompt template key differs from the file suffix
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

# Target buckets: (label, lower_exclusive, upper_inclusive)
BUCKETS = [
    ("1k",     0,     1200),
    ("2k",  1200,     2500),
    ("4k",  2500,     5500),
    ("8k",  5500,    11000),
    ("16k", 11000,   22000),
    ("32k", 22000,   40000),
]


def load_prompts_from_zip(zf, task, prompt_templates):
    fname = f"data/{task}.jsonl"
    if fname not in zf.namelist():
        return []
    tpl_key = TASK_TEMPLATE_KEY.get(task, task)
    tpl = prompt_templates.get(tpl_key)
    if tpl is None:
        return []

    items = []
    with zf.open(fname) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            try:
                prompt = tpl.format(**ex)
            except KeyError:
                continue
            items.append({
                "task": task,
                "length": ex.get("length", 0),
                "prompt": prompt,
                "answers": ex.get("answers", []),
            })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eagle/data/longbench_ctx_sampled/question.jsonl")
    parser.add_argument("--n-per-bucket", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    prompt_templates = json.loads(PROMPT_CFG.read_text())

    # Load all items from all English tasks
    print(f"Loading from {ZIP_PATH} ...")
    all_items = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for task in ENGLISH_TASKS:
            items = load_prompts_from_zip(zf, task, prompt_templates)
            all_items.extend(items)
            print(f"  {task}: {len(items)} items")

    print(f"Total items: {len(all_items)}")

    # Bucket assignment
    def get_bucket(length):
        for label, lo, hi in BUCKETS:
            if lo < length <= hi:
                return label
        return None

    bucketed = {label: [] for label, _, _ in BUCKETS}
    for item in all_items:
        b = get_bucket(item["length"])
        if b:
            bucketed[b].append(item)

    print("\nBucket counts (before sampling):")
    for label, _, _ in BUCKETS:
        print(f"  {label:5s}: {len(bucketed[label])} items")

    # Sample and write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qid = 0
    written = []
    for label, _, _ in BUCKETS:
        pool = bucketed[label]
        if not pool:
            print(f"  WARNING: no items for bucket {label}")
            continue
        # sort by distance to bucket midpoint for more representative samples
        lo = [lo for l2, lo, _ in BUCKETS if l2 == label][0]
        hi = [hi for l2, _, hi in BUCKETS if l2 == label][0]
        mid = (lo + hi) / 2
        pool.sort(key=lambda x: abs(x["length"] - mid))
        # take top candidates then shuffle for diversity
        candidates = pool[: max(args.n_per_bucket * 4, 20)]
        random.shuffle(candidates)
        sampled = candidates[: args.n_per_bucket]

        for item in sampled:
            row = {
                "question_id": qid,
                "category": f"longbench-{item['task']}-{label}",
                "turns": [item["prompt"]],
                "reference": item["answers"],
                "ctx_length_bucket": label,
                "ctx_length_tokens": item["length"],
            }
            written.append(row)
            qid += 1

    with out_path.open("w", encoding="utf-8") as fout:
        for row in written:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(written)} samples to {out_path}")
    print("\nSummary:")
    from collections import Counter
    by_bucket = Counter(r["ctx_length_bucket"] for r in written)
    for label, _, _ in BUCKETS:
        cnt = by_bucket.get(label, 0)
        if cnt:
            lengths = [r["ctx_length_tokens"] for r in written if r["ctx_length_bucket"] == label]
            tasks = [r["category"].split("-")[1] for r in written if r["ctx_length_bucket"] == label]
            print(f"  {label:5s}: {cnt} samples  lengths={[round(l/1000,1) for l in lengths]}k  tasks={tasks}")


if __name__ == "__main__":
    main()
