#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stats(records):
    acc_values = [r.get("acceptance_rate") for r in records if r.get("acceptance_rate") is not None]
    return {
        "runs": len(records),
        "questions": len({(r.get("question_idx"), r.get("question_id"), r.get("sample_id")) for r in records}),
        "mean_context_tokens": mean([r.get("context_tokens", 0) for r in records]),
        "mean_prompt_tokens": mean([r.get("prompt_tokens", 0) for r in records]),
        "mean_new_tokens": mean([r.get("new_tokens", 0) for r in records]),
        "mean_total_tps": mean([r.get("total_tps", 0.0) for r in records]),
        "mean_prefill_tps": mean([r.get("prefill_tps", 0.0) for r in records]),
        "mean_decode_tps": mean([r.get("decode_tps", 0.0) for r in records]),
        "mean_acceptance_rate": mean(acc_values),
        "mean_total_time_s": mean([r.get("total_time_s", 0.0) for r in records]),
        "mean_prefill_time_s": mean([r.get("prefill_time_s", 0.0) for r in records]),
        "mean_decode_time_s": mean([r.get("decode_time_s", 0.0) for r in records]),
    }


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_group_table(records, group_key):
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get(group_key, "")] .append(record)

    rows = []
    for key in sorted(grouped, key=lambda x: (str(x))):
        row = {group_key: key}
        row.update(stats(grouped[key]))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-raw-csv", required=True)
    parser.add_argument("--out-context-csv", required=True)
    parser.add_argument("--out-dataset-csv", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    records = load_jsonl(input_path)
    if not records:
        raise RuntimeError(f"No records found in {input_path}")

    raw_fieldnames = [
        "question_idx",
        "repeat_idx",
        "question_id",
        "sample_id",
        "source_dataset",
        "source_category",
        "context_bucket",
        "context_tokens",
        "num_turns",
        "prompt_tokens",
        "new_tokens",
        "total_time_s",
        "prefill_time_s",
        "decode_time_s",
        "total_tps",
        "prefill_tps",
        "decode_tps",
        "acceptance_rate",
        "draft_steps",
    ]
    write_csv(Path(args.out_raw_csv), records, raw_fieldnames)

    grouped_fieldnames = [
        None,
        "runs",
        "questions",
        "mean_context_tokens",
        "mean_prompt_tokens",
        "mean_new_tokens",
        "mean_total_tps",
        "mean_prefill_tps",
        "mean_decode_tps",
        "mean_acceptance_rate",
        "mean_total_time_s",
        "mean_prefill_time_s",
        "mean_decode_time_s",
    ]

    context_rows = build_group_table(records, "context_bucket")
    context_fields = ["context_bucket"] + [f for f in grouped_fieldnames[1:]]
    write_csv(Path(args.out_context_csv), context_rows, context_fields)

    dataset_rows = build_group_table(records, "source_dataset")
    dataset_fields = ["source_dataset"] + [f for f in grouped_fieldnames[1:]]
    write_csv(Path(args.out_dataset_csv), dataset_rows, dataset_fields)

    print(f"Raw CSV: {args.out_raw_csv}")
    print(f"By context bucket CSV: {args.out_context_csv}")
    print(f"By source dataset CSV: {args.out_dataset_csv}")


if __name__ == "__main__":
    main()