#!/usr/bin/env python3
"""Generate a prompt by repeating K random token ids multiple times.

Example:
python scripts/generate_repeated_random_token_prompt.py \
  --tokenizer Qwen/Qwen3-1.7B \
  --k 2500 \
  --repeat 4 \
  --seed 42 \
  --out-prefix outputs/random_k2500_x4
"""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def build_valid_token_ids(tokenizer, avoid_special):
    vocab_size = tokenizer.vocab_size
    if vocab_size is None:
        vocab_size = len(tokenizer)

    all_ids = list(range(vocab_size))
    if not avoid_special:
        return all_ids

    special_ids = set(tokenizer.all_special_ids or [])
    return [tid for tid in all_ids if tid not in special_ids]


def main():
    parser = argparse.ArgumentParser(
        description="Generate K random token ids, repeat them N times, and export prompt artifacts."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer name/path, e.g. Qwen/Qwen3-1.7B",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2500,
        help="Number of random token ids before repetition.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=4,
        help="How many times to repeat the random token block.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--avoid-special",
        action="store_true",
        help="Exclude tokenizer special token ids when sampling.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="outputs/random_prompt",
        help="Output file prefix. Will write <prefix>.json and <prefix>.txt.",
    )

    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.repeat <= 0:
        raise ValueError("--repeat must be > 0")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    valid_token_ids = build_valid_token_ids(tokenizer, args.avoid_special)
    if not valid_token_ids:
        raise RuntimeError("No valid token ids available for sampling.")

    rng = random.Random(args.seed)
    block_token_ids = [rng.choice(valid_token_ids) for _ in range(args.k)]
    full_token_ids = block_token_ids * args.repeat

    prompt_text = tokenizer.decode(
        full_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    txt_path = out_prefix.with_suffix(".txt")

    payload = {
        "tokenizer": args.tokenizer,
        "k": args.k,
        "repeat": args.repeat,
        "seed": args.seed,
        "avoid_special": args.avoid_special,
        "block_token_ids": block_token_ids,
        "full_token_ids": full_token_ids,
        "num_full_tokens": len(full_token_ids),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(prompt_text)

    print(f"Saved token metadata: {json_path}")
    print(f"Saved decoded prompt text: {txt_path}")
    print(f"Total tokens in full prompt: {len(full_token_ids)}")


if __name__ == "__main__":
    main()
