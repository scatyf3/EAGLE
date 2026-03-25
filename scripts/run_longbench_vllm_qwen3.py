import argparse
import json
import os
import time

import requests
import shortuuid
from fastchat.llm_judge.common import load_questions


def truncate_text(text, max_chars):
    if not max_chars or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + text[-half:]


def main():
    parser = argparse.ArgumentParser(description="Run LongBench with vLLM OpenAI-compatible backend.")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8000", help="vLLM server base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key for vLLM server")
    parser.add_argument("--model", type=str, default="qwen3_1.7b_eagle3_1k_128k", help="served model name")
    parser.add_argument("--bench-name", type=str, default="longbench_ctx_sampled_1k_128k")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--answer-file", type=str, required=True)
    parser.add_argument("--max-new-token", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-input-chars", type=int, default=0, help="Optional char-level truncate (head+tail) for prompt content")
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    question_file = args.question_file or os.path.join(
        root_dir, "eagle", "data", args.bench_name, "question.jsonl"
    )

    questions = load_questions(question_file, args.question_begin, args.question_end)

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    endpoint = args.api_base.rstrip("/") + "/v1/chat/completions"

    for question in questions:
        messages = []
        turns = []
        wall_time = []
        new_tokens = []

        for turn in question["turns"]:
            user_turn = truncate_text(turn, args.max_input_chars)
            messages.append({"role": "user", "content": user_turn})

            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "max_tokens": args.max_new_token,
                "stream": False,
            }

            start = time.time()
            resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=args.request_timeout)
            elapsed = time.time() - start
            resp.raise_for_status()
            data = resp.json()

            answer = data["choices"][0]["message"]["content"].strip()
            completion_tokens = int(data.get("usage", {}).get("completion_tokens", 0))

            turns.append(answer)
            wall_time.append(elapsed)
            new_tokens.append(completion_tokens)

            messages.append({"role": "assistant", "content": answer})

        record = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": args.model,
            "choices": [
                {
                    "index": 0,
                    "turns": turns,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                }
            ],
            "tstamp": time.time(),
        }

        with open(args.answer_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(questions)} samples to {args.answer_file}")


if __name__ == "__main__":
    main()
