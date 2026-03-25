import argparse
import json
import os
import time
import subprocess
import signal
import psutil

import requests
import shortuuid
from fastchat.llm_judge.common import load_questions


def truncate_text(text, max_chars):
    if not max_chars or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + text[-half:]


def start_vllm(
    base_model_path: str,
    tensor_parallel: int = 3,
    pipeline_parallel: int = 1,
    max_model_len: int = None,
    port: int = 8000,
    api_key: str = "EMPTY",
    gpu_memory_utilization: float = 0.9,
) -> subprocess.Popen:
    """Start vLLM server with specified parameters."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model_path,
        "--tensor-parallel-size", str(tensor_parallel),
        "--pipeline-parallel-size", str(pipeline_parallel),
        "--port", str(port),
        "--api-key", api_key,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", "auto",
    ]
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def stop_vllm(process: subprocess.Popen) -> None:
    """Stop vLLM server gracefully."""
    if process is None:
        return
    
    print("Stopping vLLM server...")
    try:
        process.terminate()
        process.wait(timeout=10)
        print("vLLM server stopped gracefully.")
    except subprocess.TimeoutExpired:
        print("Force killing vLLM server...")
        process.kill()
        process.wait()


def wait_for_vllm(api_base: str, timeout: int = 300, check_interval: int = 2) -> bool:
    """Wait for vLLM server to be ready."""
    start_time = time.time()
    headers = {"Content-Type": "application/json"}
    endpoint = api_base.rstrip("/") + "/v1/models"
    
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(endpoint, headers=headers, timeout=5)
            if resp.status_code == 200:
                print("vLLM server is ready!")
                return True
        except Exception as e:
            print(f"Waiting for vLLM server... ({e})")
        
        time.sleep(check_interval)
    
    print(f"vLLM server did not become ready within {timeout} seconds")
    return False


def kill_vllm_by_port(port: int = 8000) -> bool:
    """Kill vLLM process by port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections(':'):
                    if conn.laddr.port == port:
                        print(f"Killing vLLM process (PID: {proc.pid}) on port {port}")
                        proc.kill()
                        proc.wait(timeout=5)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    except Exception as e:
        print(f"Error killing vLLM process: {e}")
    return False


def group_questions_by_context_length(questions: list[dict]) -> dict[int, list[dict]]:
    """Group questions by context (input token) length."""
    grouped = {}
    for q in questions:
        # Estimate context length from 'context' field
        context = q.get("context", "")
        context_len = len(context.split())  # Rough estimate using word count
        # Round to nearest 1000 tokens for grouping
        context_bucket = (context_len // 1000) * 1000
        if context_bucket not in grouped:
            grouped[context_bucket] = []
        grouped[context_bucket].append(q)
    return dict(sorted(grouped.items()))


def main():
    parser = argparse.ArgumentParser(description="Run LongBench with vLLM OpenAI-compatible backend.")
    # vLLM parameters
    parser.add_argument("--base-model-path", type=str, required=True, help="Base model path for vLLM")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--tensor-parallel", type=int, default=3, help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model length for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization ratio")
    parser.add_argument("--auto-restart-vllm", action="store_true", help="Auto restart vLLM after each context length group")
    parser.add_argument("--vllm-timeout", type=int, default=300, help="Timeout for waiting vLLM to be ready")
    
    # Query parameters
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
    
    # Group questions by context length
    grouped_questions = group_questions_by_context_length(questions)
    print(f"Grouped {len(questions)} questions into {len(grouped_questions)} context length buckets:")
    for ctx_len, qs in grouped_questions.items():
        print(f"  Context ~{ctx_len} tokens: {len(qs)} questions")

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    endpoint = args.api_base.rstrip("/") + "/v1/chat/completions"
    
    vllm_process = None
    try:
        # Start vLLM server if auto-restart is enabled
        if args.auto_restart_vllm:
            vllm_process = start_vllm(
                base_model_path=args.base_model_path,
                tensor_parallel=args.tensor_parallel,
                pipeline_parallel=args.pipeline_parallel,
                max_model_len=args.max_model_len,
                port=args.vllm_port,
                api_key=args.api_key,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            time.sleep(2)  # Give server time to start
            if not wait_for_vllm(args.api_base, timeout=args.vllm_timeout):
                raise RuntimeError("Failed to start vLLM server")
        
        # Process each context length group
        for ctx_len, group_questions in grouped_questions.items():
            print(f"\n{'='*60}")
            print(f"Processing context length group: ~{ctx_len} tokens ({len(group_questions)} questions)")
            print(f"{'='*60}")
            
            for question in group_questions:
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
            
            # Restart vLLM after each context length group
            if args.auto_restart_vllm:
                print(f"Finished context group ~{ctx_len} tokens. Restarting vLLM server...")
                stop_vllm(vllm_process)
                time.sleep(2)
                vllm_process = start_vllm(
                    base_model_path=args.base_model_path,
                    tensor_parallel=args.tensor_parallel,
                    pipeline_parallel=args.pipeline_parallel,
                    max_model_len=args.max_model_len,
                    port=args.vllm_port,
                    api_key=args.api_key,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
                time.sleep(2)
                if not wait_for_vllm(args.api_base, timeout=args.vllm_timeout):
                    raise RuntimeError("Failed to restart vLLM server")

    finally:
        # Clean up vLLM process
        if vllm_process is not None:
            stop_vllm(vllm_process)

    print(f"\nDone. Wrote {len(questions)} samples to {args.answer_file}")



if __name__ == "__main__":
    main()
