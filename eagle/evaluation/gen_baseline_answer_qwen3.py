"""Generate baseline autoregressive answers with Qwen3 models.

Usage:
python -m eagle.evaluation.gen_baseline_answer_qwen3 --base-model-path Qwen/Qwen3-1.7B --ea-model-path AngelSlim/Qwen3-1.7B_eagle3 --bench-name mt_bench
"""
import argparse
import json
import os
import time
import sys
import traceback

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.utils import *
except Exception:
    from eagle.model.ea_model import EaModel
    from eagle.model.utils import *


script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)


def _sanitize_filename(name):
    return str(name).replace("/", "_").replace(" ", "_")


def _collect_stable_kv(model):
    stable_kv = {}
    if not hasattr(model, "past_key_values"):
        return stable_kv

    for layer_idx, layer_kv in enumerate(model.past_key_values):
        key_cache, value_cache = layer_kv
        seq_len = int(min(key_cache.current_length.item(), value_cache.current_length.item()))
        key_tensor = key_cache.data[0, :, :seq_len, :].detach().cpu()
        value_tensor = value_cache.data[0, :, :seq_len, :].detach().cpu()
        stable_kv[f"layer_{layer_idx:03d}"] = {
            "key": key_tensor,
            "value": value_tensor,
        }
    return stable_kv


def _save_kv_debug_snapshot(
    model,
    args,
    question_id,
    choice_idx,
    turn_idx,
    prompt,
    prompt_token_ids,
    generated_text,
    generated_token_ids,
):
    if not args.kv_debug:
        return

    model_part = _sanitize_filename(args.model_id)
    bench_part = _sanitize_filename(args.bench_name)
    qid_part = _sanitize_filename(question_id)
    out_dir = os.path.join(args.kv_debug_dir, bench_part, model_part, qid_part)
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "meta": {
            "bench_name": args.bench_name,
            "model_id": args.model_id,
            "question_id": question_id,
            "choice_idx": int(choice_idx),
            "turn_idx": int(turn_idx),
        },
        "prompt_text": prompt,
        "generated_text": generated_text,
        "prompt_token_ids": [int(x) for x in prompt_token_ids],
        "generated_token_ids": [int(x) for x in generated_token_ids],
        "stable_kv": _collect_stable_kv(model),
    }

    out_name = f"choice_{choice_idx:02d}_turn_{turn_idx:02d}.pt"
    torch.save(payload, os.path.join(out_dir, out_name))


def _save_attn_debug_snapshot(
    model,
    args,
    question_id,
    choice_idx,
    turn_idx,
    prompt,
    prompt_token_ids,
    generated_text,
    generated_token_ids,
):
    if not args.attn_debug:
        return

    model_part = _sanitize_filename(args.model_id)
    bench_part = _sanitize_filename(args.bench_name)
    qid_part = _sanitize_filename(question_id)
    out_dir = os.path.join(args.attn_debug_dir, bench_part, model_part, qid_part)
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "meta": {
            "bench_name": args.bench_name,
            "model_id": args.model_id,
            "question_id": question_id,
            "choice_idx": int(choice_idx),
            "turn_idx": int(turn_idx),
        },
        "prompt_text": prompt,
        "generated_text": generated_text,
        "prompt_token_ids": [int(x) for x in prompt_token_ids],
        "generated_token_ids": [int(x) for x in generated_token_ids],
        "accepted_token_attn_trace": getattr(model, "last_attn_trace", None),
    }

    out_name = f"choice_{choice_idx:02d}_turn_{turn_idx:02d}.pt"
    torch.save(payload, os.path.join(out_dir, out_name))


def truncate_input_ids(input_ids, max_input_tokens):
    if not max_input_tokens:
        return input_ids
    if len(input_ids[0]) <= max_input_tokens:
        return input_ids
    half = max_input_tokens // 2
    input_ids[0] = input_ids[0][:half] + input_ids[0][-half:]
    return input_ids


def _reset_cuda_memory_peak_stats(enabled=True):
    if not enabled or not torch.cuda.is_available():
        return
    for dev_idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(dev_idx)
        except Exception:
            pass


def _collect_weight_bytes_by_device(model):
    out = {}

    for p in model.parameters():
        if not p.is_cuda:
            continue
        dev = str(p.device)
        out[dev] = out.get(dev, 0) + int(p.numel() * p.element_size())

    for b in model.buffers():
        if not b.is_cuda:
            continue
        dev = str(b.device)
        out[dev] = out.get(dev, 0) + int(b.numel() * b.element_size())

    return out


def _collect_kv_bytes_by_device(model):
    used = {}
    capacity = {}

    if not hasattr(model, "past_key_values"):
        return {"used": used, "capacity": capacity}

    try:
        for layer_kv in model.past_key_values:
            key_cache, value_cache = layer_kv
            for cache in (key_cache, value_cache):
                data = cache.data
                if not data.is_cuda:
                    continue

                dev = str(data.device)
                cap_bytes = int(data.numel() * data.element_size())
                capacity[dev] = capacity.get(dev, 0) + cap_bytes

                cur_len = int(cache.current_length.item()) if hasattr(cache, "current_length") else data.shape[2]
                if data.dim() >= 4:
                    used_len = max(0, min(cur_len, int(data.shape[2])))
                    used_numel = int(data.shape[0] * data.shape[1] * used_len * data.shape[3])
                else:
                    used_numel = int(data.numel())
                used_bytes = int(used_numel * data.element_size())
                used[dev] = used.get(dev, 0) + used_bytes
    except Exception:
        return {"used": used, "capacity": capacity}

    return {"used": used, "capacity": capacity}


def _collect_cuda_peak_by_device():
    out = {}
    if not torch.cuda.is_available():
        return out

    for dev_idx in range(torch.cuda.device_count()):
        dev = f"cuda:{dev_idx}"
        try:
            out[dev] = {
                "current_allocated": int(torch.cuda.memory_allocated(dev_idx)),
                "peak_allocated": int(torch.cuda.max_memory_allocated(dev_idx)),
                "current_reserved": int(torch.cuda.memory_reserved(dev_idx)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(dev_idx)),
            }
        except Exception:
            out[dev] = {
                "current_allocated": 0,
                "peak_allocated": 0,
                "current_reserved": 0,
                "peak_reserved": 0,
            }
    return out


def _collect_memory_profile(model, enabled=True):
    if not enabled:
        return {"enabled": False}

    if not torch.cuda.is_available():
        return {
            "enabled": True,
            "available": False,
            "reason": "cuda_not_available",
        }

    weight_by_dev = _collect_weight_bytes_by_device(model)
    kv_by_dev = _collect_kv_bytes_by_device(model)
    peak_by_dev = _collect_cuda_peak_by_device()

    devices = sorted(set(list(weight_by_dev.keys()) + list(kv_by_dev["used"].keys()) + list(peak_by_dev.keys())))
    per_device = {}

    total_weight = 0
    total_kv_used = 0
    total_kv_capacity = 0
    total_peak_allocated = 0
    total_peak_reserved = 0
    total_activation_peak_est = 0

    for dev in devices:
        w = int(weight_by_dev.get(dev, 0))
        kv_u = int(kv_by_dev["used"].get(dev, 0))
        kv_c = int(kv_by_dev["capacity"].get(dev, 0))
        peak = peak_by_dev.get(dev, {})
        p_alloc = int(peak.get("peak_allocated", 0))
        p_res = int(peak.get("peak_reserved", 0))
        act_est = max(0, p_alloc - w - kv_u)

        per_device[dev] = {
            "weight_bytes": w,
            "kv_cache_used_bytes": kv_u,
            "kv_cache_capacity_bytes": kv_c,
            "activation_peak_est_bytes": int(act_est),
            "peak_allocated_bytes": p_alloc,
            "peak_reserved_bytes": p_res,
            "current_allocated_bytes": int(peak.get("current_allocated", 0)),
            "current_reserved_bytes": int(peak.get("current_reserved", 0)),
        }

        total_weight += w
        total_kv_used += kv_u
        total_kv_capacity += kv_c
        total_peak_allocated += p_alloc
        total_peak_reserved += p_res
        total_activation_peak_est += int(act_est)

    return {
        "enabled": True,
        "available": True,
        "totals": {
            "weight_bytes": int(total_weight),
            "kv_cache_used_bytes": int(total_kv_used),
            "kv_cache_capacity_bytes": int(total_kv_capacity),
            "activation_peak_est_bytes": int(total_activation_peak_est),
            "peak_allocated_bytes": int(total_peak_allocated),
            "peak_reserved_bytes": int(total_peak_reserved),
        },
        "per_device": per_device,
    }


def run_eval(
    base_model_path,
    ea_model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    args,
):
    questions = load_questions(question_file, question_begin, question_end)

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    base_model_path,
    ea_model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    args,
):
    try:
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_eagle3=args.use_eagle3,
        )

        tokenizer = model.get_tokenizer()
        model.eval()
        print("Check model training state:", model.training)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"ERROR: GPU Out of Memory during model loading!", file=sys.stderr)
            print(f"Details: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        else:
            raise

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("CUDA VISIBLE DEVICES:", cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        conv = get_conversation_template("qwen3")
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids
            input_ids = truncate_input_ids(input_ids, args.max_input_tokens)

            torch.cuda.synchronize()
            start_time = time.time()
            try:
                output_ids, new_token, idx = model.naivegenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    max_length=args.max_length,
                    max_new_tokens=max_new_token,
                    log=True,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"ERROR: GPU Out of Memory during inference (warmup)!", file=sys.stderr)
                    print(f"Input shape: {input_ids.shape}, Length: {len(input_ids[0])}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    sys.exit(1)
                else:
                    raise
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]) :]

            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i for i, id in enumerate(output_ids) if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            conv.stop_str = "</s>"
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print("Warmup done")

    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("qwen3")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            _reset_cuda_memory_peak_stats(args.record_memory_profile)
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids
                input_ids = truncate_input_ids(input_ids, args.max_input_tokens)

                torch.cuda.synchronize()
                start_time = time.time()
                try:
                    output_ids, new_token, idx = model.naivegenerate(
                        torch.as_tensor(input_ids).cuda(),
                        temperature=temperature,
                        max_length=args.max_length,
                        max_new_tokens=max_new_token,
                        log=True,
                        attn_debug=args.attn_debug,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"ERROR: GPU Out of Memory during inference!", file=sys.stderr)
                        print(f"Question ID: {question['question_id']}, Choice: {i}, Turn: {j}", file=sys.stderr)
                        print(f"Input shape: {input_ids.shape}, Max new tokens: {max_new_token}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        sys.exit(1)
                    else:
                        raise
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]) :]

                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i for i, id in enumerate(output_ids) if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()

                if args.kv_debug:
                    _save_kv_debug_snapshot(
                        model=model,
                        args=args,
                        question_id=question["question_id"],
                        choice_idx=i,
                        turn_idx=j,
                        prompt=prompt,
                        prompt_token_ids=input_ids[0],
                        generated_text=output,
                        generated_token_ids=output_ids.tolist(),
                    )
                if args.attn_debug:
                    _save_attn_debug_snapshot(
                        model=model,
                        args=args,
                        question_id=question["question_id"],
                        choice_idx=i,
                        turn_idx=j,
                        prompt=prompt,
                        prompt_token_ids=input_ids[0],
                        generated_text=output,
                        generated_token_ids=output_ids.tolist(),
                    )

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output

            memory_profile = _collect_memory_profile(model, args.record_memory_profile)

            choices.append(
                {
                    "index": i,
                    "turns": turns,
                    "idxs": idxs,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "memory_profile": memory_profile,
                }
            )

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication."""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="AngelSlim/Qwen3-1.7B_eagle3",
        help="Path to EAGLE weights. Required by EaModel wrapper.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Base model path or HF model id.",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="qwen3-1.7b-baseline")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=32,
        help="Needed by EaModel wrapper; not used by naive AR generation quality.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum total sequence length used by naivegenerate.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=12000,
        help="If prompt token length exceeds this value, keep head and tail halves only.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Needed by EaModel wrapper.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Needed by EaModel wrapper.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maximum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--use-eagle3",
        action="store_true",
    )
    parser.add_argument(
        "--kv-debug",
        action="store_true",
        help="Save stable KV snapshots per question turn (prompt/text/tokens + per-layer per-head KV).",
    )
    parser.add_argument(
        "--kv-debug-dir",
        type=str,
        default=f"{parent_dir}/outputs/kv_debug",
        help="Directory to save KV debug snapshots.",
    )
    parser.add_argument(
        "--attn-debug",
        action="store_true",
        help="Save per-step accepted-token attention scores to previous tokens.",
    )
    parser.add_argument(
        "--attn-debug-dir",
        type=str,
        default=f"{parent_dir}/outputs/attn_debug",
        help="Directory to save attention debug snapshots.",
    )
    parser.add_argument(
        "--record-memory-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record GPU memory profile (weights/KV cache/activation estimate) at the end of each choice inference.",
    )

    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}={v}")

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    try:
        run_eval(
            args.base_model_path,
            args.ea_model_path,
            args.model_id,
            question_file,
            args.question_begin,
            args.question_end,
            answer_file,
            args.max_new_token,
            args.num_choices,
            args.num_gpus_per_model,
            args.num_gpus_total,
            args.max_gpu_memory,
            args.temperature,
            args,
        )

        reorg_answer_file(answer_file)
    except MemoryError as e:
        print(f"ERROR: System Out of Memory!", file=sys.stderr)
        print(f"Details: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"ERROR: GPU Out of Memory!", file=sys.stderr)
            print(f"Details: {str(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        else:
            raise
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
