"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *


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


def _save_draft_attn_debug_snapshot(
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
    if not args.draft_attn_debug:
        return

    model_part = _sanitize_filename(args.model_id)
    bench_part = _sanitize_filename(args.bench_name)
    qid_part = _sanitize_filename(question_id)
    out_dir = os.path.join(args.draft_attn_debug_dir, bench_part, model_part, qid_part)
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
        "draft_accepted_attn_trace": getattr(model, "last_draft_attn_trace", None),
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


def parse_head_list(head_csv):
    if not head_csv:
        return None
    out = []
    for p in str(head_csv).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out if out else None


def select_random_heads(num_heads, count, seed):
    """Select `count` unique random attention heads from [0, num_heads)."""
    if count is None or int(count) <= 0:
        return None
    n = int(num_heads)
    k = min(int(count), n)
    if k <= 0 or n <= 0:
        return None
    rng = random.Random(int(seed))
    heads = sorted(rng.sample(list(range(n)), k))
    return heads if heads else None


def _get_linear_timing_model(model, component="target"):
    if component == "target":
        candidate = getattr(model, "base_model", None)
    elif component == "draft":
        candidate = getattr(model, "ea_layer", None)
    else:
        candidate = None

    if candidate is None:
        return None
    if hasattr(candidate, "reset_linear_timing_stats") and hasattr(candidate, "get_linear_timing_stats"):
        return candidate
    return None


def _reset_linear_timing_stats(model):
    status = {}
    for comp in ("target", "draft"):
        timing_model = _get_linear_timing_model(model, component=comp)
        if timing_model is None:
            status[comp] = False
            continue
        try:
            timing_model.reset_linear_timing_stats()
            status[comp] = True
        except Exception as e:
            print(f"Warning: failed to reset {comp} linear timing stats: {e}")
            status[comp] = False
    return status


def _get_linear_timing_stats(model, component="target"):
    timing_model = _get_linear_timing_model(model, component=component)
    if timing_model is None:
        return None
    try:
        raw = timing_model.get_linear_timing_stats()
        return {
            "self_attn_s": float(raw.get("self_attn_s", 0.0)),
            "other_linear_s": float(raw.get("other_linear_s", 0.0)),
            "self_attn_ops": int(raw.get("self_attn_ops", 0)),
            "other_linear_ops": int(raw.get("other_linear_ops", 0)),
            "steps": int(raw.get("steps", 0)),
        }
    except Exception as e:
        print(f"Warning: failed to read {component} linear timing stats: {e}")
        return None


def _collect_linear_timing_summary(model, enabled):
    out = {}
    for comp in ("target", "draft"):
        stats = _get_linear_timing_stats(model, component=comp)
        if stats is None:
            out[comp] = {
                "enabled": bool(enabled),
                "available": False,
            }
            continue
        comp_out = {
            "enabled": bool(enabled),
            "available": True,
        }
        comp_out.update(stats)
        out[comp] = comp_out
    return out


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
                    seq_dim = 2
                    used_len = max(0, min(cur_len, int(data.shape[seq_dim])))
                    used_numel = int(data.shape[0] * data.shape[1] * used_len * data.shape[3])
                else:
                    used_numel = int(data.numel())
                used_bytes = int(used_numel * data.element_size())
                used[dev] = used.get(dev, 0) + used_bytes
    except Exception:
        # Keep profiling best-effort and never fail generation.
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
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
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
        args
):
    # temperature = 0.0

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
        use_eagle3=args.use_eagle3,
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)
    os.environ["EAGLE_RECORD_LINEAR_TIME"] = "1" if args.record_linear_time else "0"
    print(f"EAGLE_RECORD_LINEAR_TIME={os.environ['EAGLE_RECORD_LINEAR_TIME']}")

    selected_streaming_heads = args.streaming_protected_draft_heads
    if args.streaming_random_draft_head_count > 0:
        try:
            # Support both draft backbones:
            # - cnets1.Model: ea_layer.layers[0].self_attn.num_heads
            # - cnets.Model:  ea_layer.midlayer.self_attn.num_heads
            num_draft_heads = None
            if hasattr(model.ea_layer, "layers") and len(model.ea_layer.layers) > 0:
                num_draft_heads = int(model.ea_layer.layers[0].self_attn.num_heads)
            elif hasattr(model.ea_layer, "midlayer") and hasattr(model.ea_layer.midlayer, "self_attn"):
                num_draft_heads = int(model.ea_layer.midlayer.self_attn.num_heads)
            elif hasattr(model.ea_layer, "config") and hasattr(model.ea_layer.config, "num_attention_heads"):
                num_draft_heads = int(model.ea_layer.config.num_attention_heads)
            else:
                raise AttributeError("Cannot infer draft num_heads from ea_layer")

            selected_streaming_heads = select_random_heads(
                num_heads=num_draft_heads,
                count=args.streaming_random_draft_head_count,
                seed=args.streaming_random_seed,
            )
            print(
                f"Streaming random protected draft heads enabled: "
                f"count={args.streaming_random_draft_head_count}, "
                f"seed={args.streaming_random_seed}, "
                f"selected={selected_streaming_heads}"
            )
        except Exception as e:
            print(f"Warning: failed to select random streaming protected heads: {e}")
            selected_streaming_heads = args.streaming_protected_draft_heads

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

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

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, _pf = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                max_length=args.max_length,
                max_new_tokens=max_new_token,
                streaming_kv_prune=args.streaming_kv_prune,
                streaming_sink_size=args.streaming_sink_size,
                streaming_window_size=args.streaming_window_size,
                streaming_full_draft_heads=selected_streaming_heads,
                draft_attn_debug=args.draft_attn_debug,
                h2o_kv_prune=args.h2o_kv_prune,
                h2o_heavy_budget=args.h2o_heavy_budget,
                h2o_recent_budget=args.h2o_recent_budget,
                log=True
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
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
    print('Warmup done')
    if args.record_linear_time:
        _reset_linear_timing_stats(model)

    # questions=questions[6:]
    for question in tqdm(questions):
        if args.record_linear_time:
            _reset_linear_timing_stats(model)

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("qwen3")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            prefill_times = []
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
                output_ids, new_token, idx, sample_prefill_time = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    max_length=args.max_length,
                    max_new_tokens=max_new_token,
                    streaming_kv_prune=args.streaming_kv_prune,
                    streaming_sink_size=args.streaming_sink_size,
                    streaming_window_size=args.streaming_window_size,
                    streaming_full_draft_heads=selected_streaming_heads,
                    draft_attn_debug=args.draft_attn_debug,
                    h2o_kv_prune=args.h2o_kv_prune,
                    h2o_heavy_budget=args.h2o_heavy_budget,
                    h2o_recent_budget=args.h2o_recent_budget,
                    log=True
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]

                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
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
                if args.draft_attn_debug:
                    _save_draft_attn_debug_snapshot(
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
                prefill_times.append(sample_prefill_time)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            memory_profile = _collect_memory_profile(model, args.record_memory_profile)
            choices.append({
                "index": i,
                "turns": turns,
                "idxs": idxs,
                "new_tokens": new_tokens,
                "wall_time": wall_time,
                "prefill_time": prefill_times,
                "memory_profile": memory_profile,
            })

        sample_linear_timing = _collect_linear_timing_summary(model, args.record_linear_time)

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "linear_timing": sample_linear_timing,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
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
        default="/workspace/yunhai/Qwen3-4B_eagle3",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="Qwen/Qwen3-4B",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="auto",
        help="Output model id in jsonl. Use 'auto' to derive from base/draft paths.",
    )
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
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum total sequence length used by eagenerate.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=12000,
        help="If prompt token length exceeds this value, keep head and tail halves only.",
    )
    parser.add_argument(
        "--streaming-kv-prune",
        action="store_true",
        help="Enable StreamingLLM-style KV pruning after prefill.",
    )
    parser.add_argument(
        "--streaming-sink-size",
        type=int,
        default=4,
        help="Number of sink tokens kept from sequence start when pruning KV.",
    )
    parser.add_argument(
        "--streaming-window-size",
        type=int,
        default=2044,
        help="Number of most recent tokens kept when pruning KV. With default sink=4, total kept tokens=2048.",
    )
    parser.add_argument(
        "--streaming-full-draft-heads",
        type=str,
        default="",
        help="[Deprecated alias] Use --streaming-protected-draft-heads. Comma-separated draft head ids kept full (protected) during streaming prune.",
    )
    parser.add_argument(
        "--streaming-protected-draft-heads",
        type=str,
        default="",
        help="Preferred name. Comma-separated draft head ids protected from streaming prune (global-like heads), e.g. '2,3,9'.",
    )
    parser.add_argument(
        "--streaming-random-draft-head-count",
        type=int,
        default=0,
        help="If > 0, randomly select this many protected draft heads (global-like) during streaming prune.",
    )
    parser.add_argument(
        "--streaming-random-seed",
        type=int,
        default=0,
        help="Random seed used when selecting random draft heads.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="The maximum number of new generated tokens.",
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
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument(
        "--use-eagle3",
        action="store_true"
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
        "--draft-attn-debug",
        action="store_true",
        help="Save draft-model attention traces aligned to accepted tokens.",
    )
    parser.add_argument(
        "--draft-attn-debug-dir",
        type=str,
        default=f"{parent_dir}/outputs/draft_attn_debug",
        help="Directory to save draft attention debug snapshots.",
    )
    parser.add_argument(
        "--h2o-kv-prune",
        action="store_true",
        help="Enable H2O KV pruning on the draft model (heavy-hitter + recent window).",
    )
    parser.add_argument(
        "--h2o-heavy-budget",
        type=int,
        default=200,
        help="Number of heavy-hitter tokens to keep when H2O pruning is active.",
    )
    parser.add_argument(
        "--h2o-recent-budget",
        type=int,
        default=2000,
        help="Number of most-recent tokens to keep when H2O pruning is active.",
    )
    parser.add_argument(
        "--record-linear-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record qkv/other linear projection timing and write per-sample stats into output jsonl.",
    )
    parser.add_argument(
        "--record-memory-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record GPU memory profile (weights/KV cache/activation estimate) at the end of each choice inference.",
    )

    args = parser.parse_args()
    old_heads = parse_head_list(args.streaming_full_draft_heads)
    new_heads = parse_head_list(args.streaming_protected_draft_heads)
    # Prefer the clearer protected-head name when both are provided.
    args.streaming_protected_draft_heads = new_heads if new_heads is not None else old_heads
    # Keep legacy field for backward compatibility in downstream calls.
    args.streaming_full_draft_heads = args.streaming_protected_draft_heads

    for k,v in vars(args).items():
        print(f"{k}={v}")

    if str(args.model_id).lower() == "auto":
        base_name = os.path.basename(str(args.base_model_path).rstrip("/"))
        if not base_name:
            base_name = "base-model"
        suffix = "eagle3" if args.use_eagle3 else "eagle"
        args.model_id = f"{base_name}-{suffix}"
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
        args
    )

    reorg_answer_file(answer_file)
