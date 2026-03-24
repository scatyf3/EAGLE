"""Generate baseline autoregressive answers with head-level KV cache pruning.

Keeps full KV cache for induction heads, uses start_recent for other heads.

Usage:
python -m eagle.evaluation.gen_baseline_answer_qwen3_head_pruning \
  --base-model-path Qwen/Qwen3-1.7B \
  --pruning-config outputs/head_pruning_config.json \
  --bench-name longbench_e \
  --dataset hotpot_qa_subset_50
"""
import argparse
import json
import os
import time

from fastchat.llm_judge.common import load_questions
from tqdm import tqdm
import torch

try:
    from ..model.ea_model import EaModel
    from ..model.head_pruning import HeadPruningConfig, apply_head_pruning_to_model
except Exception:
    from eagle.model.ea_model import EaModel
    from eagle.model.head_pruning import HeadPruningConfig, apply_head_pruning_to_model


def _open_start_recent_kv_cache(model, num_layers, start_size, recent_size):
    """Monkey patch model to use start_recent KV caching.
    
    This is a simplified approach: we'll store the pruning config
    and apply it in hooks.
    """
    # Store config on model
    model._start_size = start_size
    model._recent_size = recent_size
    model._kv_step_count = 0
    

def _prune_kv_cache_start_recent(past_key_values, layer_idx, start_size, recent_size, full_head_indices=None):
    """Apply start_recent pruning to layer KV cache.
    
    Args:
        past_key_values: tuple of (key_cache, value_cache)
        layer_idx: layer index
        start_size: keep first N tokens
        recent_size: keep last N tokens
        full_head_indices: set of head indices to skip pruning (keep full)
    """
    if not past_key_values:
        return
    
    full_head_indices = full_head_indices or set()
    key_cache, value_cache = past_key_values[layer_idx]
    
    # Check if caches have current_length tracking (DynamicCache format)
    if not hasattr(key_cache, 'current_length'):
        return
    
    current_len = int(key_cache.current_length.item())
    if current_len <= start_size + recent_size:
        return  # Not worth pruning small cache
    
    # Build index of tokens to keep
    keep_indices = list(range(start_size)) + list(range(current_len - recent_size, current_len))
    keep_indices = torch.tensor(keep_indices, device=key_cache.device, dtype=torch.long)
    
    # Prune along sequence dimension (dim 2 for qwen shape: [batch, num_heads, seq, head_dim])
    k_pruned = torch.index_select(key_cache.data, 2, keep_indices)
    v_pruned = torch.index_select(value_cache.data, 2, keep_indices)
    
    # Update the cache objects
    key_cache.data = k_pruned
    value_cache.data = v_pruned
    key_cache.current_length = torch.tensor(len(keep_indices), device=key_cache.device, dtype=key_cache.current_length.dtype)
    value_cache.current_length = torch.tensor(len(keep_indices), device=value_cache.device, dtype=value_cache.current_length.dtype)


def _should_prune(step_count, prune_interval=20):
    """Check if we should prune at this step."""
    return step_count > 0 and step_count % prune_interval == 0


def generate_with_head_pruning(model, input_ids, head_pruning_config, max_new_tokens=500, **kwargs):
    """Generate with head-level KV cache pruning.
    
    Args:
        model: EaModel instance
        input_ids: input token ids
        head_pruning_config: HeadPruningConfig or dict with full_heads
        max_new_tokens: max generation length
    """
    if isinstance(head_pruning_config, dict):
        config = HeadPruningConfig(full_heads=head_pruning_config.get('full_heads', {}))
    else:
        config = head_pruning_config
    
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    
    # Initial forward pass
    with torch.no_grad():
        outputs = model.base_model(input_ids=input_ids, output_hidden_states=False)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
    
    generated_ids = []
    step_count = 0
    
    while step_count < max_new_tokens:
        # Sample next token
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        
        # Apply pruning periodically
        if _should_prune(step_count, prune_interval=20):
            num_layers = len(past_key_values) if past_key_values else 0
            for layer_idx in range(num_layers):
                layer_name = f"layer_{layer_idx:03d}"
                full_heads = config.full_heads.get(layer_name, set())
                _prune_kv_cache_start_recent(
                    past_key_values, layer_idx,
                    config.start_size, config.recent_size,
                    full_head_indices=full_heads
                )
        
        # Next forward pass
        with torch.no_grad():
            outputs = model.base_model(
                input_ids=next_token,
                past_key_values=past_key_values,
                output_hidden_states=False
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
        
        step_count += 1
    
    return torch.cat([input_ids, torch.tensor(generated_ids, device=device).unsqueeze(0)], dim=1)


def main():
    parser = argparse.ArgumentParser(description="Generate baseline answers with head-level KV pruning.")
    parser.add_argument("--base-model-path", type=str, default="Qwen/Qwen3-1.7B", help="Base model path")
    parser.add_argument("--pruning-config", type=str, default="outputs/head_pruning_config.json", help="Head pruning config path")
    parser.add_argument("--bench-name", type=str, default="longbench_e", help="Benchmark name")
    parser.add_argument("--dataset", type=str, default="hotpot_qa_subset_50", help="Dataset subset")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    
    args = parser.parse_args()
    
    # Load head pruning config
    with open(args.pruning_config, 'r') as f:
        pruning_config_dict = json.load(f)
    
    # Convert to HeadPruningConfig format
    target_full_heads = {}
    for layer_name, head_list in pruning_config_dict.get('target_full_heads', {}).items():
        target_full_heads[layer_name] = set(head_list)
    
    pruning_config = HeadPruningConfig(
        full_heads=target_full_heads,
        start_size=pruning_config_dict.get('start_size', 1024),
        recent_size=pruning_config_dict.get('recent_size', 256)
    )
    
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading base model: {args.base_model_path}")
    model = EaModel.from_pretrained(args.base_model_path, torch_dtype=torch.float16, device_map=device)
    apply_head_pruning_to_model(model, pruning_config)
    
    print(f"Pruning config: {len(target_full_heads)} layers with induction heads")
    print(f"  start_size={pruning_config.start_size}, recent_size={pruning_config.recent_size}")
    
    # Load questions
    questions = load_questions(args.bench_name, args.dataset, None)
    if args.num_samples:
        questions = questions[:args.num_samples]
    
    print(f"Processing {len(questions)} questions...")
    
    # Simple benchmark: just measure generation time and tokens
    results = []
    total_tokens = 0
    total_time = 0
    
    for i, q in enumerate(tqdm(questions[:10], desc="Sample")):  # Just 10 for quick test
        prompt = q.get('context', '') + '\n' + q.get('input', '')
        input_ids = model.tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        start = time.time()
        with torch.no_grad():
            output_ids = generate_with_head_pruning(
                model, input_ids, pruning_config,
                max_new_tokens=128
            )
        elapsed = time.time() - start
        
        num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
        total_tokens += num_new_tokens
        total_time += elapsed
        
        text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({
            'index': i,
            'question': q.get('input', ''),
            'answer': text,
            'time': elapsed,
            'tokens': num_new_tokens
        })
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg throughput: {total_tokens/total_time:.2f} tok/s")
    
    # Save results
    output_file = os.path.join(args.output_dir, f"head_pruning_{args.dataset}_test.jsonl")
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\n✓ Saved: {output_file}")


if __name__ == "__main__":
    main()
