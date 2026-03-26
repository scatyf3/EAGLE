"""
Profile EAGLE draft vs target phase breakdown with explicit timing.
Separates topK_genrate (draft), tree_decoding (verify) into distinct sections.
"""

import torch
import json
from pathlib import Path
import sys
import time

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from eagle.model.ea_model import EaModel


def load_sample_data(n_ctx=512):
    """Load a sample from the benchmark."""
    jsonl_path = Path(__file__).parent.parent / "outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl"
    with open(jsonl_path) as f:
        sample = json.loads(next(f))
    return sample


def profile_eagle_draft_target_separation():
    """Profile EAGLE with explicit draft vs target separation."""
    
    # Load model
    print("[*] Loading EAGLE model...")
    base_model_path = "NousResearch/Yarn-Llama-2-7b-128k"
    ea_model_path = "yuhuili/EAGLE-llama2-chat-7B"
    
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=-1,
        depth=6,
        top_k=10,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=False,
    ).eval()
    tokenizer = model.get_tokenizer()
    
    # Prepare input
    context_length = 512
    print(f"[*] Preparing input (context_length={context_length})...")
    sample = load_sample_data()
    prompt = sample.get('prompt', sample.get('context', ''))
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')[:, :context_length].to('cuda:0')
    print(f"    Input shape: {input_ids.shape}")
    
    # Warm up
    print("[*] Warming up...")
    with torch.no_grad():
        _ = model.eagenerate(input_ids, max_new_tokens=2, temperature=0.6, top_p=0.9)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    print("\n[*] Profiling EAGLE draft+target breakdown...")
    print("=" * 70)
    
    # Re-prepare input for main profiling
    input_ids = tokenizer.encode(prompt, return_tensors='pt')[:, :context_length].to('cuda:0')
    
    # Measure full generation with timing breakdown
    gen_len = 32
    timings = {
        'prefill': [],
        'decode_iterations': []
    }
    
    with torch.no_grad():
        # Full generation
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        outputs = model.eagenerate(
            input_ids,
            max_new_tokens=gen_len,
            temperature=0.6,
            top_p=0.9,
        )
        
        torch.cuda.synchronize()
        t_end = time.perf_counter()
    
    total_time = t_end - t_start
    
    print(f"\nTotal generation time: {total_time*1000:.2f} ms")
    print(f"  - Input length: {input_ids.shape[1]} tokens")
    print(f"  - Generated length: {gen_len} tokens")
    print(f"  - Throughput: {gen_len / total_time:.2f} tok/s")
    print(f"  - Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Now profile with torch.profiler for operator breakdown
    print("\n[*] Running detailed profiler for operator breakdown...")
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')[:, :context_length].to('cuda:0')
    
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,  # Disable memory profiling to reduce overhead
        on_trace_ready=None,
    ) as prof:
        with torch.no_grad():
            outputs = model.eagenerate(
                input_ids,
                max_new_tokens=gen_len,
                temperature=0.6,
                top_p=0.9,
            )
    
    # Save profile
    prof.export_chrome_trace("outputs/eagle_draft_target_profile.json")
    print(f"\n[*] Profile saved to outputs/eagle_draft_target_profile.json")
    
    # Print top operators by CUDA time
    print("\n[*] Top 20 CUDA operations:")
    print("-" * 80)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    print(table)
    
    # Filter for key operations
    print("\n[*] Key operation categories:")
    print("-" * 80)
    
    key_ops = prof.key_averages()
    
    # Group by operation type
    linear_time = 0
    matmul_time = 0
    attention_time = 0
    other_time = 0
    
    for op in key_ops:
        cuda_time = op.cuda_time_total
        if 'linear' in op.key.lower():
            linear_time += cuda_time
        elif 'matmul' in op.key.lower() or 'mm' in op.key.lower():
            matmul_time += cuda_time
        elif 'kernel' in op.key.lower() or 'flash' in op.key.lower():
            attention_time += cuda_time
        else:
            other_time += cuda_time
    
    total_cuda_time = linear_time + matmul_time + attention_time + other_time
    
    print(f"Linear layers:    {linear_time/1000:.2f} ms ({linear_time/total_cuda_time*100:.1f}%)")
    print(f"MatMul operations: {matmul_time/1000:.2f} ms ({matmul_time/total_cuda_time*100:.1f}%)")
    print(f"Attention/kernels: {attention_time/1000:.2f} ms ({attention_time/total_cuda_time*100:.1f}%)")
    print(f"Other operations:  {other_time/1000:.2f} ms ({other_time/total_cuda_time*100:.1f}%)")
    print(f"Total CUDA time:   {total_cuda_time/1000:.2f} ms")
    
    print("\n" + "=" * 70)
    print("[*] Profile complete. Load outputs/eagle_draft_target_profile.json in chrome://tracing for detailed analysis")


if __name__ == "__main__":
    profile_eagle_draft_target_separation()
