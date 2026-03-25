"""
Profile a single EAGLE draft+target iteration with context_length=3000.
Shows breakdown of topK_genrate, tree_decoding, evaluate_posterior, etc.
"""

import torch
import json
from pathlib import Path
import sys

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from eagle.model.ea_model import EaModel


def load_sample_data(n_ctx=3000):
    """Load a sample from the benchmark."""
    jsonl_path = Path(__file__).parent.parent / "outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl"
    with open(jsonl_path) as f:
        sample = json.loads(next(f))
    return sample


def profile_eagle_iteration():
    """Profile one EAGLE iteration (draft tree + target verify)."""
    
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
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True, legacy=False)
    
    # Prepare input
    context_length = 512  # Very conservative to avoid CUDA issues
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
    
    # Profile
    print("[*] Profiling one generate iteration...")
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=None,  # We'll save manually
    ) as prof:
        with torch.no_grad():
            # Reset cache
            input_ids = tokenizer.encode(prompt, return_tensors='pt')[:, :context_length].to('cuda:0')
            
            # Full generation to see pattern
            outputs = model.eagenerate(
                input_ids,
                max_new_tokens=32,
                temperature=0.6,
                top_p=0.9,
            )
    
    torch.cuda.synchronize()
    
    # Save profile
    prof.export_chrome_trace("outputs/eagle_profile.json")
    print("\n[*] Profile saved to outputs/eagle_profile.json")
    
    # Print summary
    print("\n[*] Top 10 CPU operators by self time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    print("\n[*] Top 10 CUDA operators by self time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\n[*] Memory stats:")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    
    # Memory usage
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n[*] Peak GPU memory: {peak_mem:.2f} GB")


if __name__ == "__main__":
    profile_eagle_iteration()
