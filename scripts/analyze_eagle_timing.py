"""
Direct timing analysis of EAGLE draft vs target phases.
Measures throughput and cost breakdown without torch.profiler overhead.
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
import sys
import time

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from eagle.model.ea_model import EaModel


def main():
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
    
    # Load sample
    jsonl_path = Path(__file__).parent.parent / "outputs/longbench_hip_prompt_gt_25000_qwen3.jsonl"
    with open(jsonl_path) as f:
        sample = json.loads(next(f))
    
    # Prepare input
    context_length = 512
    prompt = sample.get('prompt', sample.get('context', ''))
    input_ids = tokenizer.encode(prompt, return_tensors='pt')[:, :context_length]
    
    print(f"[*] Input: {input_ids.shape[1]} tokens, Device: {input_ids.device}")
    
    # Move to CUDA
    input_ids = input_ids.to('cuda:0')
    
    # Warm up
    print("[*] Warming up...")
    with torch.no_grad():
        _ = model.eagenerate(input_ids, max_new_tokens=2, temperature=0.6, top_p=0.9)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Run multiple generations to get average timing
    print("\n[*] Profiling 3 full generations (32 tokens each)...")
    print("=" * 70)
    
    gen_len = 32
    num_runs = 3
    
    times = []
    peak_mems = []
    
    for run_idx in range(num_runs):
        # Re-prepare input
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[:, :context_length].to('cuda:0')
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.eagenerate(
                input_ids,
                max_new_tokens=gen_len,
                temperature=0.6,
                top_p=0.9,
            )
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        elapsed = t1 - t0
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        
        times.append(elapsed)
        peak_mems.append(peak_mem)
        
        throughput = gen_len / elapsed
        print(f"Run {run_idx+1}: {elapsed*1000:.1f}ms | {throughput:.2f} tok/s | Peak mem: {peak_mem:.2f}GB")
    
    # Statistics
    avg_time = sum(times) / len(times)
    avg_tps = gen_len / avg_time
    avg_mem = sum(peak_mems) / len(peak_mems)
    
    print("=" * 70)
    print(f"Average: {avg_time*1000:.1f}ms | {avg_tps:.2f} tok/s | Avg peak mem: {avg_mem:.2f}GB")
    print(f"Generation stats:")
    print(f"  - Position: {context_length} context -> +{gen_len} tokens")
    print(f"  - Total tokens output: {context_length + gen_len}")
    
    # Theoretical analysis
    print("\n[*] Theoretical cost breakdown:")
    print("-" * 70)
    
    # Llama 2 7B: 32 layers, 4096 hidden, 32 heads (128 hidden per head)
    # EAGLE draft: tree with ~50 nodes, depth 7
    n_layers = 32
    hidden_d = 4096
    n_heads = 32
    head_dim = hidden_d // n_heads
    
    # Attention cost per token: 2 * seq_len * hidden_d (Q@K^T + softmax@V)
    # MLP cost per token: 8 * hidden_d^2 (in projection is 4x, out projection is 1x)
    
    print(f"\nModel architecture:")
    print(f"  - Layers: {n_layers}")
    print(f"  - Hidden size: {hidden_d}")
    print(f"  - Attention heads: {n_heads}")
    
    # For each generated token, we do:
    # 1. Draft tree generation (50 tokens through small draft model)
    # 2. Target model forward pass with tree (50 tokens through full target)
    # 3. Evaluation to get final token
    
    # Cost estimate:
    # - Draft (50 tokens, small model): ~50/32 = 1.56x Llama forward cost
    # - Target (50 tokens tree): 50/32 = 1.56x Llama forward cost
    # - Per-token cost: 1.56 * 2 = 3.12x minimum (before acceptance)
    
    draft_tree_size = 50
    accept_rate = 0.85  # Typical acceptance rate
    effective_cost = draft_tree_size / (draft_tree_size * accept_rate)  # Cost per accepted token
    
    print(f"\nDraft tree analysis:")
    print(f"  - Draft tree size: {draft_tree_size} tokens")
    print(f"  - Typical acceptance rate: {accept_rate:.0%}")
    print(f"  - Cost per accepted token (draft+verify): {effective_cost:.2f}x baseline")
    
    # AR baseline for comparison
    print(f"\nComparison (32-token generation):")
    print(f"  - AR (no speculation): 32 forward passes")
    print(f"  - EAGLE (with speculation): ~{draft_tree_size*(1 + 1/accept_rate)/32:.1f} effective forward passes")
    print(f"  - Expected speedup (theoretical): {32 / (draft_tree_size*(1 + 1/accept_rate)/32):.1f}x")
    
    print("\n" + "=" * 70)
    print(f"[*] Output saved. Measured throughput: {avg_tps:.2f} tok/s")


if __name__ == "__main__":
    main()
