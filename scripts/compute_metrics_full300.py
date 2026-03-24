import json
from statistics import mean

files = [
    ('prune_draft_full300',
     'outputs/qwen3_hotpotqa_e_prompt_draftonlyprune_full300.jsonl'),
    ('induction_heads_full300',
     'outputs/qwen3_hotpotqa_e_prompt_draft_inductionheads_full300.jsonl'),
]

for label, path in files:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pre, wall, nt, idx = [], [], [], []
    for r in rows:
        c = r['choices'][0]
        pre.append(float(sum(c.get('prefill_time', [0]) if isinstance(c.get('prefill_time'), list) else [c.get('prefill_time', 0)])))
        wall.append(float(sum(c.get('wall_time', [0]) if isinstance(c.get('wall_time'), list) else [c.get('wall_time', 0)])))
        nt.append(float(sum(c.get('new_tokens', [0]) if isinstance(c.get('new_tokens'), list) else [c.get('new_tokens', 0)])))
        idx.append(float(sum(c.get('idxs', [0]) if isinstance(c.get('idxs'), list) else [c.get('idxs', 0)])))
    dec = [max(w - p, 0.0) for w, p in zip(wall, pre)]
    acc = mean(nt[i] / idx[i] for i in range(len(rows)) if idx[i] > 0)
    tok_ps = sum(nt) / sum(wall)
    dec_ps = sum(nt) / sum(dec)
    # context length via prompt word-count if available
    ctx_lens = [len(r.get('prompt', '').split()) for r in rows]
    avg_ctx = mean(ctx_lens) if any(ctx_lens) else 0
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"  choices[0] keys: {list(rows[0]['choices'][0].keys())}")
    print(f"  avg_context_length (words): {avg_ctx:.1f}")
    print(f"  avg_acceptance_length: {acc:.3f}")
    print(f"  avg new_tokens: {mean(nt):.2f}")
    print(f"  throughput tok/s: {tok_ps:.2f}")
    print(f"  decode-only tok/s: {dec_ps:.2f}")
    print(f"  total_wall={sum(wall):.1f}s  total_decode={sum(dec):.1f}s")
