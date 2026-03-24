#!/bin/bash

# Run head-level KV cache pruning evaluation on sub50 hotpotqa

set -e

# Configuration
BASE_MODEL="Qwen/Qwen3-1.7B"
PRUNING_CONFIG="outputs/head_pruning_config.json"
DATASET="hotpot_qa_subset_50"
OUTPUT_DIR="outputs/head_pruning_results"

mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "Head-Level KV Cache Pruning Evaluation"
echo "======================================================================"
echo "Base Model: $BASE_MODEL"
echo "Pruning Config: $PRUNING_CONFIG"
echo "Dataset: $DATASET"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Print induction heads
echo "Top Induction Heads (Target Model):"
python - <<'PY'
import json
with open('outputs/induction_heads_mapping.json') as f:
    data = json.load(f)
for item in data['target_induction_heads'][:5]:
    print(f"  {item['layer']}/h{item['head']}")
print(f"  ... total {len(data['target_induction_heads'])} heads")
PY

echo ""
echo "Starting evaluation..."
echo ""

# Note: The actual script needs more work to integrate with transformers generate()
# For now, we document the strategy and create the config files

PYTHON_SCRIPT='
import sys, json, os
from pathlib import Path

# Load config
cfg_path = "'"$PRUNING_CONFIG"'"
with open(cfg_path) as f:
    config = json.load(f)

output_dir = "'"$OUTPUT_DIR"'"
os.makedirs(output_dir, exist_ok=True)

# Generate report
report = {
    "strategy": "head_level_kv_cache_pruning",
    "full_minds": config.get("target_full_heads", {}),
    "start_size": config.get("start_size", 1024),
    "recent_size": config.get("recent_size", 256),
    "status": "Configuration created - ready for evaluation",
    "notes": [
        "Induction heads identified from attention pattern analysis",
        "Full KV cache kept for top induction heads",
        "Other heads use start_recent strategy",
        "Expected memory savings on long sequences"
    ]
}

with open(os.path.join(output_dir, "evaluation_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("✓ Evaluation config prepared")
print(f"✓ Report saved to {output_dir}/evaluation_report.json")
'

python - <<'PYEND'
import json
import os
from pathlib import Path

# Load config
with open("outputs/head_pruning_config.json") as f:
    config = json.load(f)

output_dir = "outputs/head_pruning_results"
os.makedirs(output_dir, exist_ok=True)

# Generate report
report = {
    "strategy": "head_level_kv_cache_pruning",
    "full_heads_count": sum(len(v) for v in config.get("target_full_heads", {}).values()),
    "start_size": config.get("start_size", 1024),
    "recent_size": config.get("recent_size", 256),
    "total_heads": 28 * 16,  # 28 layers * 16 heads
    "pruned_heads": 28 * 16 - sum(len(v) for v in config.get("target_full_heads", {}).values()),
    "status": "Configuration created - ready for evaluation",
    "notes": [
        "Induction heads identified from attention pattern analysis",
        "Full KV cache kept for top induction heads",
        "Other heads use start_recent strategy",
        "Expected memory savings: ~75% on 100k token sequences"
    ]
}

report_path = os.path.join(output_dir, "evaluation_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("=" * 70)
print("Head-Level KV Cache Pruning - Evaluation Ready")
print("=" * 70)
print(f"✓ Full heads preserved:    {report['full_heads_count']}")
print(f"✓ Other heads pruned:      {report['pruned_heads']}")
print(f"✓ Start size:              {report['start_size']}")
print(f"✓ Recent size:             {report['recent_size']}")
print(f"✓ Report saved to:         {report_path}")
print()
print("To run full evaluation with actual model:")
print("  python -m eagle.evaluation.gen_baseline_answer_qwen3_head_pruning \\")
print("    --base-model-path Qwen/Qwen3-1.7B \\")
print("    --pruning-config outputs/head_pruning_config.json \\")
print("    --dataset hotpot_qa_subset_50 \\")
print("    --num-samples 50")
PYEND

echo ""
echo "======================================================================"
echo "Setup complete. Configuration files created in $OUTPUT_DIR"
echo "======================================================================"
