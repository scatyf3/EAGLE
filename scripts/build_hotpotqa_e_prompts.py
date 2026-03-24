import json
from pathlib import Path

root = Path('/mnt/hdd/yf/EAGLE')
zip_path = root / 'data/longbench_raw/data.zip'
extract_path = root / 'data/longbench_raw/data/hotpotqa_e.jsonl'
prompt_cfg = Path('/mnt/hdd/yf/LongBench-hip/config/dataset2prompt.json')
out_path = root / 'outputs/hotpotqa_e_prompts_longbenchhip.jsonl'

# Extract only the needed file from LongBench archive.
import zipfile
extract_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extract('data/hotpotqa_e.jsonl', path=root / 'data/longbench_raw')

prompt_format = json.load(prompt_cfg.open('r', encoding='utf-8'))['hotpotqa']

count = 0
with extract_path.open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
    for i, line in enumerate(fin):
        if not line.strip():
            continue
        ex = json.loads(line)
        prompt = prompt_format.format(**ex)
        row = {
            'idx': i,
            'dataset': 'hotpotqa_e',
            'question': ex.get('input', ''),
            'answers': ex.get('answers', []),
            'prompt': prompt,
        }
        fout.write(json.dumps(row, ensure_ascii=False) + '\n')
        count += 1

print(f'source={extract_path}')
print(f'output={out_path}')
print(f'count={count}')
