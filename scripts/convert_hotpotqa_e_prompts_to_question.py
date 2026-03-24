import json
from pathlib import Path

src = Path('/mnt/hdd/yf/EAGLE/outputs/hotpotqa_e_prompts_longbenchhip.jsonl')
dst = Path('/mnt/hdd/yf/EAGLE/eagle/data/hotpotqa_e_prompt/question.jsonl')
dst.parent.mkdir(parents=True, exist_ok=True)

count = 0
with src.open('r', encoding='utf-8') as fin, dst.open('w', encoding='utf-8') as fout:
    for i, line in enumerate(fin):
        if not line.strip():
            continue
        obj = json.loads(line)
        out = {
            'question_id': obj.get('idx', i),
            'category': 'longbench-hotpotqa_e-prompt',
            'turns': [obj.get('prompt', '')],
            'reference': obj.get('answers', []),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + '\n')
        count += 1

print(f'written={count}')
print(f'output={dst}')
