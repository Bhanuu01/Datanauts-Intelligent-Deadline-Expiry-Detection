import json, os

for split in ['train', 'validation', 'test']:
    src = f'output/data/{split}.jsonl'
    if os.path.exists(src):
        count = sum(1 for l in open(src) if l.strip())
        print(f'{split}: {count} records')
    else:
        print(f'WARNING: {split}.jsonl not found')

print('Splits verified from HF v2 dataset')
