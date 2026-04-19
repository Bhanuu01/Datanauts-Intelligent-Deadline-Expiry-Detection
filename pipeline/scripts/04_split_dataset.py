import json, re, os

os.makedirs('output/data', exist_ok=True)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def get_year(s):
    raw = str(s.get('agreement_date_raw', ''))
    m = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2})$', raw)
    if m:
        yy = int(m.group(3))
        return 2000+yy if yy <= 68 else 1900+yy
    m4 = re.search(r'(\d{4})', raw)
    if m4:
        yr = int(m4.group(1))
        if 1990 <= yr <= 2030: return yr
    return None

cuad = load_jsonl('output/data/cuad_cleaned.jsonl')
synth = load_jsonl('output/synthetic/synthetic_ner.jsonl')

train, val, test = [], [], []
assignments = {}

for s in cuad:
    fname = s.get('filename', '')
    if fname in assignments:
        split = assignments[fname]
    else:
        yr = get_year(s)
        if yr is None or yr < 2015: split = 'train'
        elif yr <= 2018: split = 'val'
        else: split = 'test'
        assignments[fname] = split
    if split == 'train': train.append(s)
    elif split == 'val': val.append(s)
    else: test.append(s)

# Synthetic goes to train
train.extend(synth)

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for s in data:
            f.write(json.dumps(s) + '\n')

save_jsonl(train, 'output/data/train.jsonl')
save_jsonl(val,   'output/data/validation.jsonl')
save_jsonl(test,  'output/data/test.jsonl')

print(f'Train: {len(train)}  Val: {len(val)}  Test: {len(test)}')
print('Saved: output/data/train.jsonl, validation.jsonl, test.jsonl')
