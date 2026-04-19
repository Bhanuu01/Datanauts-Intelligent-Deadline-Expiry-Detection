import ast, re, json, os
import pandas as pd
import nltk
from datetime import datetime
from tqdm import tqdm

os.makedirs('output/data', exist_ok=True)

def parse_list_col(val):
    val = str(val).strip()
    if not val or val in ['[]', 'nan']:
        return []
    try:
        parsed = ast.literal_eval(val)
        return [str(x) for x in parsed if x] if isinstance(parsed, list) else [str(parsed)]
    except:
        return [val]

def normalize_date(answer):
    answer = str(answer).strip()
    if not answer or answer in ['[]', 'nan']: return None
    if 'perpetual' in answer.lower(): return None
    if '[' in answer: return None
    if ';' in answer: return None
    for fmt in ['%m/%d/%y', '%m/%d/%Y']:
        try:
            return datetime.strptime(answer, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None

DATE_RE = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}'
    r'|\b\d{1,2}/\d{1,2}/\d{2,4}'
    r'|\b\d{4}-\d{2}-\d{2}'
    r'|\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}',
    re.IGNORECASE
)

def iob_tag(sentence):
    tokens = sentence.split()
    labels = ['O'] * len(tokens)
    for match in DATE_RE.finditer(sentence):
        s, e, first = match.start(), match.end(), True
        char_pos = 0
        for i, tok in enumerate(tokens):
            ts = sentence.find(tok, char_pos)
            te = ts + len(tok)
            if ts >= s and te <= e + 2:
                labels[i] = 'B-DATE' if first else 'I-DATE'
                first = False
            char_pos = te
    return tokens, labels

print('Loading raw CUAD data...')
df = pd.read_csv('output/raw/master_clauses.csv')

CLAUSE_COLS = [
    ('Expiration Date', 'Expiration Date-Answer', 'expiration'),
    ('Effective Date', 'Effective Date-Answer', 'effective'),
    ('Agreement Date', 'Agreement Date-Answer', 'agreement'),
]

samples = []
skipped = {'no_answer':0, 'perpetual':0, 'redacted':0, 'no_text':0}

for text_col, ans_col, event_type in CLAUSE_COLS:
    print(f'Processing: {text_col}')
    col_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=text_col):
        raw_ans = str(row.get(ans_col, ''))
        if not raw_ans or raw_ans in ['[]','nan']:
            skipped['no_answer'] += 1; continue
        if 'perpetual' in raw_ans.lower():
            skipped['perpetual'] += 1; continue
        if '[' in raw_ans:
            skipped['redacted'] += 1; continue
        answer = normalize_date(raw_ans)
        if not answer:
            skipped['no_answer'] += 1; continue
        text_spans = parse_list_col(row.get(text_col, ''))
        if not text_spans:
            skipped['no_text'] += 1; continue
        for span in text_spans:
            for sent in nltk.sent_tokenize(span):
                sent = sent.strip()
                if len(sent) < 15:
                    continue
                tokens, labels = iob_tag(sent)
                samples.append({
                    'filename': row['Filename'],
                    'event_type': event_type,
                    'sentence': sent,
                    'tokens': tokens,
                    'ner_labels': labels,
                    'ground_truth_date': answer,
                    'agreement_date_raw': str(row.get('Agreement Date-Answer','')),
                })
        col_count += 1
    print(f'  -> {col_count} samples from {text_col}')

print(f'\n=== CLEANING COMPLETE ===')
print(f'Total usable samples: {len(samples)}')
print(f'Skipped - no answer: {skipped["no_answer"]}')
print(f'Skipped - perpetual: {skipped["perpetual"]}')
print(f'Skipped - redacted:  {skipped["redacted"]}')
print(f'Skipped - no text:   {skipped["no_text"]}')

with open('output/data/cuad_cleaned.jsonl', 'w') as f:
    for s in samples:
        f.write(json.dumps(s) + '\n')
print('Saved: output/data/cuad_cleaned.jsonl')
