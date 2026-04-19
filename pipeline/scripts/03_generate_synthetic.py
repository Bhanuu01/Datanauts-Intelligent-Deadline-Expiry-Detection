import json, os, re, random
from faker import Faker
from datetime import datetime, timedelta

os.makedirs('output/synthetic', exist_ok=True)
fake = Faker()
random.seed(42)

DATE_RE = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}'
    r'|\b\d{1,2}/\d{1,2}/\d{2,4}'
    r'|\b\d{4}-\d{2}-\d{2}',
    re.IGNORECASE
)

TEMPLATES = [
    'Payment of ${amount} is due by {date}.',
    'Invoice #{inv} — Amount: ${amount}. Due: {date}.',
    'Please remit ${amount} no later than {date}.',
    'Balance due ${amount} on or before {date}.',
    'This invoice is payable by {date}.',
    'Amount owed: ${amount}. Payment deadline: {date}.',
]

def make_date():
    base = datetime(2020, 1, 1)
    end = datetime(2027, 12, 31)
    delta = end - base
    return (base + timedelta(days=random.randint(0, delta.days))).strftime('%B %d, %Y')

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

samples = []
for i in range(500):
    date = make_date()
    amount = f'{random.randint(500,50000):,}.{random.randint(0,99):02d}'
    inv = fake.bothify('INV-#####')
    template = random.choice(TEMPLATES)
    sentence = template.format(date=date, amount=amount, inv=inv)
    tokens, labels = iob_tag(sentence)
    samples.append({
        'filename': f'synthetic_invoice_{i:04d}.pdf',
        'event_type': 'payment_due',
        'sentence': sentence,
        'tokens': tokens,
        'ner_labels': labels,
        'ground_truth_date': datetime.strptime(date, '%B %d, %Y').strftime('%Y-%m-%d'),
        'agreement_date_raw': '',
    })

with open('output/synthetic/synthetic_ner.jsonl', 'w') as f:
    for s in samples:
        f.write(json.dumps(s) + '\n')

print(f'Generated {len(samples)} synthetic invoice samples')
print('Saved: output/synthetic/synthetic_ner.jsonl')
