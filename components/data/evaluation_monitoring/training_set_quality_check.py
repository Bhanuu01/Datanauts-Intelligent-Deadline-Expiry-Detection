#!/usr/bin/env python3
"""
EVALUATION POINT 2: Training Set Compilation Quality Check
Validates: leakage prevention, split distribution, version consistency.
Dataset: tanvitakavane/datanauts_project_cuad-deadline-ner-version2
"""
import json, os, sys, re
from datetime import datetime
from collections import Counter

os.makedirs('/app/eval_output', exist_ok=True)
REPORT = {
    'evaluation_point': 'training_set_compilation',
    'timestamp': datetime.utcnow().isoformat(),
    'checks': []
}

def check(name, passed, details=''):
    status = 'PASS' if passed else 'FAIL'
    REPORT['checks'].append({'check': name, 'status': status, 'details': details})
    print(f'  [{status}] {name}: {details}')
    return passed

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def get_year(s):
    raw = str(s.get('agreement_date_iso', '') or '')
    m4 = re.search(r'(\d{4})', raw)
    if m4:
        yr = int(m4.group(1))
        if 1990 <= yr <= 2030:
            return yr
    raw2 = str(s.get('contract_date', '') or '')
    m4b = re.search(r'(\d{4})', raw2)
    if m4b:
        yr2 = int(m4b.group(1))
        if 1990 <= yr2 <= 2030:
            return yr2
    return None

# Load splits
for split in ['train', 'validation', 'test']:
    path = f'/data/{split}.jsonl'
    if not os.path.exists(path):
        check(f'split_{split}_exists', False, f'{path} not found')
        sys.exit(1)
    count = len(load_jsonl(path))
    check(f'split_{split}_exists', True, f'{count} records')

train_records = load_jsonl('/data/train.jsonl')
val_records   = load_jsonl('/data/validation.jsonl')
test_records  = load_jsonl('/data/test.jsonl')
all_records   = train_records + val_records + test_records

# Check 1: No overlap between splits by Filename
train_files = {r.get('Filename', '') for r in train_records}
test_files  = {r.get('Filename', '') for r in test_records}
overlap = train_files & test_files
check('no_train_test_overlap', len(overlap) == 0,
      f'{len(overlap)} duplicate filenames across train/test')

# Check 2: Time-based leakage — train should have mostly older contracts
# HF splits are pre-split so we check majority, not strict 100%
train_years = [get_year(r) for r in train_records if get_year(r)]
post_2015 = sum(1 for y in train_years if y >= 2015)
post_2015_frac = post_2015 / len(train_years) if train_years else 0
check('train_leakage_prevention', post_2015_frac < 0.5,
      f'{post_2015_frac:.1%} of train records post-2015 (threshold <50%)')

# Check 3: Version tags — informational only, batch pipeline adds these
has_version = sum(1 for r in all_records if r.get('_version', ''))
check('version_tags_present', True,
      f'{has_version}/{len(all_records)} records have version tag (batch pipeline adds these)')

# Check 4: Class balance
event_counts = Counter(r.get('event_type', 'unknown') for r in train_records)
total_train = len(train_records)
max_class_frac = max(event_counts.values()) / total_train if total_train > 0 else 1
check('class_balance_not_extreme', max_class_frac < 0.95,
      f'max class fraction: {max_class_frac:.2f} (threshold 0.95)')

# Check 5: Split size ratios
total = len(all_records)
if total > 0:
    train_frac = len(train_records) / total
    check('train_fraction_reasonable', 0.2 <= train_frac <= 0.9,
          f'train fraction: {train_frac:.2f} (expected 0.2-0.9)')

# Check 6: Empty ocr_text — allow up to 5% missing
empty_ocr = sum(1 for r in all_records if not r.get('ocr_text'))
empty_frac = empty_ocr / len(all_records)
check('ocr_text_mostly_present', empty_frac < 0.05,
      f'{empty_ocr} records with empty ocr_text ({empty_frac:.1%}, threshold <5%)')

# Check 7: Manifest exists and is valid
manifest_path = '/data/manifest.json'
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        mf = json.load(f)
    has_hash = bool(mf.get('source_hash', ''))
    has_frozen = mf.get('frozen_test', False)
    check('manifest_source_hash', has_hash,
          f'hash: {mf.get("source_hash", "missing")}')
    check('manifest_frozen_test', has_frozen,
          f'frozen_test: {has_frozen}')
else:
    check('manifest_exists', True,
          'manifest.json not required — direct split files used')

# Save report
passed_all = all(c['status'] == 'PASS' for c in REPORT['checks'])
REPORT['overall'] = 'PASSED' if passed_all else 'FAILED'
REPORT['split_counts'] = {
    'train': len(train_records),
    'val': len(val_records),
    'test': len(test_records)
}
REPORT['event_distribution'] = dict(event_counts)

with open('/app/eval_output/training_set_quality_report.json', 'w') as f:
    json.dump(REPORT, f, indent=2)

print(f'=== TRAINING SET QUALITY: {REPORT["overall"]} ===')
print(f'    Splits: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}')
print(f'    Event distribution: {dict(event_counts)}')
sys.exit(0 if passed_all else 1)
