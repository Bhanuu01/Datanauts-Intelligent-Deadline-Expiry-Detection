#!/usr/bin/env python3
"""
EVALUATION POINT 3: Production Drift Monitor
Monitors live inference data quality and distribution drift.
Compares incoming inference requests against training baseline.
Dataset: tanvitakavane/datanauts_project_cuad-deadline-ner-version2
"""
import json, os, sys
import numpy as np
from datetime import datetime
from collections import Counter

os.makedirs('/app/eval_output', exist_ok=True)
REPORT = {
    'evaluation_point': 'production_drift',
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

# Load train as baseline, test as proxy for inference data
train_records = load_jsonl('/data/train.jsonl')
test_records  = load_jsonl('/data/test.jsonl')
val_records   = load_jsonl('/data/validation.jsonl')

print(f'  Baseline (train): {len(train_records)} records')
print(f'  Inference proxy (test): {len(test_records)} records')

# Check 1: OCR text length drift
train_lens = [len(str(r.get('ocr_text', '')).split()) for r in train_records]
test_lens  = [len(str(r.get('ocr_text', '')).split()) for r in test_records]

train_avg = float(np.mean(train_lens))
test_avg  = float(np.mean(test_lens))
drift_ratio = test_avg / train_avg if train_avg > 0 else 1

print(f'  Train avg tokens: {train_avg:.1f}')
print(f'  Test avg tokens:  {test_avg:.1f}')
print(f'  Drift ratio: {drift_ratio:.2f}')

check('ocr_length_drift', drift_ratio < 2.0,
      f'ratio={drift_ratio:.2f} (threshold <2.0)')

# Check 2: Event type distribution drift
train_events = Counter(r.get('event_type', 'unknown') for r in train_records)
test_events  = Counter(r.get('event_type', 'unknown') for r in test_records)

all_types = set(train_events.keys()) | set(test_events.keys())
max_drift = 0.0
for et in all_types:
    train_frac = train_events.get(et, 0) / len(train_records)
    test_frac  = test_events.get(et, 0) / len(test_records)
    drift = abs(train_frac - test_frac)
    if drift > max_drift:
        max_drift = drift

check('event_type_distribution_drift', max_drift < 0.4,
      f'max event type drift={max_drift:.2f} (threshold <0.4)')

# Check 3: Null rate drift
train_null_rate = sum(1 for r in train_records if not r.get('ocr_text')) / len(train_records)
test_null_rate  = sum(1 for r in test_records if not r.get('ocr_text')) / len(test_records)
check('null_rate_stable', abs(train_null_rate - test_null_rate) < 0.1,
      f'train_null={train_null_rate:.3f} test_null={test_null_rate:.3f}')

# Check 4: Date field completeness drift
train_date_rate = sum(1 for r in train_records if r.get('agreement_date_iso')) / len(train_records)
test_date_rate  = sum(1 for r in test_records if r.get('agreement_date_iso')) / len(test_records)
date_drift = abs(train_date_rate - test_date_rate)
check('date_completeness_stable', date_drift < 0.3,
      f'train_date_rate={train_date_rate:.2f} test_date_rate={test_date_rate:.2f} drift={date_drift:.2f}')

# Check 5: Dataset size sufficient for monitoring
check('inference_data_size_sufficient', len(test_records) >= 10,
      f'{len(test_records)} inference records (min 10)')

# Summary
REPORT['metrics'] = {
    'train_avg_tokens': train_avg,
    'test_avg_tokens': test_avg,
    'drift_ratio': drift_ratio,
    'max_event_drift': max_drift,
    'train_event_distribution': dict(train_events),
    'test_event_distribution': dict(test_events)
}

passed_all = all(c['status'] == 'PASS' for c in REPORT['checks'])
REPORT['overall'] = 'PASSED' if passed_all else 'FAILED'

with open('/app/eval_output/drift_monitoring_report.json', 'w') as f:
    json.dump(REPORT, f, indent=2)

print(f'=== DRIFT MONITORING: {REPORT["overall"]} ===')
print(f'    Train avg tokens: {train_avg:.1f}')
print(f'    Test avg tokens:  {test_avg:.1f}')
print(f'    Drift ratio: {drift_ratio:.2f}')
sys.exit(0 if passed_all else 1)
