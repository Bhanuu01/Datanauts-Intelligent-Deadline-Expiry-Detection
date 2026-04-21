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

# ── Baseline: training data ───────────────────────────────────────────────────
train_records = load_jsonl('/data/train.jsonl')
print(f'  Baseline (train): {len(train_records)} records')

# ── Inference data: prefer live production records, fall back to CUAD test split
PROD_PATH = os.getenv('PRODUCTION_DATA_LOG_PATH', '/data/production_ingest.jsonl')
FALLBACK_PATH = '/data/test.jsonl'
MIN_PROD_RECORDS = int(os.getenv('MIN_PROD_RECORDS_FOR_DRIFT', '10'))

using_production = False
if os.path.exists(PROD_PATH):
    prod_candidates = load_jsonl(PROD_PATH)
    if len(prod_candidates) >= MIN_PROD_RECORDS:
        inference_records = prod_candidates
        using_production = True
        print(f'  Inference (production): {len(inference_records)} records from {PROD_PATH}')
    else:
        inference_records = load_jsonl(FALLBACK_PATH)
        print(f'  Inference proxy (test fallback — only {len(prod_candidates)} production records so far): '
              f'{len(inference_records)} records')
else:
    inference_records = load_jsonl(FALLBACK_PATH)
    print(f'  Inference proxy (test fallback): {len(inference_records)} records')

# ── Check 1: Text length drift ────────────────────────────────────────────────
train_lens = [len(str(r.get('ocr_text', '')).split()) for r in train_records]
train_avg  = float(np.mean(train_lens)) if train_lens else 1.0

if using_production:
    # production records hold candidate sentences inside a 'features' list
    sent_lens = [
        len(str(feat.get('sentence', '')).split())
        for r in inference_records
        for feat in r.get('features', [])
    ]
    infer_avg   = float(np.mean(sent_lens)) if sent_lens else train_avg
    drift_label = 'production sentence word counts'
else:
    infer_lens  = [len(str(r.get('ocr_text', '')).split()) for r in inference_records]
    infer_avg   = float(np.mean(infer_lens)) if infer_lens else train_avg
    drift_label = 'test split word counts'

drift_ratio = infer_avg / train_avg if train_avg > 0 else 1.0
print(f'  Train avg tokens: {train_avg:.1f}')
print(f'  Infer avg tokens ({drift_label}): {infer_avg:.1f}')
print(f'  Drift ratio: {drift_ratio:.2f}')
check('ocr_length_drift', drift_ratio < 2.0, f'ratio={drift_ratio:.2f} (threshold <2.0)')

# ── Check 2: Event type / candidate rate distribution ─────────────────────────
train_events = Counter(r.get('event_type', 'unknown') for r in train_records)

if using_production:
    total_sents    = max(sum(r.get('sentence_count', 1) for r in inference_records), 1)
    total_cands    = sum(r.get('candidate_count', 0)   for r in inference_records)
    prod_cand_rate = total_cands / total_sents
    train_dl_rate  = sum(
        1 for r in train_records if r.get('event_type', 'none') not in ('none', 'unknown')
    ) / max(len(train_records), 1)
    max_drift = abs(prod_cand_rate - train_dl_rate)
    check('event_type_distribution_drift', max_drift < 0.4,
          f'train_deadline_rate={train_dl_rate:.2f} prod_candidate_rate={prod_cand_rate:.2f} '
          f'drift={max_drift:.2f} (threshold <0.4)')
else:
    test_events = Counter(r.get('event_type', 'unknown') for r in inference_records)
    all_types   = set(train_events.keys()) | set(test_events.keys())
    max_drift   = 0.0
    for et in all_types:
        tf        = train_events.get(et, 0) / len(train_records)
        pf        = test_events.get(et, 0)  / len(inference_records)
        max_drift = max(max_drift, abs(tf - pf))
    check('event_type_distribution_drift', max_drift < 0.4,
          f'max event type drift={max_drift:.2f} (threshold <0.4)')

# ── Check 3: Null / empty record rate ─────────────────────────────────────────
train_null_rate = sum(1 for r in train_records if not r.get('ocr_text')) / max(len(train_records), 1)

if using_production:
    prod_null_rate = sum(1 for r in inference_records if not r.get('filename')) / max(len(inference_records), 1)
    check('null_rate_stable', abs(train_null_rate - prod_null_rate) < 0.1,
          f'train_null={train_null_rate:.3f} prod_null={prod_null_rate:.3f}')
else:
    test_null_rate = sum(1 for r in inference_records if not r.get('ocr_text')) / max(len(inference_records), 1)
    check('null_rate_stable', abs(train_null_rate - test_null_rate) < 0.1,
          f'train_null={train_null_rate:.3f} test_null={test_null_rate:.3f}')

# ── Check 4: Date field / candidate completeness ──────────────────────────────
train_date_rate = sum(1 for r in train_records if r.get('agreement_date_iso')) / max(len(train_records), 1)

if using_production:
    prod_date_rate = sum(
        1 for r in inference_records if r.get('candidate_count', 0) > 0
    ) / max(len(inference_records), 1)
    date_drift = abs(train_date_rate - prod_date_rate)
    check('date_completeness_stable', date_drift < 0.3,
          f'train_date_rate={train_date_rate:.2f} prod_candidate_rate={prod_date_rate:.2f} '
          f'drift={date_drift:.2f}')
else:
    test_date_rate = sum(1 for r in inference_records if r.get('agreement_date_iso')) / max(len(inference_records), 1)
    date_drift     = abs(train_date_rate - test_date_rate)
    check('date_completeness_stable', date_drift < 0.3,
          f'train_date_rate={train_date_rate:.2f} test_date_rate={test_date_rate:.2f} '
          f'drift={date_drift:.2f}')

# ── Check 5: Sufficient data for monitoring ───────────────────────────────────
check('inference_data_size_sufficient', len(inference_records) >= 10,
      f'{len(inference_records)} inference records (min 10)')

# ── Summary ───────────────────────────────────────────────────────────────────
REPORT['data_source'] = 'production_ingest' if using_production else 'test_split_proxy'
REPORT['metrics'] = {
    'train_avg_tokens':     train_avg,
    'inference_avg_tokens': infer_avg,
    'drift_ratio':          drift_ratio,
    'train_records':        len(train_records),
    'inference_records':    len(inference_records),
    'using_production_data': using_production,
}

passed_all     = all(c['status'] == 'PASS' for c in REPORT['checks'])
REPORT['overall'] = 'PASSED' if passed_all else 'FAILED'

with open('/app/eval_output/drift_monitoring_report.json', 'w') as f:
    json.dump(REPORT, f, indent=2)

print(f'=== DRIFT MONITORING: {REPORT["overall"]} ===')
print(f'    Data source:      {REPORT["data_source"]}')
print(f'    Train avg tokens: {train_avg:.1f}')
print(f'    Infer avg tokens: {infer_avg:.1f}')
print(f'    Drift ratio:      {drift_ratio:.2f}')
sys.exit(0 if passed_all else 1)
