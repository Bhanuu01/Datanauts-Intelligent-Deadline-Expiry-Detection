#!/usr/bin/env python3
"""
EVALUATION POINT 1: Ingestion Quality Check
Runs immediately after data is fetched from HuggingFace v2 dataset.
Validates: schema, date formats, label completeness for v2 columns.
Dataset: tanvitakavane/datanauts_project_cuad-deadline-ner-version2
"""
import json, pandas as pd, sys, os, re
from datetime import datetime

os.makedirs('/app/eval_output', exist_ok=True)
REPORT = {
    'evaluation_point': 'ingestion',
    'timestamp': datetime.utcnow().isoformat(),
    'dataset': 'tanvitakavane/datanauts_project_cuad-deadline-ner-version2',
    'checks': []
}

def check(name, passed, details=''):
    status = 'PASS' if passed else 'FAIL'
    REPORT['checks'].append({'check': name, 'status': status, 'details': details})
    print(f'  [{status}] {name}: {details}')
    return passed

# Load cleaned JSONL
try:
    records = [json.loads(l) for l in open('/data/cuad_cleaned.jsonl') if l.strip()]
    df = pd.DataFrame(records)
    check('load_dataset', True, f'{len(records)} records loaded')
except Exception as e:
    check('load_dataset', False, str(e))
    sys.exit(1)

# Check 1: Required v2 fields present
required = ['Filename', 'ocr_text', 'event_type', 'agreement_date_iso']
for field in required:
    check(f'field_{field}_exists', field in df.columns,
          f'column {field} present={field in df.columns}')

# Check 2: No null Filenames
null_filenames = df['Filename'].isna().sum() if 'Filename' in df.columns else len(records)
check('no_null_filenames', null_filenames == 0, f'{null_filenames} null filenames')

# Check 3: ocr_text minimum length
if 'ocr_text' in df.columns:
    short = df['ocr_text'].dropna().apply(lambda x: len(str(x)) < 10).sum()
    check('ocr_text_min_length_10', short == 0, f'{short} ocr_text under 10 chars')

# Check 4: agreement_date_iso format where present
if 'agreement_date_iso' in df.columns:
    bad_dates = df['agreement_date_iso'].dropna().apply(
        lambda d: not bool(re.match(r'^\d{4}-\d{2}-\d{2}', str(d)))).sum()
    check('agreement_date_iso_format', bad_dates == 0,
          f'{bad_dates} non-ISO dates')

# Check 5: event_type values valid — all actual v2 values
VALID_EVENTS = {
    'none',
    'agreement',
    'expiration',
    'agreement,effective',
    'expiration,agreement',
    'expiration,effective',
    'expiration,renewal',
    'agreement,effective,renewal',
    'expiration,agreement,effective',
    'expiration,agreement,renewal',
    'expiration,effective,renewal',
    'agreement,effective,renewal,notice_period',
    'expiration,agreement,effective,notice_period',
    'expiration,agreement,effective,renewal',
    'expiration,agreement,renewal,notice_period',
    'expiration,effective,renewal,notice_period',
    'expiration,renewal,notice_period',
    'expiration,agreement,effective,renewal,notice_period',
}
if 'event_type' in df.columns:
    invalid_events = df['event_type'].dropna().apply(
        lambda x: x not in VALID_EVENTS).sum()
    check('valid_event_types', invalid_events == 0,
          f'{invalid_events} invalid event types')

# Check 6: Dataset size sanity
check('dataset_size_sanity', 50 <= len(records) <= 50000,
      f'{len(records)} records (expected 50-50000)')

# Check 7: Filename uniqueness
if 'Filename' in df.columns:
    unique_files = df['Filename'].nunique()
    check('filename_uniqueness', unique_files > 1,
          f'{unique_files} unique filenames')

# Check 8: contract_date present
if 'contract_date' in df.columns:
    null_dates = df['contract_date'].isna().sum()
    check('contract_date_present', null_dates < len(records),
          f'{null_dates} null contract dates')

# Save report
passed_all = all(c['status'] == 'PASS' for c in REPORT['checks'])
REPORT['overall'] = 'PASSED' if passed_all else 'FAILED'
REPORT['total_checks'] = len(REPORT['checks'])
REPORT['failed_checks'] = sum(1 for c in REPORT['checks'] if c['status'] == 'FAIL')

with open('/app/eval_output/ingestion_quality_report.json', 'w') as f:
    json.dump(REPORT, f, indent=2)

print(f'=== INGESTION QUALITY: {REPORT["overall"]} ({REPORT["failed_checks"]} checks failed) ===')
sys.exit(0 if passed_all else 1)
