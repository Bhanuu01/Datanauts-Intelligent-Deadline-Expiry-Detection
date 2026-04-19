#!/usr/bin/env python3
import great_expectations as gx
import pandas as pd
import json, os, sys
import numpy as np
from pathlib import Path

def convert(o):
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return str(o)

os.makedirs('/app/gx_output', exist_ok=True)
results_summary = {}

print('=== Validation Point 1: Raw CSV (ingestion gate) ===')
try:
    df_raw = pd.read_csv('/data/master_clauses.csv')
    print(f'Raw CSV columns: {list(df_raw.columns[:5])}...')
    print(f'Raw CSV shape: {df_raw.shape}')
    context1 = gx.get_context()
    ds1 = context1.sources.add_pandas('raw_source')
    da1 = ds1.add_dataframe_asset('raw_cuad')
    batch1 = da1.build_batch_request(dataframe=df_raw)
    suite1 = context1.add_expectation_suite('raw_suite')
    validator1 = context1.get_validator(batch_request=batch1, expectation_suite=suite1)
    validator1.expect_column_to_exist('Filename')
    validator1.expect_column_to_exist('event_type')
    validator1.expect_column_to_exist('agreement_date_iso')
    validator1.expect_table_row_count_to_be_between(min_value=100, max_value=10000)
    validator1.expect_column_values_to_not_be_null('Filename', mostly=0.99)
    validator1.save_expectation_suite()
    results1 = validator1.validate()
    passed1 = bool(results1['success'])
    print(f'Raw CSV validation: {"PASSED" if passed1 else "FAILED"}')
    results_summary['raw_csv'] = {'passed': passed1, 'stats': {'evaluated': int(results1['statistics']['evaluated_expectations']), 'successful': int(results1['statistics']['successful_expectations'])}}
    if not passed1:
        print('CRITICAL: Raw CSV failed validation. Stopping pipeline.')
        sys.exit(1)
except Exception as e:
    print(f'Raw CSV validation error: {e}')
    results_summary['raw_csv'] = {'passed': False, 'error': str(e)}

print('=== Validation Point 2: Cleaned JSONL (training gate) ===')
try:
    records = [json.loads(l) for l in open('/data/cuad_cleaned.jsonl') if l.strip()]
    df_clean = pd.DataFrame(records)
    print(f'Cleaned JSONL columns: {list(df_clean.columns[:5])}...')
    print(f'Cleaned JSONL shape: {df_clean.shape}')
    context2 = gx.get_context()
    ds2 = context2.sources.add_pandas('clean_source')
    da2 = ds2.add_dataframe_asset('clean_cuad')
    batch2 = da2.build_batch_request(dataframe=df_clean)
    suite2 = context2.add_expectation_suite('clean_suite')
    validator2 = context2.get_validator(batch_request=batch2, expectation_suite=suite2)
    validator2.expect_column_to_exist('Filename')
    validator2.expect_column_to_exist('ocr_text')
    validator2.expect_column_to_exist('event_type')
    validator2.expect_column_to_exist('agreement_date_iso')
    validator2.expect_table_row_count_to_be_between(min_value=100, max_value=10000)
    validator2.expect_column_values_to_not_be_null('Filename', mostly=0.99)
    validator2.expect_column_value_lengths_to_be_between('Filename', min_value=5, mostly=0.95)
    validator2.save_expectation_suite()
    results2 = validator2.validate()
    passed2 = bool(results2['success'])
    print(f'Cleaned JSONL validation: {"PASSED" if passed2 else "FAILED"}')
    results_summary['clean_jsonl'] = {'passed': passed2, 'stats': {'evaluated': int(results2['statistics']['evaluated_expectations']), 'successful': int(results2['statistics']['successful_expectations'])}}
    if not passed2:
        failed = [r for r in results2['results'] if not r['success']]
        for f in failed:
            print(f'  FAILED: {f["expectation_config"]["expectation_type"]}')
except Exception as e:
    print(f'Cleaned JSONL validation error: {e}')
    results_summary['clean_jsonl'] = {'passed': False, 'error': str(e)}

print('=== Validation Point 3: Inference data drift check ===')
try:
    train_records = [json.loads(l) for l in open('/data/train.jsonl') if l.strip()]
    val_records   = [json.loads(l) for l in open('/data/validation.jsonl') if l.strip()]
    test_records  = [json.loads(l) for l in open('/data/test.jsonl') if l.strip()]
    train_lens = [len(str(r.get('ocr_text', '')).split()) for r in train_records]
    test_lens  = [len(str(r.get('ocr_text', '')).split()) for r in test_records]
    print(f'  Train: {len(train_records)} records, avg tokens: {np.mean(train_lens):.1f}')
    print(f'  Val:   {len(val_records)} records')
    print(f'  Test:  {len(test_records)} records, avg tokens: {np.mean(test_lens):.1f}')
    drift_ok = bool(np.mean(test_lens) < 2 * np.mean(train_lens))
    print(f'  Drift check: {"PASSED" if drift_ok else "FAILED"}')
    results_summary['drift_check'] = {
        'passed': drift_ok,
        'train_avg_tokens': float(np.mean(train_lens)),
        'test_avg_tokens': float(np.mean(test_lens))
    }
except Exception as e:
    print(f'Drift check error: {e}')
    results_summary['drift_check'] = {'passed': False, 'error': str(e)}

with open('/app/gx_output/validation_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=convert)

all_passed = all(v.get('passed', False) for v in results_summary.values())
print(f'=== OVERALL: {"ALL PASSED" if all_passed else "SOME FAILED"} ===')
print('Results saved to /app/gx_output/validation_results.json')
sys.exit(0 if all_passed else 1)
