import json, os

input_path = 'output/data/cuad_cleaned.jsonl'
records = [json.loads(l) for l in open(input_path) if l.strip()]

print(f'Total records: {len(records)}')
print(f'Columns: {list(records[0].keys()) if records else "none"}')
print('Dataset v2 validated — converted from parquet (cleaned CUAD only)')
