from huggingface_hub import hf_hub_download
import shutil, os, pandas as pd

os.makedirs('output/raw', exist_ok=True)
os.makedirs('output/data', exist_ok=True)

REPO = os.environ.get('HF_REPO', 'tanvitakavane/datanauts_project_cuad-deadline-ner-version2')

# Download parquet files
parquet_files = [
    'data/train-00000-of-00001.parquet',
    'data/val-00000-of-00001.parquet',
    'data/test-00000-of-00001.parquet',
]
for filename in parquet_files:
    local = hf_hub_download(repo_id=REPO, filename=filename, repo_type='dataset')
    dest = f'output/{filename}'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy(local, dest)
    print(f'Downloaded: {filename}')

# Convert parquet to jsonl
train_df = pd.read_parquet('output/data/train-00000-of-00001.parquet')
val_df   = pd.read_parquet('output/data/val-00000-of-00001.parquet')
test_df  = pd.read_parquet('output/data/test-00000-of-00001.parquet')

train_df.to_json('output/data/train.jsonl',      orient='records', lines=True)
val_df.to_json(  'output/data/validation.jsonl', orient='records', lines=True)
test_df.to_json( 'output/data/test.jsonl',       orient='records', lines=True)

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
all_df.to_json('output/data/cuad_cleaned.jsonl', orient='records', lines=True)
all_df.to_csv( 'output/raw/master_clauses.csv',  index=False)

print('Downloaded: raw/master_clauses.csv')
print('Downloaded: data/train.jsonl')
print('Downloaded: data/validation.jsonl')
print('Downloaded: data/test.jsonl')
print('Downloaded: data/cuad_cleaned.jsonl')
