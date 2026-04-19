from datasets import load_dataset
import pandas as pd, os

os.makedirs('output/raw', exist_ok=True)
print('Downloading CUAD from Hugging Face...')
dataset = load_dataset('theatticusproject/cuad', split='train')
df = dataset.to_pandas()
df.to_csv('output/raw/master_clauses.csv', index=False)
print(f'Downloaded {len(df)} contracts, {len(df.columns)} columns')
print('Saved: output/raw/master_clauses.csv')
