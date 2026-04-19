#!/bin/bash
set -e
echo '================================================'
echo '  CUAD Ingestion Pipeline — Starting'
echo '================================================'
echo '[1/5] Fetching CUAD from Hugging Face...'
python3 scripts/01_fetch_cuad.py
echo '[2/5] Cleaning dataset...'
python3 scripts/02_clean_cuad.py
echo '[3/5] Generating synthetic invoice data...'
python3 scripts/03_generate_synthetic.py
echo '[4/5] Time-based train/val/test split...'
python3 scripts/04_split_dataset.py
echo '[5/5] Uploading to Chameleon object storage...'
python3 scripts/05_upload_chameleon.py
echo '================================================'
echo '  PIPELINE COMPLETE'
echo '================================================'
