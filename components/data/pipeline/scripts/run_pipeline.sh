#!/bin/bash
set -e
echo '================================================'
echo ' CUAD Ingestion Pipeline v2 - Starting'
echo '================================================'
echo '[1/4] Fetching CUAD from HuggingFace v2...'
python3 scripts/01_fetch_cuad.py
echo '[2/4] Cleaning dataset...'
python3 scripts/02_clean_cuad.py
echo '[3/4] Verifying train/val/test split...'
python3 scripts/03_split_dataset.py
echo '[4/4] Uploading to Chameleon object storage...'
python3 scripts/04_upload_chameleon.py
echo '================================================'
echo ' PIPELINE COMPLETE'
echo '================================================'
