#!/bin/bash
# run_demo.sh — single fast training run for Q2.3 demo video
# Runs bert_ner_v1 for 1 epoch to demonstrate the full pipeline end-to-end
# Usage: bash run_demo.sh
# Record with: asciinema rec demo.cast && bash run_demo.sh

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================="
echo " Datanauts Deadline Detection Pipeline — Demo Run"
echo "=================================================="
echo ""

echo "[1/4] Checking environment..."
python -c "
import torch, transformers, datasets, mlflow, seqeval
print('  PyTorch      :', torch.__version__)
print('  CUDA GPU     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
print('  Transformers :', transformers.__version__)
print('  MLflow       :', mlflow.__version__)
print('  All imports  : OK')
"

echo ""
echo "[2/4] Building sentence-level dataset from CUAD v2..."
if [ ! -d "$ROOT/data/deadline_sentences" ]; then
  python "$ROOT/src/build_dataset.py"
else
  echo "  Dataset already exists — skipping"
fi

echo ""
echo "[3/4] Training NER model (bert_ner_v1, 3 epochs)..."
python "$ROOT/src/train_ner.py" --model bert_ner_v1

echo ""
echo "[4/4] Training Classifier model (roberta_clf_v1, 3 epochs)..."
python "$ROOT/src/train_classifier.py" --model roberta_clf_v1

echo ""
echo "=================================================="
echo " Demo complete! Results logged to MLflow:"
echo " http://129.114.27.190:8000"
echo "=================================================="
