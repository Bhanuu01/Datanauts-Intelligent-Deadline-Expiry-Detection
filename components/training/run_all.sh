#!/bin/bash
# run_all.sh — run every NER + classifier variant in sequence and log to MLflow
# Usage: bash run_all.sh
# Requires: build_dataset.py to have been run first (./data/deadline_sentences must exist)

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY="$LOG_DIR/run_summary_${TIMESTAMP}.txt"

log() {
  echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"
}

run_variant() {
  SCRIPT=$1
  MODEL=$2
  log "START  $SCRIPT --model $MODEL"
  T0=$(date +%s)
  python "$ROOT/src/$SCRIPT" --model "$MODEL" 2>&1 | tee -a "$LOG_DIR/${MODEL}_${TIMESTAMP}.log"
  T1=$(date +%s)
  log "DONE   $SCRIPT --model $MODEL  ($(( T1 - T0 ))s)"
}

log "========================================================"
log "Datanauts Deadline Detection — Full Training Suite"
log "========================================================"

# ── Step 0: Build dataset if not already built ────────────────────────────────
if [ ! -d "$ROOT/data/deadline_sentences" ]; then
  log "Building sentence-level dataset..."
  python "$ROOT/src/build_dataset.py" 2>&1 | tee -a "$SUMMARY"
else
  log "Dataset already built — skipping build_dataset.py"
fi

# ── NER variants ──────────────────────────────────────────────────────────────
log "--- NER Training Runs ---"
run_variant train_ner.py baseline
run_variant train_ner.py bert_ner_v1
run_variant train_ner.py bert_ner_v2
run_variant train_ner.py bert_ner_v3
run_variant train_ner.py bert_ner_v4
run_variant train_ner.py bert_ner_v5
run_variant train_ner.py bert_base_cased

# ── Classifier variants ───────────────────────────────────────────────────────
log "--- Classifier Training Runs ---"
run_variant train_classifier.py baseline
run_variant train_classifier.py roberta_clf_v1
run_variant train_classifier.py roberta_clf_v2
run_variant train_classifier.py roberta_clf_v3
run_variant train_classifier.py roberta_clf_v4
run_variant train_classifier.py roberta_clf_v5
run_variant train_classifier.py roberta_clf_v6

# ── End-to-end evaluation (override with NER_MODEL_PATH / CLF_MODEL_PATH) ─────
log "--- End-to-End Evaluation ---"
NER_MODEL="${NER_MODEL_PATH:-/tmp/deadline-ner-bert_ner_v4}"
CLF_MODEL="${CLF_MODEL_PATH:-/tmp/deadline-clf-roberta_clf_v3}"
if [ -d "$NER_MODEL" ] && [ -d "$CLF_MODEL" ]; then
  python "$ROOT/src/evaluate.py" \
    --clf_model "$CLF_MODEL" \
    --ner_model "$NER_MODEL" \
    --threshold 0.7 \
    2>&1 | tee -a "$LOG_DIR/evaluate_${TIMESTAMP}.log"
  log "Evaluation complete — results logged to MLflow"
else
  log "Skipping evaluation: model dirs not found. Set NER_MODEL_PATH and CLF_MODEL_PATH."
fi

log "========================================================"
log "All runs complete. Summary: $SUMMARY"
log "MLflow UI: http://129.114.27.190:8000"
log ""
log "Next steps:"
log "  Feedback:  python src/feedback_loop.py --status"
log "  Predict:   python src/predict.py --clf_model \$CLF_MODEL --ner_model \$NER_MODEL --sentences 'sentence'"
log "========================================================"
