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
run_variant train_ner.py bert_base_cased

# ── Classifier variants ───────────────────────────────────────────────────────
log "--- Classifier Training Runs ---"
run_variant train_classifier.py baseline
run_variant train_classifier.py roberta_clf_v1
run_variant train_classifier.py roberta_clf_v2
run_variant train_classifier.py roberta_clf_v3

log "========================================================"
log "All runs complete. Summary saved to: $SUMMARY"
log "MLflow UI: http://129.114.27.190:8000"
log "========================================================"
