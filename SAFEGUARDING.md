# Safeguarding Plan — Data Team
**Student:** Tanvi Takavane (tt2884) | Datanauts | NYU
**Branch:** data/phase2-submission
**Date:** April 20, 2026

## Overview
The data pipeline implements safeguarding across 4 concrete mechanisms covering fairness, transparency, privacy, accountability, and robustness principles.

---

## 1. Leakage Prevention (Robustness + Fairness)

**Mechanism:** Time-based split with contract-level grouping in the batch pipeline.

**Implementation (`batch_pipeline/batch_pipeline.py`):**
- Contracts split by `agreement_date_iso` — pre-2005 to train, 2005-2010 to validation, 2011+ to test
- All clauses from one contract always go to the same split
- No clause from a post-2010 contract ever appears in train
- Test set is frozen — never used for retraining (`frozen_test: true` in manifest)

**Why it matters:** Prevents the model from being evaluated on data it has effectively "seen" during training, ensuring honest performance estimates and fair evaluation.

---

## 2. Data Quality Gates (Robustness + Accountability)

**Mechanism:** Three-point Great Expectations validation at ingestion, training, and inference.

**Implementation (`evaluation_monitoring/`):**

| Gate | Script | What it checks |
|---|---|---|
| EP1 — Ingestion | `ingestion_quality_check.py` | Schema validity, date formats, null rates, event type validity |
| EP2 — Training | `training_set_quality_check.py` | No train/test overlap, class balance, split ratios |
| EP3 — Drift | `drift_monitor.py` | OCR length drift, event type distribution shift, null rate stability |

**Thresholds:**
- OCR length drift ratio > 2.0 → pipeline alert
- Train/test overlap > 0 → pipeline stops (data leakage)
- Empty OCR text > 5% → warning
- Event type drift > 0.4 → warning

**Why it matters:** Catches bad data before it reaches the model, ensuring accountability at every stage of the pipeline.

---

## 3. Versioned Datasets with Manifest Tracking (Transparency + Accountability)

**Mechanism:** Every batch pipeline run produces a versioned output with a manifest.

**Implementation (`batch_pipeline/batch_pipeline.py`):**
- Every run tagged with timestamp: `versioned/v20260419_182857/`
- `manifest.json` records:
  - `source_hash` — SHA256 of input data
  - `pipeline_run` — exact timestamp
  - `counts` — train/val/test record counts
  - `split_logic` — human-readable split rule
  - `frozen_test: true` — confirms test set integrity
  - `synthetic_data: false` — confirms no synthetic data used

**Why it matters:** Any model training run can be traced back to the exact data snapshot used. Supports full reproducibility and auditability.

---

## 4. No Synthetic Data — Real CUAD Contracts Only (Fairness + Transparency)

**Mechanism:** Dataset uses only real legal contracts from the CUAD dataset.

**Implementation:**
- Dataset: `tanvitakavane/datanauts_project_cuad-deadline-ner-version2`
- 510 real contracts from CUAD (Contract Understanding Atticus Dataset)
- `synthetic_data: false` explicitly recorded in every manifest
- No Faker-generated or LLM-generated training samples

**Why it matters:** Models trained on real contracts reflect actual legal language. Synthetic data can introduce bias or unrealistic patterns that hurt model fairness on real documents.

---

## Summary

| Principle | Mechanism |
|---|---|
| Fairness | Time-based split prevents leakage; real data only |
| Transparency | Manifest tracking with source hash; no synthetic data flag |
| Accountability | 3-point GX quality gates; versioned outputs |
| Robustness | Drift monitoring; frozen test set; contract-level grouping |
| Privacy | No PII in training data — CUAD contracts are public legal documents |

