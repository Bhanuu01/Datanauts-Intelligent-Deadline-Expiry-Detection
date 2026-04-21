# Data Team — Tanvi Takavane (tt2884) | Datanauts | NYU
**Branch:** `data/phase2-submission` | **Project:** proj11 | **Lease:** data_final_proj11_tt2884

---

## Overview

This branch contains the complete data team implementation for the Intelligent Deadline & Expiry Detection system built on Paperless-ngx. It includes all work from Initial Implementation (Apr 6) that is re-implemented with the updated dataset and Final System Implementation (Apr 20).

---

## Dataset

**HuggingFace:** https://huggingface.co/datasets/tanvitakavane/datanauts_project_cuad-deadline-ner-version2

**Chameleon Bucket (v2 — Active):** https://chi.tacc.chameleoncloud.org/project/containers/container/cuad-data-proj11-v2

### What Changed from v1 to v2

The dataset was updated based on feedback:

- **6 event types** (up from 4): `none`, `expiration`, `effective`, `renewal`, `agreement`, `notice_period`
- **No synthetic data** — only real CUAD contracts (510 contracts total)
- **Time-based contract-level split** (replaces sentence-level split):
  - pre-2005 → train (373 records)
  - 2005–2010 → validation (68 records)
  - 2011+ → test (69 records)
- **Leakage prevention**: All clauses from one contract always go to the same split. No clause from a post-2010 contract appears in train. This prevents the model from being evaluated on contracts it has effectively seen during training.

---

## Repository Structure

| Folder | Question | Description |
|---|---|---|
| `object_storage/` | Q2.2 | Live object storage bucket — cuad-data-proj11-v2 |
| `pipeline/` | Q2.3 | Ingestion pipeline — HuggingFace to Chameleon bucket |
| `data_generator/` | Q2.4 | Data generator — simulates live contract uploads with 6 event types |
| `online_features/` | Q2.5 | Online feature service — FastAPI + Redis |
| `batch_pipeline/` | Q2.6 | Batch pipeline — versioned datasets with manifest |
| `gx_quality/` | Q3 | Great Expectations validation — bonus |
| `evaluation_monitoring/` | Final (Apr 20) | Final evaluation & monitoring — all 3 points PASSED |
| `SAFEGUARDING.md` | Final (Apr 20) | Safeguarding plan — leakage prevention, data quality gates, versioning |

---

## Implementation Timeline

### Initial Implementation (Apr 6) that is re-implementated with updated dataset
- Q2.1: High-level data design document uploaded as `high_level_data_design_tt2884.pdf` — updated for v2 dataset and cuad-data-proj11-v2 bucket
- Q2.2: Created Chameleon object storage bucket with CUAD dataset files
- Q2.3: Built Docker ingestion pipeline fetching from HuggingFace
- Q2.4: Built data generator simulating contract upload events
- Q2.5: Built FastAPI + Redis online feature service
- Q2.6: Built batch pipeline with versioned dataset outputs


All of the above components were re-implemented with the new v2 dataset:
- New container `cuad-data-proj11-v2` created with updated files
- Pipeline updated to handle parquet → JSONL conversion
- Batch pipeline updated to use v2 column names (`Filename`, `ocr_text`, `agreement_date_iso`)
- Time-based contract-level split implemented for leakage prevention

### Final System Implementation (Apr 20)
- **EP1 — Ingestion Quality Check**: 12/12 checks PASSED
- **EP2 — Training Set Quality Check**: 11/11 checks PASSED
- **EP3 — Production Drift Monitoring**: 5/5 checks PASSED
- Safeguarding plan documented in `SAFEGUARDING.md`

---

## Key Design Decisions

**Time-based contract-level split vs sentence-level split:**
Previous implementation used a random sentence-level split which risked data leakage — sentences from the same contract could appear in both train and test. The new implementation uses a time-based contract-level split anchored at `agreement_date_iso`. All sentences from one contract are assigned to the same split, and the split boundary is time-based (not random). This ensures the model is evaluated on contracts from a different time period than it was trained on, giving an honest estimate of real-world performance.

**No synthetic data:**
The v2 dataset uses only real CUAD contracts. This ensures the model learns from authentic legal language rather than artificially generated text.

**Versioned datasets:**
Every batch pipeline run produces a versioned output with a manifest recording the source hash, split counts, and split logic. This allows any training run to be reproduced from the exact data snapshot used.

---

## How to Run

```bash
# Setup
source ~/openrc.sh

# Q2.3 Ingestion Pipeline
cd pipeline && docker compose run --rm ingest-pipeline

# Q2.4 Data Generator
cd data_generator && docker compose up --build

# Q2.5 Online Feature Service
cd online_features && docker compose up --build -d
curl http://localhost:8000/health

# Q2.6 Batch Pipeline
cd batch_pipeline && docker compose run --rm batch-pipeline

# Final Evaluation & Monitoring
cd evaluation_monitoring
docker compose run --rm ingestion-check
docker compose run --rm training-quality-check
docker compose run --rm drift-monitor
```

---

**Tanvi Takavane | tt2884 | Datanauts | NYU | Data Team | April 2026**
