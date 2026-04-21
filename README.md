# Data Team – Intelligent Deadline & Expiry Detection

**Author:** Tanvi Takavane (tt2884) | **Team:** Datanauts – Data Team

---

> **⚠️ Note:** This branch captures the **initial implementation (Apr 6)** on the first dataset and storage container.
> All work here has been **re-implemented with an updated dataset and container** that contains both the initial (Apr 6) and final (Apr 20) implementations.
> 👉 See the latest: [`data/phase2-submission`](https://github.com/Bhanuu01/Datanauts-Intelligent-Deadline-Expiry-Detection/tree/data/phase2-submission)

---

## What This Branch Contains

Initial (Apr 6) data pipeline implementation for the Intelligent Deadline & Expiry Detection feature added to Paperless-ngx. When a document is uploaded, the system extracts OCR text, identifies date-bearing sentences via a fine-tuned NER model, and classifies whether each date is an actionable deadline. High-confidence detections create automatic reminders; lower-confidence ones prompt user confirmation.

---

## Initial Dataset & Storage (Apr 6)

| Resource | Link |
|---|---|
| HuggingFace Dataset | [tanvitakavane/datanauts_project_cuad-deadline-ner](https://huggingface.co/datasets/tanvitakavane/datanauts_project_cuad-deadline-ner) |
| Chameleon Object Storage | [cuad-data-proj11 @ CHI@TACC](https://chi.tacc.chameleoncloud.org/project/containers/container/cuad-data-proj11) |

**633 labeled training samples:**
- 133 real contract expiration clauses from [CUAD](https://huggingface.co/datasets/theatticusproject/cuad) with explicit date tokens
- 500 synthetic modern-format (2020–2027) invoices generated with Faker + ReportLab

---

## ML Models

| Model | Base | Task |
|---|---|---|
| NER | `dslim/bert-base-NER` (fine-tuned) | Date span extraction (IOB2: `B-DATE`, `I-DATE`, `O`) |
| Classifier | `roberta-base` (fine-tuned) | Event type: `expiration` / `payment_due` / `effective` / `agreement` |

---

## Data Repositories

**Chameleon Object Storage (`cuad-data-proj11`)** — Raw CUAD CSV, cleaned NER JSONL samples, train/test splits, and versioned batch pipeline outputs. Each run writes to `versioned/v<YYYYMMDD_HHMMSS>/` with a `manifest.json` (pipeline version, SHA256 source hash, sample counts, `frozen_test: true`).

**HuggingFace Dataset** — `train.jsonl`, `test.jsonl`, and pipeline scripts. Git-backed; every push creates a commit SHA recorded in model metadata for reproducibility. Load via:
```python
load_dataset('tanvitakavane/datanauts_project_cuad-deadline-ner')
```

**Redis Feature Store** — Pre-computed NER input features per document keyed by `document_id` (TTL: 1 hour). Written by FastAPI feature service on document ingest; read by the Serving team at `GET /features/<doc_id>`.

**PostgreSQL Feedback DB** *(Planned)* — Append-only user feedback events (confirm / edit / dismiss / manual_add) feeding monthly batch retraining.

---

## Training Data Schema

| Field | Type | Description |
|---|---|---|
| `filename` | string | Source PDF filename |
| `event_type` | string | `expiration` or `payment_due` |
| `sentence` | string | OCR sentence containing a date |
| `tokens` | string[] | Whitespace-split tokens |
| `ner_labels` | string[] | IOB2 tag per token |
| `ground_truth_date` | string | ISO 8601 date (`YYYY-MM-DD`) |
| `agreement_date_raw` | string | Raw CUAD date — used for time-based split only |
| `_version` | string | Batch pipeline version tag |
| `_source_hash` | string | SHA256 prefix of source file |

