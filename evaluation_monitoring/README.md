# Evaluation & Monitoring — Final Implementation (Apr 20)

**Student:** Tanvi Takavane (tt2884) | Datanauts | NYU
**Dataset:** tanvitakavane/datanauts_project_cuad-deadline-ner-version2
**Bucket:** cuad-data-proj11-v2

## Overview
This is the FINAL implementation of the data evaluation and monitoring pipeline.
All 3 required evaluation points are implemented and PASSED.

## Evaluation Points

| Point | Script | Result |
|---|---|---|
| EP1 — Ingestion Quality Check | `ingestion_quality_check.py` | ✅ PASSED 12/12 |
| EP2 — Training Set Quality Check | `training_set_quality_check.py` | ✅ PASSED 11/11 |
| EP3 — Production Drift Monitoring | `drift_monitor.py` | ✅ PASSED 5/5 |

## How to Run
```bash
source ~/openrc.sh
docker compose run --rm ingestion-check
docker compose run --rm training-quality-check
docker compose run --rm drift-monitor
```

## Results
All evaluation reports are uploaded to Chameleon bucket cuad-data-proj11-v2:
- eval_output/ingestion_quality_report.json
- eval_output/training_set_quality_report.json
- eval_output/drift_monitoring_report.json
