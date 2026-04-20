# Datanauts — Intelligent Deadline & Expiry Detection

An MLOps pipeline that automatically detects deadlines, expiration dates, and renewal terms from contract text. This repository houses the **Serving, Monitoring, and Deployment** infrastructure for the project, featuring a highly optimized, two-stage ONNX+INT8 inference engine.

---

## 🏗 System Architecture

The inference pipeline utilizes a two-stage NLP architecture to maximize both speed and accuracy:
1. **Gatekeeper (RoBERTa):** A sentence-level sequence classifier that filters out non-relevant text.
2. **Extractor (BERT):** A token-level NER model that extracts exact date spans from the filtered sentences.

```text
Raw contract text (OCR / Paperless-ngx)
        │
        ▼
   predict.py (app_onnx_quant.py)
        │
        ├──► STAGE 1: RoBERTa Sequence Classifier (ONNX+INT8)
        │       └─ Filters sentences (e.g., Expiration, Renewal, None)
        │
        └──► STAGE 2: BERT Token Classifier (ONNX+INT8)
                └─ Extracts exact date bounds (NER) from positive sentences
        │
        ▼
 Structured JSON Output (event_type, date, confidence, uncertain_flag)
        │
        ├──► Prometheus Metrics (/metrics) ─► Grafana Dashboard
        │
        └──► Feedback Loop (/feedback) ────► Data Team DB (Retraining)