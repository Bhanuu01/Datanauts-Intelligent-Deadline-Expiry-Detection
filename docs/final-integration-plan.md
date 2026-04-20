# Final Integration Workstream

This branch turns the project from role-separated milestone work into a single deployment-oriented layout.

## Added in this branch

- `components/data`: imported from `origin/data/phase2-submission`
- `components/training`: imported from `origin/training/phase2-submission`
- `components/inference_service`: legacy API kept for historical reference while the integrated deployment uses the ONNX serving path
- `components/platform_automation`: platform-owned automation scripts for retrain checks and promotion gates
- `components/serving`: serving teammate's quantized ONNX path, adapted to shared model storage and feedback logging
- `k8s/ml`: Kubernetes manifests for the ML namespace, model storage, online features, ONNX serving, and scheduled automation jobs
- `k8s/release`: staging, canary, production ONNX serving manifests plus release-promotion automation
- `k8s/monitoring`: Prometheus, Grafana, kube-state-metrics, scrape config, and basic alert rules
- `components/paperless_hooks`: Paperless post-consume integration that calls the inference service and tags processed documents
- `scripts/chameleon-health-check.sh`: one-command cluster health snapshot for demos and recovery
- `scripts/rebuild-k3s-images.sh`: rebuild/import helper for locally managed images inside the single-node k3s cluster
- automated Docker cleanup after image import to prevent the Chameleon node from re-entering disk pressure during rebuild cycles

## Intended deployment flow

1. Paperless ingests OCR text.
2. `online-features` prepares candidate deadline sentences and writes production ingest records to shared storage.
3. `deadline-onnx-serving` runs the live two-stage ONNX model path and persists serving feedback on the shared volume.
4. The Paperless post-consume hook writes feedback events to the shared `online-features` feedback endpoint; the synthetic generator writes the same event format to emulate user confirmations/edits/dismissals.
5. `retrain-pipeline` evaluates thresholds and launches the training scripts on schedule.
6. Data quality and drift jobs run on their own cadence.
7. A separate release layer runs staging, canary, and production ONNX serving deployments in parallel.
8. The release-promotion job can patch the canonical `deadline-onnx-serving` service to point at the next approved release channel.

## Current integrated state

1. Paperless, Postgres, Redis, MinIO, and MLflow are deployed in Kubernetes on Chameleon.
2. Paperless document ingestion triggers a post-consume hook that calls `deadline-onnx-serving`.
3. `online-features` and `deadline-onnx-serving` expose `/metrics` and are scraped by Prometheus.
4. Retrain, promotion, drift-monitor, and data-quality jobs are defined as Kubernetes CronJobs.
5. Staging, canary, and production inference deployments are defined in `k8s/release` and expose release-aware `/health` responses.
6. Model artifacts, production-ingest logs, prediction summaries, and feedback events are stored on shared persistent volume storage.
7. Prometheus alerts cover service outages, pod health, image pull failures, inference failures, latency, and node disk pressure.

## Team-owned production contracts

- Data pipeline entrypoints
  - ingestion: `components/data/pipeline/scripts/run_pipeline.sh`
  - batch compilation: `components/data/batch_pipeline/batch_pipeline.py`
  - feedback/data generator: `components/data/data_generator/generator.py`
  - online features: `components/data/online_features/feature_service.py`
  - quality/drift checks: `components/data/evaluation_monitoring/*.py`
- Data monitoring thresholds
  - OCR length drift ratio `> 2.0`: alert
  - train/test overlap `> 0`: alert
  - null filenames `> 0`: alert
  - event type drift `> 0.4`: warning
  - empty OCR text `> 5%`: warning
- Training-owned promotion gate
  - NER F1 `>= 0.65`
  - classifier macro F1 `>= 0.75`
  - end-to-end coverage `>= 0.60`
  - false alarm count `<= 10`
- Current training artifacts
  - `/tmp/deadline-ner-bert_ner_v5`
  - `/tmp/deadline-clf-roberta_clf_v5`
- Current evaluation command
  - `python src/evaluate.py --clf_model <path> --ner_model <path> --threshold 0.7`
- Serving-owned live path
  - `components/serving/app_onnx_quant.py`
  - K8s manifests: `k8s/ml/onnx-serving-deployment.yaml`, `k8s/ml/onnx-serving-service.yaml`
  - expected model location: `/models/onnx_quantized_model`
- Release progression
  - staging: `deadline-onnx-serving-staging`
  - canary: `deadline-onnx-serving-canary`
  - production: `deadline-onnx-serving-production`
  - promotion planner: `components/platform_automation/promote_release.py`
  - live service patch target: `deadline-onnx-serving`
  - release manifests: `k8s/release/kustomization.yaml`

## DevOps Completion Notes

1. CI validation exists in `.github/workflows/integration-ci.yml` and checks the integrated services plus release manifests.
2. Release automation now supports `promote` and `rollback` actions and can patch the canonical ONNX serving service selector inside Kubernetes.
3. Alerting covers the main failure mode encountered during integration: node `DiskPressure` leading to evicted pods and image-pull failures.
4. The rebuild/import helper now prunes Docker artifacts after import so repeated image refreshes do not consume the node's local disk indefinitely.
5. Paperless now exercises the same `online-features -> inference -> feedback-log` path used by scheduled retraining and the synthetic data generator.
6. Final demo prep should focus on healthy pod state, one live Paperless upload, Prometheus/Grafana views, shared JSONL logs on the PVC, and the release-promotion JSON plus service-selector story.
