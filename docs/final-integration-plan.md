# Final Integration Workstream

This branch turns the project from role-separated milestone work into a single deployment-oriented layout.

## Added in this branch

- `components/data`: imported from `origin/data/phase2-submission`
- `components/training`: imported from `origin/training/phase2-submission`
- `components/inference_service`: internal API that wraps the training inference flow and provides a fallback mode when model artifacts are not mounted yet
- `components/platform_automation`: platform-owned automation scripts for retrain checks and promotion gates
- `k8s/ml`: Kubernetes manifests for the ML namespace, model storage, online features, inference, and scheduled automation jobs
- `k8s/monitoring`: Prometheus, Grafana, kube-state-metrics, scrape config, and basic alert rules
- `components/paperless_hooks`: Paperless post-consume integration that calls the inference service and tags processed documents
- `scripts/chameleon-health-check.sh`: one-command cluster health snapshot for demos and recovery
- `scripts/rebuild-k3s-images.sh`: rebuild/import helper for locally managed images inside the single-node k3s cluster

## Intended deployment flow

1. Paperless ingests OCR text.
2. `online-features` prepares candidate deadline sentences.
3. `deadline-inference` runs model inference or a fallback extraction path.
4. Feedback metrics are written to shared storage.
5. `retrain-pipeline` evaluates thresholds and launches the training scripts on schedule.
6. Data quality and drift jobs run on their own cadence.

## Current integrated state

1. Paperless, Postgres, Redis, MinIO, and MLflow are deployed in Kubernetes on Chameleon.
2. Paperless document ingestion triggers a post-consume hook that calls `deadline-inference`.
3. `online-features` and `deadline-inference` expose `/metrics` and are scraped by Prometheus.
4. Retrain, promotion, drift-monitor, and data-quality jobs are defined as Kubernetes CronJobs.
5. Model artifacts and monitoring inputs are stored on shared persistent volume storage.

## Remaining polish

1. Capture a demo-ready Grafana dashboard view for inference latency, request volume, and pod health.
2. Rehearse the end-to-end upload flow on 2-3 representative documents for the final presentation.
3. Align the promotion and rollback story with the training and serving teammates' final thresholds.
4. Keep the custom k3s images fresh on the node with `scripts/rebuild-k3s-images.sh` if the cluster is restarted.
