# Final Integration Workstream

This branch turns the project from role-separated milestone work into a single deployment-oriented layout.

## Added in this branch

- `components/data`: imported from `origin/data/phase2-submission`
- `components/training`: imported from `origin/training/phase2-submission`
- `components/inference_service`: internal API that wraps the training inference flow and provides a fallback mode when model artifacts are not mounted yet
- `components/platform_automation`: platform-owned automation scripts for retrain checks and promotion gates
- `k8s/ml`: Kubernetes manifests for the ML namespace, model storage, online features, inference, and scheduled automation jobs

## Intended deployment flow

1. Paperless ingests OCR text.
2. `online-features` prepares candidate deadline sentences.
3. `deadline-inference` runs model inference or a fallback extraction path.
4. Feedback metrics are written to shared storage.
5. `retrain-pipeline` evaluates thresholds and launches the training scripts on schedule.
6. Data quality and drift jobs run on their own cadence.

## Immediate next steps

1. Build and publish container images referenced in `k8s/ml/*.yaml`.
2. Mount trained model artifacts into `model-storage-pvc`.
3. Add the serving teammate's final API or route Paperless directly to `deadline-inference`.
4. Patch Paperless upload flow so the complementary ML feature is exercised in the normal user path.
5. Add Prometheus/Grafana manifests and alert rules for queue depth, pod health, and inference failures.
