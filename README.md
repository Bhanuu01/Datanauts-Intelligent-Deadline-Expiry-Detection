# Datanauts Intelligent Deadline Detection

Integrated MLOps system for deadline extraction inside Paperless-ngx, deployed on Kubernetes.

This repository is the main integrated codebase for the team project. It contains:

- the Paperless post-consume ML integration
- ONNX serving for deadline detection
- online features and synthetic production traffic generation
- retraining, promotion, and release automation
- MLflow and MinIO platform services
- Prometheus and Grafana monitoring
- Kubernetes manifests for `paperless`, `platform`, `ml`, `monitoring`, and release environments

## What This System Does

The deployed system takes documents uploaded into Paperless-ngx, extracts OCR text, runs deadline detection, writes the result back into Paperless as tags, stores prediction artifacts, records feedback, and supports retraining and release automation.

Main flow:

1. A user uploads a document into Paperless.
2. Paperless OCR/parsing runs.
3. A post-consume hook calls the ONNX deadline detection service.
4. The prediction result is saved to the shared volume.
5. Paperless adds ML tags such as:
   - `Type:Deadline`
   - `Deadline:YYYY-MM-DD`
   - `Type:Effective`
   - `Effective:YYYY-MM-DD`
   - `Status:Review Needed`
6. Users can review the result in Paperless using:
   - `Action:Accept`
   - `Action:Reject`
7. Feedback and production traffic are logged for retraining.
8. Retraining and model promotion jobs run inside Kubernetes.

## Repository Layout

```text
components/
  data/
    data_generator/            Synthetic production traffic generator
    evaluation_monitoring/     Data quality and drift jobs
    online_features/           Ingest + feedback capture service
  paperless_hooks/             Paperless post-consume integration
  platform_automation/         Retrain, evaluation, promotion, release scripts
  serving/                     ONNX serving runtime and export/quantization helpers
  training/                    Training code for classifier and NER models

k8s/
  paperless/                   Paperless, Postgres, Redis
  platform/                    MLflow, MinIO, MLflow Postgres
  ml/                          ONNX serving, online-features, generator, cronjobs
  monitoring/                  Prometheus, Grafana, kube-state-metrics
  release/                     Staging / canary / production serving layer

scripts/
  provision.sh                 Base cluster bootstrap
  create-secrets.sh            Secret creation
  rebuild-k3s-images.sh        Build and import local images into k3s
  chameleon-health-check.sh    Cluster sanity check
```

## Branches

- `main`: current integrated branch and recommended source of truth
- `paperless-ngx-import`: snapshot branch containing imported upstream Paperless code under `third_party/`

If you are reproducing or extending the current system, use `main`.

## Prerequisites

Recommended environment:

- Ubuntu VM on Chameleon
- single-node `k3s`
- Docker installed on the node
- enough disk and RAM for local image builds

Useful baseline:

- 1 node
- 48 vCPU
- 240 GiB RAM
- at least 40 GiB free root disk recommended before rebuilding images

## One-Time Cluster Setup

### 1. Clone the repository

```bash
git clone https://github.com/Bhanuu01/Datanauts-Intelligent-Deadline-Expiry-Detection.git
cd Datanauts-Intelligent-Deadline-Expiry-Detection
git checkout main
```

### 2. Install k3s

```bash
curl -sfL https://get.k3s.io | sh -
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$(id -u)":"$(id -g)" ~/.kube/config
export KUBECONFIG=~/.kube/config
echo 'export KUBECONFIG=~/.kube/config' >> ~/.zshrc
```

Verify:

```bash
kubectl get nodes
```

### 3. Create namespaces

```bash
kubectl apply -f k8s/namespace-paperless.yaml
kubectl apply -f k8s/namespace-platform.yaml
kubectl apply -f k8s/namespace-ml.yaml
kubectl apply -f k8s/monitoring/namespace.yaml
```

### 4. Create secrets

Run once:

```bash
bash scripts/create-secrets.sh
```

This creates:

- `paperless-secrets` in `paperless`
- mirrored `paperless-secrets` in `ml`
- `platform-secrets` in `platform`
- mirrored `platform-secrets` in `ml`

Store the generated credentials securely after running the script. They are injected into Kubernetes Secrets and are not intended to be committed to Git.

## Build and Import Local Images

This project depends on locally built images that must be imported into the `k3s` container runtime.

Build everything:

```bash
./scripts/rebuild-k3s-images.sh
```

Build only one image family:

```bash
./scripts/rebuild-k3s-images.sh onnx-serving
./scripts/rebuild-k3s-images.sh platform-automation
./scripts/rebuild-k3s-images.sh data-monitoring
```

The script builds Docker images and imports them into `k3s` using:

- `ghcr.io/bhanuu01/datanauts-online-features:latest`
- `ghcr.io/bhanuu01/datanauts-data-generator:latest`
- `ghcr.io/bhanuu01/datanauts-data-monitoring:latest`
- `ghcr.io/bhanuu01/datanauts-platform-automation:latest`
- `ghcr.io/bhanuu01/datanauts-inference-runtime:latest`
- `ghcr.io/bhanuu01/datanauts-onnx-serving:latest`

## Deploy the System

### 1. Paperless

```bash
kubectl apply -f k8s/paperless/
```

### 2. Platform Services

```bash
kubectl apply -f k8s/platform/
```

### 3. Monitoring

```bash
kubectl apply -k k8s/monitoring
```

### 4. ML Services

```bash
kubectl apply -k k8s/ml
```

### 5. Release Layer

```bash
kubectl apply -k k8s/release
```

### 6. Verify rollouts

```bash
kubectl rollout status deployment/paperless-ngx -n paperless --timeout=300s
kubectl rollout status deployment/mlflow -n platform --timeout=300s
kubectl rollout status deployment/minio -n platform --timeout=300s
kubectl rollout status deployment/grafana -n monitoring --timeout=300s
kubectl rollout status deployment/prometheus -n monitoring --timeout=300s
kubectl rollout status deployment/online-features -n ml --timeout=300s
kubectl rollout status deployment/deadline-onnx-serving -n ml --timeout=300s
```

## Access URLs

Current Chameleon deployment:

- Paperless: [http://129.114.27.190](http://129.114.27.190)
- MLflow: [http://129.114.27.190:30500](http://129.114.27.190:30500)
- MinIO Console: [http://129.114.27.190:30901](http://129.114.27.190:30901)
- MinIO S3 API: [http://129.114.27.190:30900](http://129.114.27.190:30900)
- Grafana: [http://129.114.27.190/grafana/login](http://129.114.27.190/grafana/login)
- Prometheus: [http://129.114.27.190/prometheus/graph](http://129.114.27.190/prometheus/graph)

If you deploy on another node, replace the IP with your node’s public address.

## Access and Credentials

Public service endpoints:

### Paperless

- URL: [http://129.114.27.190](http://129.114.27.190)

### MLflow

- URL: [http://129.114.27.190:30500](http://129.114.27.190:30500)

### MinIO Console

- URL: [http://129.114.27.190:30901](http://129.114.27.190:30901)

### MinIO S3 API

- Endpoint: [http://129.114.27.190:30900](http://129.114.27.190:30900)

### Grafana

- URL: [http://129.114.27.190/grafana/login](http://129.114.27.190/grafana/login)

### Prometheus

- URL: [http://129.114.27.190/prometheus/graph](http://129.114.27.190/prometheus/graph)

Retrieve credentials from Kubernetes Secrets instead of storing them in documentation. Example commands:

```bash
kubectl get secret -n paperless paperless-secrets -o yaml
kubectl get secret -n platform platform-secrets -o yaml
```

To print a decoded value safely when needed:

```bash
kubectl get secret -n paperless paperless-secrets -o jsonpath='{.data.PAPERLESS_ADMIN_PASSWORD}' | base64 --decode && echo
kubectl get secret -n platform platform-secrets -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 --decode && echo
```

## Internal Service Endpoints

Inside the cluster, use:

- Paperless API: `http://paperless-ngx.paperless.svc.cluster.local:8000`
- Online features: `http://online-features.ml.svc.cluster.local:8000`
- ONNX serving: `http://deadline-onnx-serving.ml.svc.cluster.local:8004`
- MLflow: `http://mlflow.platform.svc.cluster.local:5000`
- MinIO: `http://minio.platform.svc.cluster.local:9000`
- Prometheus: `http://prometheus.monitoring.svc.cluster.local:9090/prometheus`

## How to Run the Live System

### Cluster health check

```bash
export KUBECONFIG=~/.kube/config
bash scripts/chameleon-health-check.sh
```

### Current pod view

```bash
kubectl get pods -n paperless
kubectl get pods -n platform
kubectl get pods -n monitoring
kubectl get pods -n ml
```

### Data generator logs

```bash
kubectl logs -n ml deploy/data-generator --tail=50
```

### ONNX serving logs

```bash
kubectl logs -n ml deploy/deadline-onnx-serving --tail=100
```

### Paperless logs

```bash
kubectl logs -n paperless deploy/paperless-ngx --tail=150
```

## How to Use the Product

### Upload a document in Paperless

1. Open Paperless.
2. Upload a `.txt` or `.pdf`.
3. Wait for processing.
4. Open the document details page.

Expected ML tags on a positive example:

- `ML Deadline Detected`
- `ML Deadline Date: YYYY-MM-DD`
- `ML Review Pending`
- `ML Event: Effective`

Feedback tags available in the tag picker:

- `ML Feedback Correct`
- `ML Feedback Wrong`

### Result files

Prediction artifacts are written here inside Paperless:

```bash
kubectl exec -n paperless deploy/paperless-ngx -- sh -c 'ls -lah /usr/src/paperless/data/deadline-results'
```

Inspect results:

```bash
kubectl exec -n paperless deploy/paperless-ngx -- sh -c 'tail -n 200 /usr/src/paperless/data/deadline-results/*.json'
```

### Production traffic and feedback logs

```bash
kubectl exec -n ml deploy/online-features -- sh -c 'tail -n 20 /data/production_ingest.jsonl'
kubectl exec -n ml deploy/online-features -- sh -c 'tail -n 20 /data/feedback_events.jsonl'
```

## Direct Service Testing

### ONNX serving

```bash
kubectl port-forward -n ml svc/deadline-onnx-serving 18005:8004
```

Then:

```bash
curl -X POST http://127.0.0.1:18005/predict \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "demo-1",
    "ocr_text": "This agreement is effective as of April 20, 2026. Written notice must be delivered by May 20, 2026.",
    "document_type": "contract",
    "filename": "demo.txt"
  }'
```

### Online features feedback

```bash
kubectl port-forward -n ml svc/online-features 18000:8000
```

Then:

```bash
curl -X POST http://127.0.0.1:18000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "event": "confirm",
    "document_id": "demo-1",
    "event_type": "effective",
    "confidence": 0.9
  }'
```

## Retraining, Promotion, and Release

CronJobs:

- `retrain-pipeline`
- `model-promotion-gate`
- `release-promotion`
- `data-quality-checks`
- `drift-monitor`

List them:

```bash
kubectl get cronjobs -n ml
kubectl get jobs -n ml
```

Manual reruns:

```bash
kubectl create job --from=cronjob/retrain-pipeline retrain-manual -n ml
kubectl create job --from=cronjob/model-promotion-gate model-promotion-manual -n ml
kubectl create job --from=cronjob/release-promotion release-promotion-manual -n ml
```

View retrain logs:

```bash
kubectl logs -n ml job/retrain-manual --tail=200
```

Decision files:

```bash
kubectl exec -n ml deploy/online-features -- sh -c 'ls -lah /data && echo --- && cat /data/retrain_decision.json && echo --- && cat /data/promotion_decision.json'
```

## Monitoring

### Grafana

Open one of these Grafana dashboards:

- `Datanauts Overview`
- `Datanauts Serving`
- `Datanauts Data & Feedback`
- `Datanauts Platform Health`

Useful panels:

- serving targets up
- ONNX predictions/sec
- online-features throughput
- feedback/review activity
- platform and monitoring memory utilization

### Prometheus

Prometheus is mainly used for:

- Targets page
- Alerts page
- raw query debugging

Useful queries:

```promql
up
deadline_onnx_predictions_total
online_features_ingest_requests_total
```

## Current Known Limitations

- PDF deadline extraction is not perfect.
- Some non-contract documents are outside the strongest model domain.
- The UI review flow currently uses Paperless tags, not custom accept/reject buttons.
- Some deadline dates may still be partially normalized if the model returns poor date spans.

## Troubleshooting

### `kubectl` permission errors

If you see:

```text
error loading config file "/etc/rancher/k3s/k3s.yaml": permission denied
```

run:

```bash
export KUBECONFIG=~/.kube/config
```

### Services up but old error pods still visible

Clean stale pods:

```bash
kubectl delete pod -n paperless --field-selector=status.phase=Succeeded --ignore-not-found
kubectl delete pod -n platform --field-selector=status.phase=Succeeded --ignore-not-found
kubectl delete pod -n monitoring --field-selector=status.phase=Succeeded --ignore-not-found
kubectl delete pod -n ml --field-selector=status.phase=Succeeded --ignore-not-found
```

### Local images missing after node restart

Rebuild and re-import:

```bash
./scripts/rebuild-k3s-images.sh
```

Then restart the affected deployment:

```bash
kubectl rollout restart deployment/deadline-onnx-serving -n ml
kubectl rollout restart deployment/online-features -n ml
kubectl rollout restart deployment/paperless-ngx -n paperless
```

### Disk pressure on the node

Check disk:

```bash
df -h /
```

This system depends on local image builds. If the root disk fills up, pods can be evicted and local images can disappear from k3s.

## Quick Demo Checklist

1. Show `kubectl get pods` for all namespaces.
2. Open Paperless and upload a document.
3. Show ML tags on the Paperless document.
4. Add `ML Feedback Correct` or `ML Feedback Wrong`.
5. Show `/data/production_ingest.jsonl` and `/data/feedback_events.jsonl`.
6. Show retrain/promotion CronJobs and decision files.
7. Show Grafana `Datanauts Overview` and at least one role-specific dashboard.
8. Show Prometheus targets or alerts.

## Source of Truth

Use branch `main`.

If you are deploying this system again from scratch, clone `main`, create secrets, rebuild/import the images, and apply the Kubernetes manifests in the order described above.
