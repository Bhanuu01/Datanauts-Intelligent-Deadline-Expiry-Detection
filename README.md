# Datanauts Intelligent Deadline Detection

Integrated MLOps system for deadline detection inside Paperless-ngx, deployed on Kubernetes on Chameleon.

This branch, `devops/final-hardening`, is the current demo-ready branch used for the integrated system, monitoring fixes, secret handling updates, and Paperless deadline-tag flow.

## Overview

The system adds an ML-powered deadline feature directly into the normal Paperless-ngx workflow.

When a document is ingested:

1. Paperless OCR/parsing runs.
2. A Paperless post-consume hook sends OCR text to the ONNX serving service.
3. ONNX serving returns deadline-related events and extracted date candidates.
4. The hook writes prediction artifacts to Paperless storage.
5. Paperless updates the document with ML-generated tags.
6. Feedback is captured with Paperless tags and logged for retraining.
7. Retraining, gating, and release automation run as Kubernetes jobs and cronjobs.

## What You Should See In Paperless

For positive examples, Paperless should add tags such as:

- `Type:Deadline`
- `Deadline:YYYY-MM-DD`
- `Type:Effective`
- `Effective:YYYY-MM-DD`
- `Type:Renewal`
- `Status:Review Needed`

Reviewer feedback is captured with:

- `Action:Accept`
- `Action:Reject`

If a document repeats multiple notice blocks, you may see multiple `Deadline:*` and `Effective:*` tags on the same document. That is expected behavior.

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
  ml/                          ONNX serving, online-features, data generator, cronjobs
  monitoring/                  Prometheus, Grafana, kube-state-metrics
  release/                     Staging / canary / production serving layer
  sealed-secrets/              Optional sealed-secret manifests

scripts/
  provision.sh                 Cluster bootstrap helper
  create-secrets.sh            Secret creation
  sync-runtime-secrets.sh      Mirrors runtime secrets into ml namespace
  rebuild-k3s-images.sh        Build and import local images into k3s
  chameleon-health-check.sh    Cluster sanity check
  demo-readiness-check.sh      Demo validation helper
  demo-links.sh                Prints public URLs
  latest-job-log.sh            Prints newest matching job log
```

## Branches

- `devops/final-hardening`: current demo/integration branch
- `main`: stable integrated baseline
- `paperless-ngx-import`: imported upstream Paperless snapshot

If you want the current demo behavior, use `devops/final-hardening`.

## Prerequisites

Recommended target environment:

- Ubuntu VM on Chameleon
- single-node `k3s`
- Docker available on the node
- enough disk space for local image builds and imported model artifacts

Practical baseline:

- 1 node
- 48 vCPU
- 240 GiB RAM
- at least 40 GiB free root disk before rebuilding images

## Quick Start

### 1. Clone and check out the branch

```bash
git clone https://github.com/Bhanuu01/Datanauts-Intelligent-Deadline-Expiry-Detection.git
cd Datanauts-Intelligent-Deadline-Expiry-Detection
git checkout devops/final-hardening
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

### 3. Create secrets

```bash
bash scripts/create-secrets.sh
```

This creates:

- `paperless-secrets` in `paperless`
- `platform-secrets` in `platform`
- `monitoring-secrets` in `monitoring`
- mirrored runtime copies in `ml`

Optional:

```bash
SAVE_CREDENTIALS_FILE=~/datanauts-secrets.env bash scripts/create-secrets.sh
```

If you use Sealed Secrets:

```bash
kubectl apply -k k8s/sealed-secrets/
USE_EXISTING_SOURCE_SECRETS=true bash scripts/create-secrets.sh
```

### 4. Build and import local images

Build all custom images:

```bash
./scripts/rebuild-k3s-images.sh
```

Build only one family:

```bash
./scripts/rebuild-k3s-images.sh onnx-serving
./scripts/rebuild-k3s-images.sh platform-automation
./scripts/rebuild-k3s-images.sh data-monitoring
```

Custom images used by this deployment include:

- `ghcr.io/bhanuu01/datanauts-online-features:latest`
- `ghcr.io/bhanuu01/datanauts-data-generator:latest`
- `ghcr.io/bhanuu01/datanauts-data-monitoring:latest`
- `ghcr.io/bhanuu01/datanauts-platform-automation:latest`
- `ghcr.io/bhanuu01/datanauts-onnx-serving:latest`

### 5. Deploy the stack

```bash
kubectl apply -f k8s/paperless/
kubectl apply -f k8s/platform/
kubectl apply -k k8s/monitoring
kubectl apply -k k8s/ml
kubectl apply -k k8s/release
```

### 6. Verify deployments

```bash
kubectl rollout status deployment/paperless-ngx -n paperless --timeout=300s
kubectl rollout status deployment/mlflow -n platform --timeout=300s
kubectl rollout status deployment/minio -n platform --timeout=300s
kubectl rollout status deployment/grafana -n monitoring --timeout=300s
kubectl rollout status deployment/prometheus -n monitoring --timeout=300s
kubectl rollout status deployment/online-features -n ml --timeout=300s
kubectl rollout status deployment/deadline-onnx-serving -n ml --timeout=300s
```

## Public URLs

Current Chameleon deployment:

- Paperless: [http://129.114.27.190](http://129.114.27.190)
- MLflow: [http://129.114.27.190:30500](http://129.114.27.190:30500)
- MinIO Console: [http://129.114.27.190:30901](http://129.114.27.190:30901)
- MinIO S3 API: [http://129.114.27.190:30900](http://129.114.27.190:30900)
- Grafana: [http://129.114.27.190/grafana/login](http://129.114.27.190/grafana/login)
- Prometheus: [http://129.114.27.190/prometheus/graph](http://129.114.27.190/prometheus/graph)

If you deploy on another node, replace the public IP.

## Credentials

Do not store credentials in documentation. Read them from Kubernetes secrets.

### Paperless

```bash
kubectl get secret -n paperless paperless-secrets -o jsonpath='{.data.PAPERLESS_ADMIN_USER}' | base64 --decode && echo
kubectl get secret -n paperless paperless-secrets -o jsonpath='{.data.PAPERLESS_ADMIN_PASSWORD}' | base64 --decode && echo
```

### Grafana

```bash
kubectl get secret -n monitoring monitoring-secrets -o jsonpath='{.data.GRAFANA_ADMIN_USER}' | base64 --decode && echo
kubectl get secret -n monitoring monitoring-secrets -o jsonpath='{.data.GRAFANA_ADMIN_PASSWORD}' | base64 --decode && echo
```

### MinIO

```bash
echo mlflow
kubectl get secret -n platform platform-secrets -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 --decode && echo
```

### MLflow and Prometheus

In this deployment, MLflow and Prometheus are normally shown without separate login credentials.

## Internal Service Endpoints

Inside the cluster:

- Paperless API: `http://paperless-ngx.paperless.svc.cluster.local:8000`
- Online features: `http://online-features.ml.svc.cluster.local:8000`
- ONNX serving: `http://deadline-onnx-serving.ml.svc.cluster.local:8004`
- MLflow: `http://mlflow.platform.svc.cluster.local:5000`
- MinIO: `http://minio.platform.svc.cluster.local:9000`
- Prometheus: `http://prometheus.monitoring.svc.cluster.local:9090/prometheus`

## Daily Operations

### Cluster health

```bash
export KUBECONFIG=~/.kube/config
bash scripts/chameleon-health-check.sh
```

### Pod status

```bash
kubectl get pods -n paperless
kubectl get pods -n platform
kubectl get pods -n monitoring
kubectl get pods -n ml
```

### Service logs

```bash
kubectl logs -n paperless deploy/paperless-ngx --tail=150
kubectl logs -n ml deploy/deadline-onnx-serving --tail=100
kubectl logs -n ml deploy/online-features --tail=100
kubectl logs -n ml deploy/data-generator --tail=50
```

## Using the Product

### Upload a document in Paperless

1. Open Paperless.
2. Upload a `.txt` or `.pdf`.
3. Wait for Paperless processing to complete.
4. Open the document details page.
5. Inspect generated tags.

Expected tags on a positive example:

- `Type:Deadline`
- `Deadline:YYYY-MM-DD`
- `Type:Effective`
- `Effective:YYYY-MM-DD`
- `Status:Review Needed`

Feedback tags available in Paperless:

- `Action:Accept`
- `Action:Reject`

### Result artifacts inside Paperless

```bash
kubectl exec -n paperless deploy/paperless-ngx -- sh -c 'ls -lah /usr/src/paperless/data/deadline-results'
```

Inspect a result:

```bash
kubectl exec -n paperless deploy/paperless-ngx -- sh -c 'cat /usr/src/paperless/data/deadline-results/<document-id>.json'
```

### Production data and feedback logs

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
    "ocr_text": "This agreement is effective as of September 22, 2029. Written notice must be delivered by May 25, 2029.",
    "document_type": "contract",
    "filename": "demo.txt"
  }'
```

### Online-features feedback

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

Inspect the latest matching job with helper scripts:

```bash
./scripts/latest-job-log.sh ml retrain 120
./scripts/latest-job-log.sh ml drift-monitor 60
./scripts/latest-job-log.sh ml data-quality 60
```

Decision artifacts:

```bash
kubectl exec -n ml deploy/online-features -- sh -c 'cat /data/retrain_decision.json && echo --- && cat /data/promotion_decision.json'
```

## Monitoring

### Grafana dashboards

Main dashboards:

- `Datanauts Overview`
- `Datanauts Serving`
- `Datanauts Data & Feedback`
- `Datanauts Platform Health`

Typical useful panels:

- serving targets up
- ONNX predictions/sec
- feedback / review activity
- ingest throughput
- platform memory utilization
- monitoring health

### Prometheus

Useful pages:

- Targets
- Alerts
- Graph

Useful queries:

```promql
up
deadline_onnx_predictions_total
online_features_ingest_requests_total
sum(kube_deployment_status_replicas_unavailable{namespace=~"paperless|platform|monitoring|ml"})
```

## Demo Prep

Before a live demo:

```bash
export KUBECONFIG=~/.kube/config
./scripts/demo-readiness-check.sh
./scripts/demo-links.sh
```

Recommended checks:

1. Verify all core deployments are available.
2. Confirm Grafana and Prometheus load.
3. Confirm Paperless login works.
4. Upload one known-good positive document.
5. Confirm Paperless tags include `Deadline:*` and `Effective:*`.
6. Confirm feedback logs and production ingest logs update.

## Notes

- The UI review flow uses Paperless tags, not custom accept/reject buttons.
- Date extraction is strongest on contract-style text and supported date formats.
- If you rebuild serving or platform-automation images, re-import them into k3s with `./scripts/rebuild-k3s-images.sh`.

