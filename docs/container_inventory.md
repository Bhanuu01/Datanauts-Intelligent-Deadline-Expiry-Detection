# Container Inventory — Datanauts Intelligent Deadline & Expiry Detection

**Team:** Datanauts (4-person)
**Project:** Intelligent Deadline & Expiry Detection Pipeline
**Infrastructure:** Chameleon Cloud (GPU node `node-llm-single-jk9286`)

---

## 1. Training Container

| Field | Value |
|-------|-------|
| **Image** | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |
| **Built from** | `Dockerfile` (project root) |
| **Purpose** | Run NER and classifier training scripts |
| **GPU** | NVIDIA GPU via `--gpus all` |
| **Key packages** | `transformers==4.46.3`, `datasets==2.19.0`, `mlflow==2.13.0`, `seqeval`, `scikit-learn`, `accelerate` |
| **Entry point** | `python src/train_ner.py --model bert_ner_v1` |
| **Env vars** | `MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL` |
| **Data mount** | `-v ~/workspace:/home/jovyan/work` |
| **Source files** | `src/build_dataset.py`, `src/train_ner.py`, `src/train_classifier.py` |

**Run commands:**
```bash
# Build
docker build -t deadline-training .

# Run NER
docker run --gpus all \
  -e MLFLOW_TRACKING_URI=http://129.114.27.190:8000 \
  -e AWS_ACCESS_KEY_ID=datanauts-key \
  -e AWS_SECRET_ACCESS_KEY=datanauts-secret \
  -e MLFLOW_S3_ENDPOINT_URL=http://129.114.27.190:9000 \
  -v ~/workspace/data:/app/data \
  deadline-training python src/train_ner.py --model bert_ner_v1

# Run Classifier
docker run --gpus all ... deadline-training python src/train_classifier.py --model roberta_clf_v1
```

---

## 2. Jupyter Notebook Container

| Field | Value |
|-------|-------|
| **Image** | `quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1` |
| **Purpose** | Interactive development, data inspection, running scripts via terminal |
| **Port** | `8888:8888` |
| **GPU** | `--gpus all` |
| **Data mount** | `-v ~/workspace:/home/jovyan/work` |
| **Access URL** | `http://<node-ip>:8888/?token=<token>` |
| **Env vars** | `MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL` |

**Run command:**
```bash
docker run -d --rm --gpus all \
  -p 8888:8888 \
  -v ~/workspace:/home/jovyan/work \
  -e MLFLOW_TRACKING_URI=http://129.114.27.190:8000 \
  -e AWS_ACCESS_KEY_ID=datanauts-key \
  -e AWS_SECRET_ACCESS_KEY=datanauts-secret \
  -e MLFLOW_S3_ENDPOINT_URL=http://129.114.27.190:9000 \
  --name jupyter \
  quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1
```

---

## 3. MLflow Tracking Server

| Field | Value |
|-------|-------|
| **Image** | `ghcr.io/mlflow/mlflow:v3.9.0` |
| **Purpose** | Experiment tracking — logs parameters, metrics, and artifacts for every training run |
| **Port** | `8000:8000` |
| **Backend store** | PostgreSQL (`postgres:16`) — stores run metadata |
| **Artifact store** | MinIO (`s3://mlflow-artifacts/`) — stores model checkpoints, reports |
| **Live URL** | `http://129.114.27.190:8000` |
| **Experiments** | `deadline-detection-ner`, `deadline-detection-classifier` |

**Start command (via docker-compose):**
```bash
docker compose -f docker-compose-mlflow.yaml up -d
```

---

## 4. MinIO Object Storage

| Field | Value |
|-------|-------|
| **Image** | `minio/minio:RELEASE.2025-09-07T16-13-09Z` |
| **Purpose** | S3-compatible artifact storage for MLflow (model checkpoints, classification reports) |
| **API port** | `9000:9000` |
| **Console port** | `9001:9001` |
| **Bucket** | `mlflow-artifacts` (auto-created on start) |
| **Credentials** | `datanauts-key` / `datanauts-secret` |
| **Data volume** | `minio_data:/data` (persistent) |
| **Console URL** | `http://129.114.27.190:9001` |

---

## 5. PostgreSQL (MLflow Backend)

| Field | Value |
|-------|-------|
| **Image** | `postgres:16` |
| **Purpose** | Relational backend store for MLflow — run IDs, parameters, metrics, tags |
| **Port** | `5432:5432` |
| **Database** | `mlflowdb` |
| **Credentials** | `user` / `password` |
| **Data volume** | `postgres_data:/var/lib/postgresql/data` (persistent) |

---

## Container Dependency Graph

```
Training Container  ──────────────────────────────────────────►  MLflow Server (:8000)
                                                                        │
Jupyter Container   ──────────────────────────────────────────►         │
                                                                    ┌───┴────────────┐
                                                                    │                │
                                                               PostgreSQL        MinIO (:9000)
                                                               (:5432)           mlflow-artifacts/
```

---

## Environment Variables Summary

| Variable | Value | Used By |
|----------|-------|---------|
| `MLFLOW_TRACKING_URI` | `http://129.114.27.190:8000` | Training, Jupyter |
| `MLFLOW_S3_ENDPOINT_URL` | `http://129.114.27.190:9000` | Training, MLflow server |
| `AWS_ACCESS_KEY_ID` | `datanauts-key` | Training, MLflow server |
| `AWS_SECRET_ACCESS_KEY` | `datanauts-secret` | Training, MLflow server |
| `PYTHONPATH` | `/app` | Training container |
| `GIT_PYTHON_REFRESH` | `quiet` | Training container |

---

## Reproducing the Full Stack

```bash
# 1. Start MLflow + MinIO + PostgreSQL
docker compose -f docker-compose-mlflow.yaml up -d

# 2. Start Jupyter for interactive work
docker run -d --rm --gpus all -p 8888:8888 \
  -v ~/workspace:/home/jovyan/work \
  -e MLFLOW_TRACKING_URI=http://129.114.27.190:8000 \
  -e AWS_ACCESS_KEY_ID=datanauts-key \
  -e AWS_SECRET_ACCESS_KEY=datanauts-secret \
  -e MLFLOW_S3_ENDPOINT_URL=http://129.114.27.190:9000 \
  --name jupyter quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1

# 3. Build and run training
docker build -t deadline-training .
docker run --gpus all -e MLFLOW_TRACKING_URI=... deadline-training
```
