# MLOps Project — Paperless-ngx on Kubernetes
**DevOps / Platform — NYU MLOps (Spring 2026)**

This repository contains all Infrastructure as Code (IaC) and Kubernetes manifests to deploy [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) and shared ML platform services on a single-node Kubernetes cluster hosted on Chameleon Cloud.

---

## Architecture Overview

```
Chameleon VM (48 vCPU, 240 GB RAM, Ubuntu 22.04)
└── k3s Kubernetes
    ├── namespace: paperless
    │   ├── paperless-ngx       (webserver + worker)
    │   ├── postgres            (document database)
    │   ├── redis               (task queue)
    │   └── PersistentVolumes   (documents, media, db)
    └── namespace: platform
        ├── mlflow              (experiment tracking)
        ├── mlflow-postgres     (mlflow metadata db)
        ├── minio               (artifact object storage)
        └── PersistentVolumes   (mlflow runs, minio data)
```

---

## Repository Structure

```
k8s/
├── namespace-paperless.yaml
├── namespace-platform.yaml
├── paperless/
│   ├── paperless-deployment.yaml
│   ├── paperless-ingress.yaml
│   ├── paperless-pvc.yaml
│   ├── paperless-service.yaml
│   ├── postgres-deployment.yaml
│   ├── postgres-pvc.yaml
│   ├── postgres-service.yaml
│   ├── redis-deployment.yaml
│   └── redis-service.yaml
├── platform/
│   ├── mlflow-deployment.yaml
│   ├── mlflow-ingress.yaml
│   ├── mlflow-postgres-deployment.yaml
│   ├── mlflow-postgres-service.yaml
│   ├── mlflow-pvc.yaml
│   ├── mlflow-service.yaml
│   ├── minio-deployment.yaml
│   ├── minio-pvc.yaml
│   └── minio-service.yaml
└── secrets/
    └── README.md              ← instructions for creating secrets
scripts/
├── provision.sh               ← installs k3s + deploys everything
└── create-secrets.sh          ← creates K8s secrets (run once, never commit)
```

---

## Prerequisites

- A Chameleon Cloud account with an active lease
- An Ubuntu 22.04 VM provisioned on Chameleon
- SSH access to the VM
- This repo cloned on the VM

---

## Deployment Instructions

### 1. SSH into your Chameleon VM

```bash
ssh -i your-key.pem cc@<YOUR_NODE_IP>
```

### 2. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/MLOps_project_Paperless-ngx.git
cd MLOps_project_Paperless-ngx
```

### 3. Install k3s

```bash
curl -sfL https://get.k3s.io | sh -
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
export KUBECONFIG=~/.kube/config
echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc
kubectl get nodes  # should show Ready
```

### 4. Create namespaces

```bash
kubectl apply -f k8s/namespace-paperless.yaml
kubectl apply -f k8s/namespace-platform.yaml
```

### 5. Create secrets

> ⚠️ Edit `scripts/create-secrets.sh` to set your own passwords before running.
> Never commit actual secret values to Git.

```bash
bash scripts/create-secrets.sh
```

### 6. Deploy Paperless-ngx stack

```bash
kubectl apply -f k8s/paperless/
```

### 7. Deploy platform stack

```bash
kubectl apply -f k8s/platform/
```

### 8. Verify everything is running

```bash
kubectl get pods -A
kubectl get pvc -A
kubectl get ingress -A
```

All pods should show `Running` and all PVCs should show `Bound`.

---

## Accessing the Services

| Service | URL |
|---|---|
| Paperless-ngx | `http://<NODE_IP>` |
| MLflow | `http://<NODE_IP>:30500` |
| MinIO Console | `http://<NODE_IP>:30901` |

> Make sure your Chameleon security group has ports `80`, `30500`, and `30901` open for inbound TCP traffic.

### Default Credentials

| Service | Username | Password |
|---|---|---|
| Paperless-ngx | `admin` | set in `create-secrets.sh` |
| MinIO | `mlflow` | set in `create-secrets.sh` |
| MLflow | — | no login required |

---

## Secrets Hygiene

No secrets are stored in this repository. All passwords and keys are created directly on the cluster using `scripts/create-secrets.sh` and stored as Kubernetes Secrets. See `k8s/secrets/README.md` for details.

---

## Infrastructure Resource Sizing

| Service | Namespace | CPU Request | CPU Limit | RAM Request | RAM Limit |
|---|---|---|---|---|---|
| paperless-ngx | paperless | 500m | 2000m | 1Gi | 2Gi |
| postgres | paperless | 250m | 500m | 256Mi | 512Mi |
| redis | paperless | 100m | 200m | 64Mi | 128Mi |
| mlflow | platform | 200m | 1000m | 512Mi | 2Gi |
| mlflow-postgres | platform | 200m | 400m | 256Mi | 512Mi |
| minio | platform | 200m | 500m | 256Mi | 512Mi |

Resource limits were determined by observing actual usage with `kubectl top pods` after deployment on a 240 GB RAM, 48 vCPU Chameleon node (flavor: `g1.h100.pci.1`).

---

## Persistent Storage

All stateful services use Kubernetes PersistentVolumeClaims backed by k3s's built-in `local-path` StorageClass. Data survives pod restarts.

| PVC | Namespace | Size | Used By |
|---|---|---|---|
| paperless-data-pvc | paperless | 20Gi | paperless-ngx data |
| paperless-media-pvc | paperless | 20Gi | paperless-ngx media |
| postgres-pvc | paperless | 10Gi | postgres database |
| mlflow-pvc | platform | 10Gi | mlflow artifacts |
| mlflow-postgres-pvc | platform | 5Gi | mlflow metadata db |
| minio-pvc | platform | 20Gi | minio object storage |

---

## For Teammates

| What | Value |
|---|---|
| MLflow tracking URI | `http://<NODE_IP>:30500` |
| MinIO S3 endpoint | `http://<NODE_IP>:30900` |
| MinIO bucket | `mlflow` |
| Paperless-ngx API | `http://<NODE_IP>/api/` |

---

## Chameleon Node Info

| Property | Value |
|---|---|
| Flavor | `g1.h100.pci.1` |
| vCPUs | 48 |
| RAM | 240 GB |
| GPU | 1x H100 94GB (PCI passthrough) |
| OS | Ubuntu 22.04 |
| Kubernetes | k3s v1.34.6 |
