# Sealed Secrets

Secrets in this directory are encrypted using Bitnami Sealed Secrets.
They are SAFE to commit to Git — only the cluster controller can decrypt them.

## How it works
1. Sealed Secrets controller runs in kube-system namespace
2. kubeseal CLI encrypts secrets using the controller's public key
3. SealedSecret CRDs are committed to Git
4. On apply, the controller decrypts them into regular K8s Secrets
5. `scripts/sync-runtime-secrets.sh` mirrors the runtime-only copies needed in `ml`
6. `k8s/platform/minio-bootstrap-job.yaml` creates the `mlflow` bucket after MinIO comes up

## To apply sealed secrets to a fresh cluster:
```bash
# Install controller first
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.26.3/controller.yaml

# Then apply sealed secrets for the source namespaces
kubectl apply -k k8s/sealed-secrets/

# Create any non-sealed runtime-only secrets (for example Grafana admin credentials)
USE_EXISTING_SOURCE_SECRETS=true bash scripts/create-secrets.sh

# Sync the current public IP into runtime config that needs an external URL
bash scripts/sync-public-endpoints.sh

# Ensure the MLflow artifact bucket exists in MinIO
kubectl apply -f k8s/platform/minio-bootstrap-job.yaml
kubectl wait --for=condition=complete job/minio-bootstrap -n platform --timeout=180s
```

## To re-seal (if you change passwords):
```bash
kubectl create secret generic paperless-secrets \
  --namespace=paperless \
  --from-literal=POSTGRES_PASSWORD=newpassword \
  --dry-run=client -o yaml \
  | kubeseal --format yaml > k8s/sealed-secrets/paperless-sealed-secret.yaml
```

After applying updated SealedSecrets, refresh the mirrored copies used by ML jobs:

```bash
bash scripts/sync-runtime-secrets.sh
```

Any time you rotate Paperless or MinIO credentials, rerun the sync step before
starting retraining jobs so the `ml` namespace runtime copies stay current.

Only source namespace secrets are sealed in Git. The `ml` namespace receives the
minimum runtime copies it needs after decryption so retraining and release jobs
can authenticate without keeping duplicate plaintext secrets in the repository.
