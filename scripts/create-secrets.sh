#!/bin/bash
# Run this ONCE on your Chameleon node after k3s is installed.
# Never commit actual passwords to Git.
set -e

PAPERLESS_DB_PASSWORD="${PAPERLESS_DB_PASSWORD:-$(openssl rand -hex 16)}"
PAPERLESS_SECRET_KEY="${PAPERLESS_SECRET_KEY:-$(openssl rand -hex 32)}"
PAPERLESS_ADMIN_PASSWORD="${PAPERLESS_ADMIN_PASSWORD:-$(openssl rand -hex 12)}"
MLFLOW_DB_PASSWORD="${MLFLOW_DB_PASSWORD:-$(openssl rand -hex 16)}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-$(openssl rand -hex 16)}"

# Paperless secrets
kubectl create secret generic paperless-secrets \
  --namespace=paperless \
  --from-literal=POSTGRES_PASSWORD="${PAPERLESS_DB_PASSWORD}" \
  --from-literal=PAPERLESS_SECRET_KEY="${PAPERLESS_SECRET_KEY}" \
  --from-literal=PAPERLESS_ADMIN_PASSWORD="${PAPERLESS_ADMIN_PASSWORD}" \
  --dry-run=client -o yaml | kubectl apply -f -

# Platform secrets
kubectl create secret generic platform-secrets \
  --namespace=platform \
  --from-literal=POSTGRES_PASSWORD="${MLFLOW_DB_PASSWORD}" \
  --from-literal=MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Secrets created."
echo "Paperless admin password: ${PAPERLESS_ADMIN_PASSWORD}"
echo "MinIO root password: ${MINIO_ROOT_PASSWORD}"
