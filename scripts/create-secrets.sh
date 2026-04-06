#!/bin/bash
# Run this ONCE on your Chameleon node after k3s is installed.
# Never commit actual passwords to Git.

# Paperless secrets
kubectl create secret generic paperless-secrets \
  --namespace=paperless \
  --from-literal=POSTGRES_PASSWORD=changeme_db_pass \
  --from-literal=PAPERLESS_SECRET_KEY=$(openssl rand -hex 32) \
  --from-literal=PAPERLESS_ADMIN_PASSWORD=admin123 \
  --dry-run=client -o yaml | kubectl apply -f -

# Platform secrets
kubectl create secret generic platform-secrets \
  --namespace=platform \
  --from-literal=POSTGRES_PASSWORD=changeme_mlflow_pass \
  --from-literal=MINIO_ROOT_PASSWORD=changeme_minio_pass \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Secrets created. Never commit these values to Git!"