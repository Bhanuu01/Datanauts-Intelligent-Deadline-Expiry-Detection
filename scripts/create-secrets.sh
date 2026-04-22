#!/bin/bash
# Run this ONCE on your Chameleon node after k3s is installed.
# Never commit actual passwords to Git.
set -euo pipefail
umask 077

PAPERLESS_DB_PASSWORD="${PAPERLESS_DB_PASSWORD:-$(openssl rand -hex 16)}"
PAPERLESS_SECRET_KEY="${PAPERLESS_SECRET_KEY:-$(openssl rand -hex 32)}"
PAPERLESS_ADMIN_PASSWORD="${PAPERLESS_ADMIN_PASSWORD:-$(openssl rand -hex 12)}"
MLFLOW_DB_PASSWORD="${MLFLOW_DB_PASSWORD:-$(openssl rand -hex 16)}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-$(openssl rand -hex 16)}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-$(openssl rand -hex 12)}"
SHOW_GENERATED_SECRETS="${SHOW_GENERATED_SECRETS:-false}"
SAVE_CREDENTIALS_FILE="${SAVE_CREDENTIALS_FILE:-}"
USE_EXISTING_SOURCE_SECRETS="${USE_EXISTING_SOURCE_SECRETS:-false}"

decode_secret() {
  local namespace="$1"
  local name="$2"
  local key="$3"
  kubectl get secret "${name}" -n "${namespace}" -o "jsonpath={.data.${key}}" | base64 --decode
}

if [[ "${USE_EXISTING_SOURCE_SECRETS}" != "true" ]]; then
  # Paperless secrets
  kubectl create secret generic paperless-secrets \
    --namespace=paperless \
    --from-literal=POSTGRES_PASSWORD="${PAPERLESS_DB_PASSWORD}" \
    --from-literal=PAPERLESS_SECRET_KEY="${PAPERLESS_SECRET_KEY}" \
    --from-literal=PAPERLESS_ADMIN_USER="admin" \
    --from-literal=PAPERLESS_ADMIN_PASSWORD="${PAPERLESS_ADMIN_PASSWORD}" \
    --dry-run=client -o yaml | kubectl apply -f -

  # Platform secrets
  kubectl create secret generic platform-secrets \
    --namespace=platform \
    --from-literal=POSTGRES_PASSWORD="${MLFLOW_DB_PASSWORD}" \
    --from-literal=MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD}" \
    --dry-run=client -o yaml | kubectl apply -f -
fi

# Monitoring secrets
kubectl create secret generic monitoring-secrets \
  --namespace=monitoring \
  --from-literal=GRAFANA_ADMIN_USER="admin" \
  --from-literal=GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD}" \
  --dry-run=client -o yaml | kubectl apply -f -

bash "$(dirname "$0")/sync-runtime-secrets.sh"

if [[ "${USE_EXISTING_SOURCE_SECRETS}" == "true" ]]; then
  PAPERLESS_ADMIN_PASSWORD="$(decode_secret paperless paperless-secrets PAPERLESS_ADMIN_PASSWORD)"
  MINIO_ROOT_PASSWORD="$(decode_secret platform platform-secrets MINIO_ROOT_PASSWORD)"
fi
GRAFANA_ADMIN_PASSWORD="$(decode_secret monitoring monitoring-secrets GRAFANA_ADMIN_PASSWORD)"

echo "Secrets created."

if [[ -n "${SAVE_CREDENTIALS_FILE}" ]]; then
  cat > "${SAVE_CREDENTIALS_FILE}" <<EOF
PAPERLESS_ADMIN_USER=admin
PAPERLESS_ADMIN_PASSWORD=${PAPERLESS_ADMIN_PASSWORD}
MINIO_ROOT_USER=mlflow
MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
EOF
  chmod 600 "${SAVE_CREDENTIALS_FILE}"
  echo "Credential snapshot written to ${SAVE_CREDENTIALS_FILE}"
fi

if [[ "${SHOW_GENERATED_SECRETS}" == "true" ]]; then
  echo "Paperless admin password: ${PAPERLESS_ADMIN_PASSWORD}"
  echo "MinIO root password: ${MINIO_ROOT_PASSWORD}"
  echo "Grafana admin password: ${GRAFANA_ADMIN_PASSWORD}"
else
  echo "Generated secret values were not printed."
  echo "Set SHOW_GENERATED_SECRETS=true or SAVE_CREDENTIALS_FILE=/secure/path.env if you need to persist them outside the cluster."
fi
