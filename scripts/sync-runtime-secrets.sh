#!/bin/bash
set -euo pipefail

encode_literal() {
  printf "%s" "$1" | base64 | tr -d '\n'
}

secret_data() {
  local namespace="$1"
  local name="$2"
  local key="$3"
  kubectl get secret "${name}" -n "${namespace}" -o "jsonpath={.data.${key}}"
}

apply_secret() {
  local namespace="$1"
  local name="$2"
  local data_block="$3"

  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: ${name}
  namespace: ${namespace}
type: Opaque
data:
${data_block}
EOF
}

paperless_admin_user_b64="$(secret_data paperless paperless-secrets PAPERLESS_ADMIN_USER)"
paperless_admin_password_b64="$(secret_data paperless paperless-secrets PAPERLESS_ADMIN_PASSWORD)"
minio_root_password_b64="$(secret_data platform platform-secrets MINIO_ROOT_PASSWORD)"

apply_secret ml paperless-secrets "  PAPERLESS_ADMIN_USER: ${paperless_admin_user_b64}
  PAPERLESS_ADMIN_PASSWORD: ${paperless_admin_password_b64}"

apply_secret ml platform-secrets "  MINIO_ROOT_PASSWORD: ${minio_root_password_b64}"

echo "Mirrored paperless and platform runtime secrets into namespace ml."
