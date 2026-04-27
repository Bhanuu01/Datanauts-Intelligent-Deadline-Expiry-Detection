#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="${REPO_ROOT}/infra/terraform/openstack"
INVENTORY_FILE="${REPO_ROOT}/infra/ansible/inventory/inventory.ini"

resolve_public_host() {
  if [[ -n "${1:-}" ]]; then
    printf "%s" "$1"
    return 0
  fi

  if [[ -n "${PUBLIC_HOST:-}" ]]; then
    printf "%s" "${PUBLIC_HOST}"
    return 0
  fi

  if command -v terraform >/dev/null 2>&1 && [[ -d "${TF_DIR}" ]]; then
    local tf_host
    tf_host="$(terraform -chdir="${TF_DIR}" output -raw control_plane_public_ip 2>/dev/null || true)"
    if [[ -n "${tf_host}" ]]; then
      printf "%s" "${tf_host}"
      return 0
    fi
  fi

  if [[ -f "${INVENTORY_FILE}" ]]; then
    local inventory_host
    inventory_host="$(awk '/^\[control_plane\]/{flag=1;next}/^\[/{flag=0}flag && /ansible_host=/{for(i=1;i<=NF;i++) if($i ~ /^ansible_host=/){split($i,a,"="); print a[2]; exit}}' "${INVENTORY_FILE}")"
    if [[ -n "${inventory_host}" ]]; then
      printf "%s" "${inventory_host}"
      return 0
    fi
  fi

  return 1
}

PUBLIC_HOST="$(resolve_public_host "${1:-}")" || {
  echo "Unable to determine the public host automatically." >&2
  echo "Pass it explicitly: bash scripts/demo-links.sh <public-host>" >&2
  exit 1
}

cat <<EOF
Demo URLs
---------
Paperless:      http://${PUBLIC_HOST}
MLflow:         http://${PUBLIC_HOST}/
MinIO Console:  http://${PUBLIC_HOST}:30901
Grafana:        http://${PUBLIC_HOST}/grafana
Prometheus:     http://${PUBLIC_HOST}/prometheus

Logins
------
Paperless: admin / <paperless admin password from secret>
Grafana:   admin / <monitoring secret password>
MinIO:     mlflow / <platform secret password>
EOF
