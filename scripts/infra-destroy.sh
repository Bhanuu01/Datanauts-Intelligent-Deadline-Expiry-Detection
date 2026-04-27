#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="${REPO_ROOT}/infra/terraform/openstack"

if ! command -v terraform >/dev/null 2>&1; then
  echo "terraform is required but not installed."
  exit 1
fi

if [ ! -f "${TF_DIR}/terraform.tfvars" ]; then
  echo "Missing ${TF_DIR}/terraform.tfvars"
  exit 1
fi

if terraform -chdir="${TF_DIR}" output -raw enable_durable_block_storage >/dev/null 2>&1; then
  durable_enabled="$(terraform -chdir="${TF_DIR}" output -raw enable_durable_block_storage)"
  using_existing_volume="$(terraform -chdir="${TF_DIR}" output -raw using_existing_durable_volume 2>/dev/null || echo false)"
  if [ "${durable_enabled}" = "true" ] && [ "${using_existing_volume}" != "true" ] && [ "${ALLOW_DURABLE_VOLUME_DESTROY:-false}" != "true" ]; then
    echo "Durable OpenStack block volumes are enabled for this cluster."
    echo "Refusing to run a full terraform destroy because it would also delete durable data volumes."
    echo "If you really want to destroy the volumes too, rerun with ALLOW_DURABLE_VOLUME_DESTROY=true."
    exit 1
  fi
fi

terraform -chdir="${TF_DIR}" destroy -auto-approve
