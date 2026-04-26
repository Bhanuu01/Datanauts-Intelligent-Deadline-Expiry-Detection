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

terraform -chdir="${TF_DIR}" destroy -auto-approve
