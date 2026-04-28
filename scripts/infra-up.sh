#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="${REPO_ROOT}/infra/terraform/openstack"
ANSIBLE_DIR="${REPO_ROOT}/infra/ansible"
BOOTSTRAP_CONFIG_SCRIPT="${REPO_ROOT}/scripts/bootstrap-config.sh"
IMAGE_REGISTRY_OWNER="${IMAGE_REGISTRY_OWNER:-${GHCR_USERNAME:-bhanuu01}}"
IMAGE_REGISTRY_OWNER="${IMAGE_REGISTRY_OWNER,,}"

tfvars_incomplete() {
  if [ ! -f "${TF_DIR}/terraform.tfvars" ]; then
    return 0
  fi

  if grep -Eq 'replace-(me|with-your-)' "${TF_DIR}/terraform.tfvars"; then
    return 0
  fi

  return 1
}

if ! command -v terraform >/dev/null 2>&1; then
  echo "terraform is required but not installed."
  exit 1
fi

if ! command -v ansible-playbook >/dev/null 2>&1; then
  echo "ansible-playbook is required but not installed."
  exit 1
fi

if tfvars_incomplete; then
  if [[ -t 0 ]]; then
    echo "Terraform variables are missing or incomplete."
    echo "Launching first-run configuration..."
    "${BOOTSTRAP_CONFIG_SCRIPT}"
  else
    echo "Missing or incomplete ${TF_DIR}/terraform.tfvars"
    echo "Run scripts/bootstrap-config.sh first, or provide the DATANAUTS_* environment variables it expects."
    exit 1
  fi
fi

if [ -z "${OS_CLOUD:-}" ] && [ -z "${OS_AUTH_URL:-}" ]; then
  echo "OpenStack credentials are not loaded."
  echo "Source your Chameleon openrc file first, or export OS_CLOUD / OS_AUTH_URL and related OS_* variables."
  exit 1
fi

if tfvars_incomplete; then
  echo "${TF_DIR}/terraform.tfvars still contains placeholder values."
  echo "Re-run scripts/bootstrap-config.sh or update the DATANAUTS_* environment variables."
  exit 1
fi

terraform -chdir="${TF_DIR}" init
terraform -chdir="${TF_DIR}" apply -auto-approve

"${REPO_ROOT}/scripts/generate-ansible-inventory.sh"
ANSIBLE_CONFIG="${ANSIBLE_DIR}/ansible.cfg" ansible-playbook -i "${ANSIBLE_DIR}/inventory/inventory.ini" "${ANSIBLE_DIR}/playbooks/bootstrap-k3s.yml"
"${REPO_ROOT}/scripts/fetch-kubeconfig.sh"

CLUSTER_NAME="$(terraform -chdir="${TF_DIR}" output -raw cluster_name)"
CONTROL_PLANE_PUBLIC_IP="$(terraform -chdir="${TF_DIR}" output -raw control_plane_public_ip)"
CONTROL_PLANE_NODE_NAME="$(terraform -chdir="${TF_DIR}" output -raw control_plane_name)"
KUBECONFIG_PATH="${ANSIBLE_DIR}/inventory/${CLUSTER_NAME}.kubeconfig.yaml"

ANSIBLE_CONFIG="${ANSIBLE_DIR}/ansible.cfg" ansible-playbook -i localhost, "${ANSIBLE_DIR}/playbooks/deploy-platform-stack.yml" \
  --extra-vars "repo_root=${REPO_ROOT}" \
  --extra-vars "kubeconfig_path=${KUBECONFIG_PATH}" \
  --extra-vars "control_plane_public_ip=${CONTROL_PLANE_PUBLIC_IP}" \
  --extra-vars "control_plane_node_name=${CONTROL_PLANE_NODE_NAME}" \
  --extra-vars "image_registry_owner=${IMAGE_REGISTRY_OWNER}"

echo
echo "Infrastructure, k3s bootstrap, and Kubernetes stack deployment completed."
echo "Kubeconfig: ${KUBECONFIG_PATH}"
