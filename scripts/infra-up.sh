#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="${REPO_ROOT}/infra/terraform/openstack"
ANSIBLE_DIR="${REPO_ROOT}/infra/ansible"

if ! command -v terraform >/dev/null 2>&1; then
  echo "terraform is required but not installed."
  exit 1
fi

if ! command -v ansible-playbook >/dev/null 2>&1; then
  echo "ansible-playbook is required but not installed."
  exit 1
fi

if [ -z "${OS_CLOUD:-}" ] && [ -z "${OS_AUTH_URL:-}" ]; then
  echo "OpenStack credentials are not loaded."
  echo "Source your Chameleon openrc file first, or export OS_CLOUD / OS_AUTH_URL and related OS_* variables."
  exit 1
fi

if [ ! -f "${TF_DIR}/terraform.tfvars" ]; then
  echo "Missing ${TF_DIR}/terraform.tfvars"
  echo "Create it from infra/terraform/openstack/terraform.tfvars.example first."
  exit 1
fi

if grep -Eq 'replace-(me|with-your-)' "${TF_DIR}/terraform.tfvars"; then
  echo "${TF_DIR}/terraform.tfvars still contains placeholder values."
  echo "Fill in your real Chameleon network, reservation, keypair, and SSH key values first."
  exit 1
fi

terraform -chdir="${TF_DIR}" init
terraform -chdir="${TF_DIR}" apply -auto-approve

"${REPO_ROOT}/scripts/generate-ansible-inventory.sh"
ANSIBLE_CONFIG="${ANSIBLE_DIR}/ansible.cfg" ansible-playbook -i "${ANSIBLE_DIR}/inventory/inventory.ini" "${ANSIBLE_DIR}/playbooks/bootstrap-k3s.yml"
"${REPO_ROOT}/scripts/fetch-kubeconfig.sh"

CLUSTER_NAME="$(terraform -chdir="${TF_DIR}" output -raw cluster_name)"
CONTROL_PLANE_PUBLIC_IP="$(terraform -chdir="${TF_DIR}" output -raw control_plane_public_ip)"
KUBECONFIG_PATH="${ANSIBLE_DIR}/inventory/${CLUSTER_NAME}.kubeconfig.yaml"

ANSIBLE_CONFIG="${ANSIBLE_DIR}/ansible.cfg" ansible-playbook -i localhost, "${ANSIBLE_DIR}/playbooks/deploy-platform-stack.yml" \
  --extra-vars "repo_root=${REPO_ROOT}" \
  --extra-vars "kubeconfig_path=${KUBECONFIG_PATH}" \
  --extra-vars "control_plane_public_ip=${CONTROL_PLANE_PUBLIC_IP}"

echo
echo "Infrastructure, k3s bootstrap, and Kubernetes stack deployment completed."
echo "Kubeconfig: ${KUBECONFIG_PATH}"
