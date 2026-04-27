#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="${REPO_ROOT}/infra/terraform/openstack"
TFVARS_PATH="${TF_DIR}/terraform.tfvars"

has_openstack_context() {
  [[ -n "${OS_CLOUD:-}" || -n "${OS_AUTH_URL:-}" ]]
}

lookup_openstack_value() {
  local cmd="$1"
  if ! command -v openstack >/dev/null 2>&1; then
    return 1
  fi
  if ! has_openstack_context; then
    return 1
  fi
  eval "${cmd}" 2>/dev/null || return 1
}

prompt_value() {
  local prompt="$1"
  local default_value="${2:-}"
  local response=""

  if [[ -n "${default_value}" ]]; then
    read -r -p "${prompt} [${default_value}]: " response || true
    printf "%s" "${response:-${default_value}}"
  else
    while [[ -z "${response}" ]]; do
      read -r -p "${prompt}: " response || true
    done
    printf "%s" "${response}"
  fi
}

resolve_string() {
  local env_name="$1"
  local prompt="$2"
  local default_value="${3:-}"
  local required="${4:-true}"
  local current_value="${!env_name:-}"

  if [[ -n "${current_value}" ]]; then
    printf "%s" "${current_value}"
    return 0
  fi

  if [[ -t 0 ]]; then
    if [[ "${required}" == "false" && -z "${default_value}" ]]; then
      local response=""
      read -r -p "${prompt} [optional]: " response || true
      printf "%s" "${response}"
      return 0
    fi

    prompt_value "${prompt}" "${default_value}"
    return 0
  fi

  if [[ -n "${default_value}" ]]; then
    printf "%s" "${default_value}"
    return 0
  fi

  if [[ "${required}" == "false" ]]; then
    printf ""
    return 0
  fi

  echo "Missing required environment variable: ${env_name}" >&2
  exit 1
}

resolve_bool() {
  local env_name="$1"
  local prompt="$2"
  local default_value="$3"
  local current_value="${!env_name:-}"

  if [[ -n "${current_value}" ]]; then
    printf "%s" "${current_value}"
    return 0
  fi

  if [[ -t 0 ]]; then
    local answer
    answer="$(prompt_value "${prompt} (true/false)" "${default_value}")"
    printf "%s" "${answer}"
    return 0
  fi

  printf "%s" "${default_value}"
}

REGION="$(resolve_string DATANAUTS_REGION "OpenStack region" "KVM@TACC")"
CLUSTER_NAME="$(resolve_string DATANAUTS_CLUSTER_NAME "Cluster name prefix" "datanauts")"
PRIVATE_NETWORK_ID="$(resolve_string DATANAUTS_PRIVATE_NETWORK_ID "Private network UUID")"

PRIVATE_NETWORK_NAME_DEFAULT="$(lookup_openstack_value "openstack network show '${PRIVATE_NETWORK_ID}' -f value -c name" || true)"
PRIVATE_NETWORK_NAME="$(resolve_string DATANAUTS_PRIVATE_NETWORK_NAME "Private network name" "${PRIVATE_NETWORK_NAME_DEFAULT}")"

PRIVATE_SUBNET_ID="$(resolve_string DATANAUTS_PRIVATE_SUBNET_ID "Private subnet UUID")"
PRIVATE_SUBNET_CIDR_DEFAULT="$(lookup_openstack_value "openstack subnet show '${PRIVATE_SUBNET_ID}' -f value -c cidr" || true)"
PRIVATE_SUBNET_CIDR="$(resolve_string DATANAUTS_PRIVATE_SUBNET_CIDR "Private subnet CIDR" "${PRIVATE_SUBNET_CIDR_DEFAULT}")"

EXTERNAL_NETWORK_NAME="$(resolve_string DATANAUTS_EXTERNAL_NETWORK_NAME "External network name" "public")"
IMAGE_NAME="$(resolve_string DATANAUTS_IMAGE_NAME "Image name" "CC-Ubuntu24.04")"
SSH_KEYPAIR_NAME="$(resolve_string DATANAUTS_SSH_KEYPAIR_NAME "OpenStack keypair name")"
SSH_PRIVATE_KEY_PATH="$(resolve_string DATANAUTS_SSH_PRIVATE_KEY_PATH "Local SSH private key path")"
SSH_USER="$(resolve_string DATANAUTS_SSH_USER "SSH username" "cc")"
SSH_ALLOWED_CIDR="$(resolve_string DATANAUTS_SSH_ALLOWED_CIDR "SSH allowed CIDR" "0.0.0.0/0")"
KUBERNETES_API_ALLOWED_CIDR="$(resolve_string DATANAUTS_KUBERNETES_API_ALLOWED_CIDR "Kubernetes API allowed CIDR" "0.0.0.0/0")"
HTTP_ALLOWED_CIDR="$(resolve_string DATANAUTS_HTTP_ALLOWED_CIDR "HTTP allowed CIDR" "0.0.0.0/0")"
HTTPS_ALLOWED_CIDR="$(resolve_string DATANAUTS_HTTPS_ALLOWED_CIDR "HTTPS allowed CIDR" "0.0.0.0/0")"
CONTROL_PLANE_LEASE_ID="$(resolve_string DATANAUTS_CONTROL_PLANE_LEASE_ID "Control-plane lease ID")"
CONTROL_PLANE_RESERVATION_ID="$(resolve_string DATANAUTS_CONTROL_PLANE_RESERVATION_ID "Control-plane reservation flavor ID")"
WORKER_LEASE_ID="$(resolve_string DATANAUTS_WORKER_LEASE_ID "Worker lease ID")"
WORKER_RESERVATION_ID="$(resolve_string DATANAUTS_WORKER_RESERVATION_ID "Worker reservation flavor ID")"
ENABLE_DURABLE_BLOCK_STORAGE="$(resolve_bool DATANAUTS_ENABLE_DURABLE_BLOCK_STORAGE "Enable durable block storage" "true")"
EXISTING_DURABLE_VOLUME_ID="$(resolve_string DATANAUTS_EXISTING_DURABLE_VOLUME_ID "Existing durable block volume UUID (leave blank to let Terraform create managed volumes)" "" false)"
EXISTING_DURABLE_VOLUME_DEVICE="$(resolve_string DATANAUTS_EXISTING_DURABLE_VOLUME_DEVICE "Existing durable block volume device path" "/dev/vdb")"
BOOTSTRAP_OBJECT_STORAGE_CONTAINER="$(resolve_string DATANAUTS_BOOTSTRAP_OBJECT_STORAGE_CONTAINER "Bootstrap object storage container name")"

mkdir -p "${TF_DIR}"
cat > "${TFVARS_PATH}" <<EOF
region                       = "${REGION}"
cluster_name                 = "${CLUSTER_NAME}"
private_network_id           = "${PRIVATE_NETWORK_ID}"
private_network_name         = "${PRIVATE_NETWORK_NAME}"
private_subnet_id            = "${PRIVATE_SUBNET_ID}"
private_subnet_cidr          = "${PRIVATE_SUBNET_CIDR}"
external_network_name        = "${EXTERNAL_NETWORK_NAME}"
image_name                   = "${IMAGE_NAME}"
ssh_keypair_name             = "${SSH_KEYPAIR_NAME}"
ssh_private_key_path         = "${SSH_PRIVATE_KEY_PATH}"
ssh_user                     = "${SSH_USER}"
ssh_allowed_cidr             = "${SSH_ALLOWED_CIDR}"
kubernetes_api_allowed_cidr  = "${KUBERNETES_API_ALLOWED_CIDR}"
http_allowed_cidr            = "${HTTP_ALLOWED_CIDR}"
https_allowed_cidr           = "${HTTPS_ALLOWED_CIDR}"
control_plane_lease_id       = "${CONTROL_PLANE_LEASE_ID}"
control_plane_reservation_id = "${CONTROL_PLANE_RESERVATION_ID}"
worker_lease_id              = "${WORKER_LEASE_ID}"
worker_reservation_id        = "${WORKER_RESERVATION_ID}"
enable_durable_block_storage = ${ENABLE_DURABLE_BLOCK_STORAGE}
existing_durable_volume_id   = "${EXISTING_DURABLE_VOLUME_ID}"
existing_durable_volume_device = "${EXISTING_DURABLE_VOLUME_DEVICE}"
bootstrap_object_storage_container = "${BOOTSTRAP_OBJECT_STORAGE_CONTAINER}"
EOF

echo "Wrote ${TFVARS_PATH}"
