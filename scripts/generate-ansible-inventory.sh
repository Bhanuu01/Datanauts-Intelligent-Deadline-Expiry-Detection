#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="${REPO_ROOT}/infra/terraform/openstack"
INVENTORY_DIR="${REPO_ROOT}/infra/ansible/inventory"
INVENTORY_PATH="${INVENTORY_DIR}/inventory.ini"

mkdir -p "${INVENTORY_DIR}"

cluster_name="$(terraform -chdir="${TF_DIR}" output -raw cluster_name)"
ssh_user="$(terraform -chdir="${TF_DIR}" output -raw ssh_user)"
ssh_private_key_path_raw="$(terraform -chdir="${TF_DIR}" output -raw ssh_private_key_path)"
control_plane_name="$(terraform -chdir="${TF_DIR}" output -raw control_plane_name)"
control_plane_public_ip="$(terraform -chdir="${TF_DIR}" output -raw control_plane_public_ip)"
control_plane_private_ip="$(terraform -chdir="${TF_DIR}" output -raw control_plane_private_ip)"
worker_name="$(terraform -chdir="${TF_DIR}" output -raw worker_name)"
worker_public_ip="$(terraform -chdir="${TF_DIR}" output -raw worker_public_ip)"
worker_private_ip="$(terraform -chdir="${TF_DIR}" output -raw worker_private_ip)"
k3s_token="$(terraform -chdir="${TF_DIR}" output -raw k3s_token)"
durable_block_storage_enabled="$(terraform -chdir="${TF_DIR}" output -raw enable_durable_block_storage)"
using_existing_durable_volume="$(terraform -chdir="${TF_DIR}" output -raw using_existing_durable_volume)"
existing_durable_volume_device="$(terraform -chdir="${TF_DIR}" output -raw existing_durable_volume_device)"
paperless_volume_device="$(terraform -chdir="${TF_DIR}" output -raw paperless_volume_device)"
platform_volume_device="$(terraform -chdir="${TF_DIR}" output -raw platform_volume_device)"
ml_volume_device="$(terraform -chdir="${TF_DIR}" output -raw ml_volume_device)"
monitoring_volume_device="$(terraform -chdir="${TF_DIR}" output -raw monitoring_volume_device)"
bootstrap_object_storage_container="$(terraform -chdir="${TF_DIR}" output -raw bootstrap_object_storage_container)"
ssh_private_key_path="${ssh_private_key_path_raw/#\~/${HOME}}"

cat > "${INVENTORY_PATH}" <<EOF
[control_plane]
control-plane ansible_host=${control_plane_public_ip} private_ip=${control_plane_private_ip} node_name=${control_plane_name}

[workers]
worker-1 ansible_host=${worker_public_ip} private_ip=${worker_private_ip} node_name=${worker_name}

[k3s_cluster:children]
control_plane
workers

[k3s_cluster:vars]
ansible_user=${ssh_user}
ansible_ssh_private_key_file=${ssh_private_key_path}
cluster_name=${cluster_name}
k3s_token=${k3s_token}
k3s_control_plane_ip=${control_plane_private_ip}
durable_block_storage_enabled=${durable_block_storage_enabled}
using_existing_durable_volume=${using_existing_durable_volume}
existing_durable_volume_device=${existing_durable_volume_device}
paperless_volume_device=${paperless_volume_device}
platform_volume_device=${platform_volume_device}
ml_volume_device=${ml_volume_device}
monitoring_volume_device=${monitoring_volume_device}
bootstrap_object_storage_container=${bootstrap_object_storage_container}
EOF

echo "Wrote ${INVENTORY_PATH}"
