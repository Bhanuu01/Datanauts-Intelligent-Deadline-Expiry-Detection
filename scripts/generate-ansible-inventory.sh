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
EOF

echo "Wrote ${INVENTORY_PATH}"
