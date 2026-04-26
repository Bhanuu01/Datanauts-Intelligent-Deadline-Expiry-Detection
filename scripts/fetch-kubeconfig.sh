#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="${REPO_ROOT}/infra/terraform/openstack"
OUTPUT_DIR="${REPO_ROOT}/infra/ansible/inventory"

control_plane_public_ip="$(terraform -chdir="${TF_DIR}" output -raw control_plane_public_ip)"
ssh_user="$(terraform -chdir="${TF_DIR}" output -raw ssh_user)"
ssh_private_key_path_raw="$(terraform -chdir="${TF_DIR}" output -raw ssh_private_key_path)"
cluster_name="$(terraform -chdir="${TF_DIR}" output -raw cluster_name)"
kubeconfig_path="${OUTPUT_DIR}/${cluster_name}.kubeconfig.yaml"
ssh_private_key_path="${ssh_private_key_path_raw/#\~/${HOME}}"

mkdir -p "${OUTPUT_DIR}"

scp -i "${ssh_private_key_path}" -o StrictHostKeyChecking=no "${ssh_user}@${control_plane_public_ip}:/etc/rancher/k3s/k3s.yaml" "${kubeconfig_path}"

tmp_path="${kubeconfig_path}.tmp"
sed "s/127.0.0.1/${control_plane_public_ip}/g" "${kubeconfig_path}" > "${tmp_path}"
mv "${tmp_path}" "${kubeconfig_path}"

echo "Wrote ${kubeconfig_path}"
