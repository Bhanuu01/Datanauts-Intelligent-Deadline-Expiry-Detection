#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RENDER_ROOT="${RENDER_ROOT:-/tmp/datanauts-rendered-k8s}"
RENDERED_K8S_ROOT="${RENDER_ROOT}/k8s"
KUBECONFIG_PATH="${KUBECONFIG:-}"
CONTROL_PLANE_PUBLIC_IP="${CONTROL_PLANE_PUBLIC_IP:-}"
CONTROL_PLANE_NODE_NAME="${CONTROL_PLANE_NODE_NAME:-}"
RUN_BOOTSTRAP_JOB="${RUN_BOOTSTRAP_JOB:-false}"
USE_EXISTING_SOURCE_SECRETS="${USE_EXISTING_SOURCE_SECRETS:-true}"
PUBLIC_HOST="${PUBLIC_HOST:-${CONTROL_PLANE_PUBLIC_IP}}"
GHCR_USERNAME="${GHCR_USERNAME:-}"
GHCR_TOKEN="${GHCR_TOKEN:-}"
IMAGE_REGISTRY_OWNER="${IMAGE_REGISTRY_OWNER:-}"

require_command() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "Required command not found: ${cmd}" >&2
    exit 1
  }
}

require_command kubectl
require_command cp
require_command rm
require_command mkdir
require_command sed
require_command bash

if [[ -z "${KUBECONFIG_PATH}" ]]; then
  if [[ -f "${HOME}/.kube/config" ]]; then
    KUBECONFIG_PATH="${HOME}/.kube/config"
  else
    echo "KUBECONFIG must point to the target cluster kubeconfig." >&2
    exit 1
  fi
fi

if [[ -z "${CONTROL_PLANE_PUBLIC_IP}" ]]; then
  echo "CONTROL_PLANE_PUBLIC_IP is required." >&2
  exit 1
fi

if [[ -z "${CONTROL_PLANE_NODE_NAME}" ]]; then
  echo "CONTROL_PLANE_NODE_NAME is required." >&2
  exit 1
fi

if [[ -z "${IMAGE_REGISTRY_OWNER}" ]]; then
  IMAGE_REGISTRY_OWNER="bhanuu01"
fi
IMAGE_REGISTRY_OWNER="${IMAGE_REGISTRY_OWNER,,}"

rollout_status() {
  local namespace="$1"
  local deployment="$2"
  local timeout="${3:-300s}"

  if kubectl rollout status "deployment/${deployment}" -n "${namespace}" --timeout="${timeout}"; then
    return 0
  fi

  echo "Rollout status timed out for deployment/${deployment} in namespace ${namespace}. Inspecting deployment health..." >&2

  local replicas observed_generation generation updated_replicas available_replicas unavailable_replicas
  replicas="$(kubectl get deployment "${deployment}" -n "${namespace}" -o jsonpath='{.spec.replicas}')"
  generation="$(kubectl get deployment "${deployment}" -n "${namespace}" -o jsonpath='{.metadata.generation}')"
  observed_generation="$(kubectl get deployment "${deployment}" -n "${namespace}" -o jsonpath='{.status.observedGeneration}')"
  updated_replicas="$(kubectl get deployment "${deployment}" -n "${namespace}" -o jsonpath='{.status.updatedReplicas}')"
  available_replicas="$(kubectl get deployment "${deployment}" -n "${namespace}" -o jsonpath='{.status.availableReplicas}')"
  unavailable_replicas="$(kubectl get deployment "${deployment}" -n "${namespace}" -o jsonpath='{.status.unavailableReplicas}')"

  replicas="${replicas:-0}"
  generation="${generation:-0}"
  observed_generation="${observed_generation:-0}"
  updated_replicas="${updated_replicas:-0}"
  available_replicas="${available_replicas:-0}"
  unavailable_replicas="${unavailable_replicas:-0}"

  if [[ "${observed_generation}" -ge "${generation}" ]] && \
     [[ "${updated_replicas}" -ge "${replicas}" ]] && \
     [[ "${available_replicas}" -ge "${replicas}" ]] && \
     [[ "${unavailable_replicas}" -eq 0 ]]; then
    echo "Deployment ${namespace}/${deployment} is healthy on the new ReplicaSet; accepting rollout despite slow old-pod termination." >&2
    kubectl get deployment "${deployment}" -n "${namespace}" -o wide
    return 0
  fi

  echo "Deployment ${namespace}/${deployment} is not healthy after timeout." >&2
  kubectl get deployment "${deployment}" -n "${namespace}" -o wide || true
  kubectl describe deployment "${deployment}" -n "${namespace}" || true
  kubectl get pods -n "${namespace}" -l "app=${deployment}" -o wide || true
  return 1
}

restart_deployment() {
  local namespace="$1"
  local deployment="$2"
  echo "Restarting deployment/${deployment} in namespace ${namespace} to pick up refreshed :latest images"
  kubectl rollout restart "deployment/${deployment}" -n "${namespace}"
}

echo "Preparing rendered manifest workspace at ${RENDER_ROOT}"
rm -rf "${RENDER_ROOT}"
mkdir -p "${RENDER_ROOT}"
cp -R "${REPO_ROOT}/k8s" "${RENDER_ROOT}/"

sed -i.bak "s/__CONTROL_PLANE_PUBLIC_IP__/${CONTROL_PLANE_PUBLIC_IP}/g" \
  "${RENDERED_K8S_ROOT}/paperless/paperless-deployment.yaml" \
  "${RENDERED_K8S_ROOT}/monitoring/grafana-deployment.yaml" \
  "${RENDERED_K8S_ROOT}/monitoring/prometheus-deployment.yaml" \
  "${RENDERED_K8S_ROOT}/platform/mlflow-deployment.yaml"
rm -f \
  "${RENDERED_K8S_ROOT}/paperless/paperless-deployment.yaml.bak" \
  "${RENDERED_K8S_ROOT}/monitoring/grafana-deployment.yaml.bak" \
  "${RENDERED_K8S_ROOT}/monitoring/prometheus-deployment.yaml.bak" \
  "${RENDERED_K8S_ROOT}/platform/mlflow-deployment.yaml.bak"

sed -i.bak "s/__CONTROL_PLANE_NODE_NAME__/${CONTROL_PLANE_NODE_NAME}/g" \
  "${RENDERED_K8S_ROOT}/durable-pvs.yaml"
rm -f "${RENDERED_K8S_ROOT}/durable-pvs.yaml.bak"

find "${RENDERED_K8S_ROOT}" -type f \( -name '*.yaml' -o -name '*.yml' \) -print0 | \
  while IFS= read -r -d '' manifest; do
    sed -i.bak "s#ghcr\\.io/bhanuu01/#ghcr.io/${IMAGE_REGISTRY_OWNER}/#g" "${manifest}"
    rm -f "${manifest}.bak"
  done

echo "Applying namespaces"
kubectl apply -f "${RENDERED_K8S_ROOT}/namespace-paperless.yaml"
kubectl apply -f "${RENDERED_K8S_ROOT}/namespace-platform.yaml"
kubectl apply -f "${RENDERED_K8S_ROOT}/namespace-ml.yaml"
kubectl apply -f "${RENDERED_K8S_ROOT}/monitoring/namespace.yaml"

if [[ -n "${GHCR_USERNAME}" && -n "${GHCR_TOKEN}" ]]; then
  echo "Refreshing GHCR pull secret in ml namespace"
  kubectl create secret docker-registry ghcr-pull-secret \
    --namespace=ml \
    --docker-server=ghcr.io \
    --docker-username="${GHCR_USERNAME}" \
    --docker-password="${GHCR_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl patch serviceaccount default -n ml --type merge \
    -p '{"imagePullSecrets":[{"name":"ghcr-pull-secret"}]}'
fi

echo "Refreshing runtime secrets"
USE_EXISTING_SOURCE_SECRETS="${USE_EXISTING_SOURCE_SECRETS}" \
  KUBECONFIG="${KUBECONFIG_PATH}" \
  bash "${REPO_ROOT}/scripts/create-secrets.sh"

echo "Syncing public endpoint config"
KUBECONFIG="${KUBECONFIG_PATH}" PUBLIC_HOST="${PUBLIC_HOST}" \
  bash "${REPO_ROOT}/scripts/sync-public-endpoints.sh" "${PUBLIC_HOST}"

echo "Applying durable PV definitions"
kubectl apply -f "${RENDERED_K8S_ROOT}/durable-pvs.yaml"

echo "Applying Paperless manifests"
kubectl apply -f "${RENDERED_K8S_ROOT}/paperless/"

echo "Applying platform manifests"
kubectl apply -f "${RENDERED_K8S_ROOT}/platform/"

echo "Applying monitoring manifests"
kubectl apply -k "${RENDERED_K8S_ROOT}/monitoring"

echo "Applying ML manifests"
kubectl apply -k "${RENDERED_K8S_ROOT}/ml"

echo "Applying release manifests"
kubectl apply -k "${RENDERED_K8S_ROOT}/release"

if [[ -n "${GHCR_USERNAME}" && -n "${GHCR_TOKEN}" ]]; then
  kubectl patch serviceaccount release-promotion -n ml --type merge \
    -p '{"imagePullSecrets":[{"name":"ghcr-pull-secret"}]}' || true
fi

if [[ "${RUN_BOOTSTRAP_JOB}" == "true" ]]; then
  echo "Recreating bootstrap training job"
  kubectl delete job/model-bootstrap -n ml --ignore-not-found=true
  kubectl apply -f "${RENDERED_K8S_ROOT}/ml/model-bootstrap-job.yaml"
  kubectl wait --for=condition=complete job/model-bootstrap -n ml --timeout=3600s
fi

echo "Restarting image-based workloads"
restart_deployment ml data-generator
restart_deployment ml online-features
restart_deployment ml deadline-onnx-serving
restart_deployment ml deadline-onnx-serving-staging
restart_deployment ml deadline-onnx-serving-canary
restart_deployment ml deadline-onnx-serving-production

echo "Waiting for rollouts"
rollout_status paperless postgres
rollout_status paperless redis
rollout_status paperless paperless-ngx
rollout_status platform mlflow-postgres
rollout_status platform minio
rollout_status platform mlflow
rollout_status monitoring kube-state-metrics
rollout_status monitoring prometheus
rollout_status monitoring grafana
rollout_status ml data-generator
rollout_status ml online-features
rollout_status ml deadline-onnx-serving
rollout_status ml deadline-onnx-serving-staging
rollout_status ml deadline-onnx-serving-canary
rollout_status ml deadline-onnx-serving-production

echo "--- Deployments ---"
kubectl get deploy -A
