#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

PUBLIC_HOST="${PUBLIC_HOST:-${1:-}}"
BUILD_IMAGES="${BUILD_IMAGES:-1}"
INSTALL_K3S="${INSTALL_K3S:-1}"
DISABLE_FIREWALL="${DISABLE_FIREWALL:-1}"
SEED_MODEL_STORAGE="${SEED_MODEL_STORAGE:-1}"
SEED_MLFLOW_REGISTRY="${SEED_MLFLOW_REGISTRY:-1}"

MODEL_SOURCE_ROOT="${MODEL_SOURCE_ROOT:-${ROOT_DIR}/models}"
QUANTIZED_MODEL_DIR="${QUANTIZED_MODEL_DIR:-${MODEL_SOURCE_ROOT}/onnx_quantized_model}"
NER_BASE_MODEL_DIR="${NER_BASE_MODEL_DIR:-${MODEL_SOURCE_ROOT}/deadline-ner-bert_ner_v1}"
CLF_BASE_MODEL_DIR="${CLF_BASE_MODEL_DIR:-${MODEL_SOURCE_ROOT}/deadline-clf-roberta_clf_v6}"
BASELINE_DATA_DIR="${BASELINE_DATA_DIR:-${ROOT_DIR}/components/data/gx_quality/data}"
ARTIFACT_CACHE_DIR="${ARTIFACT_CACHE_DIR:-${ROOT_DIR}/.bootstrap-cache}"
BOOTSTRAP_BUCKET_NAME="${BOOTSTRAP_BUCKET_NAME:-cuad-data-proj11-v2}"
BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX:-bootstrap}"
BOOTSTRAP_OBJECT_BASE_URL="${BOOTSTRAP_OBJECT_BASE_URL:-https://chi.tacc.chameleoncloud.org/project/containers/container/${BOOTSTRAP_BUCKET_NAME}/${BOOTSTRAP_PREFIX}}"

PAPERLESS_ADMIN_USER="${PAPERLESS_ADMIN_USER:-admin}"
GRAFANA_ADMIN_USER="${GRAFANA_ADMIN_USER:-admin}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-mlflow}"

TEMP_POD_NAME="model-storage-seed"
MLFLOW_SEED_POD="mlflow-seed"

log() {
  printf '\n=== %s ===\n' "$*"
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

ensure_k3s() {
  if command -v k3s >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${INSTALL_K3S}" != "1" ]]; then
    echo "k3s is not installed and INSTALL_K3S=0." >&2
    exit 1
  fi

  require_cmd curl
  log "Installing k3s"
  curl -sfL https://get.k3s.io | sh -
}

configure_kubeconfig() {
  log "Configuring kubeconfig"
  mkdir -p "${HOME}/.kube"
  sudo cp /etc/rancher/k3s/k3s.yaml "${KUBECONFIG}"
  sudo chown "$(id -u):$(id -g)" "${KUBECONFIG}"
}

disable_firewall_if_present() {
  if [[ "${DISABLE_FIREWALL}" != "1" ]]; then
    return 0
  fi

  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^firewalld\.service'; then
    if systemctl is-active --quiet firewalld; then
      log "Stopping firewalld"
      sudo systemctl disable --now firewalld
    fi
  fi

  if command -v ufw >/dev/null 2>&1; then
    local status
    status="$(sudo ufw status 2>/dev/null | head -1 || true)"
    if [[ "${status}" == "Status: active" ]]; then
      log "Disabling ufw"
      sudo ufw disable || true
    fi
  fi
}

detect_public_host() {
  local candidate=""

  if [[ -n "${PUBLIC_HOST}" ]]; then
    printf '%s' "${PUBLIC_HOST}"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    candidate="$(curl -fsS https://api.ipify.org || true)"
    if [[ -n "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  fi

  if command -v dig >/dev/null 2>&1; then
    candidate="$(dig +short myip.opendns.com @resolver1.opendns.com || true)"
    if [[ -n "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  fi

  return 1
}

create_namespaces() {
  log "Creating namespaces"
  kubectl apply -f k8s/namespace-paperless.yaml
  kubectl apply -f k8s/namespace-platform.yaml
  kubectl apply -f k8s/namespace-ml.yaml
  kubectl apply -f k8s/monitoring/namespace.yaml
}

create_runtime_secrets() {
  log "Creating runtime secrets"
  bash scripts/create-secrets.sh
}

sync_public_endpoints() {
  local host
  host="$(detect_public_host)" || {
    echo "Unable to detect a public IP. Pass one as the first argument or set PUBLIC_HOST." >&2
    exit 1
  }

  log "Syncing public endpoints to ${host}"
  bash scripts/sync-public-endpoints.sh "${host}"
}

build_and_import_images() {
  if [[ "${BUILD_IMAGES}" != "1" ]]; then
    warn "Skipping image rebuild/import because BUILD_IMAGES=${BUILD_IMAGES}"
    return 0
  fi

  require_cmd docker
  log "Building and importing local images into k3s"
  bash scripts/rebuild-k3s-images.sh
}

apply_core_manifests() {
  log "Applying Paperless manifests"
  kubectl apply -f k8s/paperless/

  log "Applying Platform manifests"
  kubectl apply -f k8s/platform/

  log "Applying Monitoring manifests"
  kubectl apply -k k8s/monitoring/

  log "Applying ML manifests"
  kubectl apply -k k8s/ml/

  log "Applying release manifests"
  kubectl apply -k k8s/release/
}

wait_for_core_rollouts() {
  log "Waiting for core rollouts"
  kubectl rollout status deployment/paperless-ngx -n paperless --timeout=300s
  kubectl rollout status deployment/minio -n platform --timeout=300s
  kubectl rollout status deployment/mlflow -n platform --timeout=300s
  kubectl rollout status deployment/prometheus -n monitoring --timeout=300s
  kubectl rollout status deployment/grafana -n monitoring --timeout=300s
  kubectl rollout status deployment/data-generator -n ml --timeout=300s
  kubectl rollout status deployment/online-features -n ml --timeout=300s
}

bootstrap_minio_bucket() {
  log "Bootstrapping MinIO bucket"
  kubectl delete job minio-bootstrap -n platform --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl apply -f k8s/platform/minio-bootstrap-job.yaml
  kubectl wait --for=condition=complete job/minio-bootstrap -n platform --timeout=180s
}

has_files() {
  local dir="$1"
  [[ -d "${dir}" ]] && find "${dir}" -mindepth 1 -print -quit | grep -q .
}

download_file() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "${output}")"
  curl -fsSL "${url}" -o "${output}"
}

resolve_baseline_file() {
  local local_path="$1"
  local remote_name="$2"
  local cache_path="${ARTIFACT_CACHE_DIR}/data/${remote_name}"

  if [[ -f "${local_path}" ]]; then
    printf '%s' "${local_path}"
    return 0
  fi

  printf 'Downloading baseline %s from object storage\n' "${remote_name}" >&2
  download_file "${BOOTSTRAP_OBJECT_BASE_URL}/data/${remote_name}" "${cache_path}"
  printf '%s' "${cache_path}"
}

resolve_model_dir() {
  local current_dir="$1"
  local archive_name="$2"
  local cache_parent="${ARTIFACT_CACHE_DIR}/models"
  local archive_path="${cache_parent}/${archive_name}.tar.gz"
  local extracted_dir="${cache_parent}/${archive_name}"

  if has_files "${current_dir}"; then
    printf '%s' "${current_dir}"
    return 0
  fi

  printf 'Downloading %s from object storage\n' "${archive_name}" >&2
  mkdir -p "${cache_parent}"
  download_file "${BOOTSTRAP_OBJECT_BASE_URL}/models/${archive_name}.tar.gz" "${archive_path}"
  rm -rf "${extracted_dir}"
  tar -xzf "${archive_path}" -C "${cache_parent}"

  if ! has_files "${extracted_dir}"; then
    warn "Downloaded archive for ${archive_name}, but extracted directory has no files."
    return 1
  fi

  printf '%s' "${extracted_dir}"
}

hydrate_bootstrap_artifacts() {
  local resolved_train=""
  local resolved_test=""
  local resolved_ner=""
  local resolved_clf=""
  local resolved_onnx=""

  if [[ ! -f "${BASELINE_DATA_DIR}/train.jsonl" || ! -f "${BASELINE_DATA_DIR}/test.jsonl" ]]; then
    resolved_train="$(resolve_baseline_file "${BASELINE_DATA_DIR}/train.jsonl" "train.jsonl" || true)"
    resolved_test="$(resolve_baseline_file "${BASELINE_DATA_DIR}/test.jsonl" "test.jsonl" || true)"
    if [[ -n "${resolved_train}" && -n "${resolved_test}" ]]; then
      BASELINE_DATA_DIR="$(dirname "${resolved_train}")"
    else
      warn "Could not hydrate baseline train/test files from object storage."
    fi
  fi

  if ! has_files "${NER_BASE_MODEL_DIR}"; then
    resolved_ner="$(resolve_model_dir "${NER_BASE_MODEL_DIR}" "deadline-ner-bert_ner_v1" || true)"
    if [[ -n "${resolved_ner}" ]]; then
      NER_BASE_MODEL_DIR="${resolved_ner}"
    fi
  fi

  if ! has_files "${CLF_BASE_MODEL_DIR}"; then
    resolved_clf="$(resolve_model_dir "${CLF_BASE_MODEL_DIR}" "deadline-clf-roberta_clf_v6" || true)"
    if [[ -n "${resolved_clf}" ]]; then
      CLF_BASE_MODEL_DIR="${resolved_clf}"
    fi
  fi

  if ! has_files "${QUANTIZED_MODEL_DIR}"; then
    resolved_onnx="$(resolve_model_dir "${QUANTIZED_MODEL_DIR}" "onnx_quantized_model" || true)"
    if [[ -n "${resolved_onnx}" ]]; then
      QUANTIZED_MODEL_DIR="${resolved_onnx}"
    fi
  fi
}

start_model_seed_pod() {
  kubectl delete pod "${TEMP_POD_NAME}" -n ml --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl run "${TEMP_POD_NAME}" \
    -n ml \
    --image=busybox:1.36 \
    --restart=Never \
    --overrides='
{
  "spec": {
    "containers": [{
      "name": "seed",
      "image": "busybox:1.36",
      "command": ["sh", "-c", "sleep 3600"],
      "volumeMounts": [{"name": "model-storage", "mountPath": "/data"}]
    }],
    "volumes": [{
      "name": "model-storage",
      "persistentVolumeClaim": {"claimName": "model-storage-pvc"}
    }]
  }
}' >/dev/null
  kubectl wait --for=condition=Ready "pod/${TEMP_POD_NAME}" -n ml --timeout=180s
}

seed_baseline_data() {
  if [[ ! -f "${BASELINE_DATA_DIR}/train.jsonl" || ! -f "${BASELINE_DATA_DIR}/test.jsonl" ]]; then
    warn "Baseline train/test JSONL files not found under ${BASELINE_DATA_DIR}; skipping data seed."
    return 0
  fi

  log "Seeding baseline train/test data into model storage PVC"
  kubectl cp "${BASELINE_DATA_DIR}/train.jsonl" "ml/${TEMP_POD_NAME}:/data/train.jsonl"
  kubectl cp "${BASELINE_DATA_DIR}/test.jsonl" "ml/${TEMP_POD_NAME}:/data/test.jsonl"
}

seed_base_models() {
  local seeded=0

  if has_files "${NER_BASE_MODEL_DIR}"; then
    log "Seeding NER base model"
    kubectl exec -n ml "${TEMP_POD_NAME}" -- mkdir -p /data/base-models
    kubectl cp "${NER_BASE_MODEL_DIR}" "ml/${TEMP_POD_NAME}:/data/base-models/deadline-ner-bert_ner_v1"
    seeded=1
  fi

  if has_files "${CLF_BASE_MODEL_DIR}"; then
    log "Seeding classifier base model"
    kubectl exec -n ml "${TEMP_POD_NAME}" -- mkdir -p /data/base-models
    kubectl cp "${CLF_BASE_MODEL_DIR}" "ml/${TEMP_POD_NAME}:/data/base-models/deadline-clf-roberta_clf_v6"
    seeded=1
  fi

  if [[ "${seeded}" != "1" ]]; then
    warn "No local base-model files found under ${MODEL_SOURCE_ROOT}. Training will fall back to remote model sources."
  fi
}

seed_quantized_models() {
  if ! has_files "${QUANTIZED_MODEL_DIR}"; then
    warn "No quantized ONNX models found at ${QUANTIZED_MODEL_DIR}. ONNX serving may not become ready until you seed them."
    return 1
  fi

  log "Seeding quantized ONNX models"
  kubectl exec -n ml "${TEMP_POD_NAME}" -- mkdir -p /data/onnx_quantized_model
  kubectl cp "${QUANTIZED_MODEL_DIR}/." "ml/${TEMP_POD_NAME}:/data/onnx_quantized_model"
  return 0
}

seed_model_storage() {
  if [[ "${SEED_MODEL_STORAGE}" != "1" ]]; then
    warn "Skipping PVC seed because SEED_MODEL_STORAGE=${SEED_MODEL_STORAGE}"
    return 0
  fi

  log "Preparing model storage seed pod"
  start_model_seed_pod
  seed_baseline_data
  seed_base_models
  local have_quantized=0
  if seed_quantized_models; then
    have_quantized=1
  fi
  kubectl delete pod "${TEMP_POD_NAME}" -n ml --ignore-not-found=true >/dev/null 2>&1 || true

  if [[ "${have_quantized}" == "1" ]]; then
    log "Restarting ONNX serving after model seed"
    kubectl rollout restart deployment/deadline-onnx-serving -n ml
    kubectl rollout status deployment/deadline-onnx-serving -n ml --timeout=300s
  fi
}

seed_mlflow_registry() {
  if [[ "${SEED_MLFLOW_REGISTRY}" != "1" ]]; then
    warn "Skipping MLflow registry seed because SEED_MLFLOW_REGISTRY=${SEED_MLFLOW_REGISTRY}"
    return 0
  fi

  if ! has_files "${QUANTIZED_MODEL_DIR}/onnx_quantized_ner" || ! has_files "${QUANTIZED_MODEL_DIR}/onnx_quantized_clf"; then
    warn "Skipping MLflow registry seed because local quantized model directories are incomplete."
    return 0
  fi

  log "Seeding MLflow registry with current production ONNX models"
  kubectl delete pod "${MLFLOW_SEED_POD}" -n ml --ignore-not-found=true >/dev/null 2>&1 || true
  kubectl run "${MLFLOW_SEED_POD}" \
    -n ml \
    --image=ghcr.io/bhanuu01/datanauts-platform-automation:latest \
    --image-pull-policy=Never \
    --restart=Never \
    --env="MLFLOW_TRACKING_URI=http://mlflow.platform.svc.cluster.local:5000" \
    --env="MLFLOW_S3_ENDPOINT_URL=http://minio.platform.svc.cluster.local:9000" \
    --env="AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}" \
    --env="AWS_SECRET_ACCESS_KEY=$(kubectl get secret platform-secrets -n platform -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 --decode)" \
    --command -- sleep 3600 >/dev/null
  kubectl wait --for=condition=Ready "pod/${MLFLOW_SEED_POD}" -n ml --timeout=180s

  kubectl cp "${QUANTIZED_MODEL_DIR}/onnx_quantized_ner" "ml/${MLFLOW_SEED_POD}:/tmp/onnx_quantized_ner"
  kubectl cp "${QUANTIZED_MODEL_DIR}/onnx_quantized_clf" "ml/${MLFLOW_SEED_POD}:/tmp/onnx_quantized_clf"

  kubectl exec -n ml "${MLFLOW_SEED_POD}" -- env \
    AWS_ACCESS_KEY_ID="${MINIO_ROOT_USER}" \
    AWS_SECRET_ACCESS_KEY="$(kubectl get secret platform-secrets -n platform -o jsonpath='{.data.MINIO_ROOT_PASSWORD}' | base64 --decode)" \
    MLFLOW_TRACKING_URI="http://mlflow.platform.svc.cluster.local:5000" \
    MLFLOW_S3_ENDPOINT_URL="http://minio.platform.svc.cluster.local:9000" \
    python -c '
import mlflow
from mlflow.tracking import MlflowClient

tracking_uri = "http://mlflow.platform.svc.cluster.local:5000"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)
mlflow.set_experiment("seeded-production-models")

pairs = [
    ("deadline-ner", "/tmp/onnx_quantized_ner", "Seeded production ONNX NER model"),
    ("deadline-classifier", "/tmp/onnx_quantized_clf", "Seeded production ONNX classifier model"),
]

for name, path, desc in pairs:
    with mlflow.start_run(run_name=f"seed-{name}") as run:
        mlflow.set_tag("seeded", "true")
        mlflow.set_tag("source", "bootstrap-production")
        mlflow.set_tag("model_name", name)
        mlflow.log_param("artifact_path", path)
        mlflow.log_artifacts(path, artifact_path="model")
        run_id = run.info.run_id
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)
    version = client.create_model_version(name=name, source=f"runs:/{run_id}/model", run_id=run_id, description=desc)
    client.transition_model_version_stage(name=name, version=version.version, stage="Production", archive_existing_versions=False)
'

  kubectl delete pod "${MLFLOW_SEED_POD}" -n ml --ignore-not-found=true >/dev/null 2>&1 || true
}

print_summary() {
  local public_ip
  public_ip="$(kubectl get configmap public-endpoints -n paperless -o jsonpath='{.data.PUBLIC_HOST}' 2>/dev/null || true)"

  log "Production bootstrap complete"
  cat <<EOF
URLs
----
Paperless:      http://${public_ip}
MLflow:         http://${public_ip}/mlflow/
MinIO Console:  http://${public_ip}:30901
Grafana:        http://${public_ip}/grafana/login
Prometheus:     http://${public_ip}/prometheus/graph

Logins
------
Paperless: ${PAPERLESS_ADMIN_USER} / ${PAPERLESS_ADMIN_PASSWORD:-<from paperless-secrets>}
Grafana:   ${GRAFANA_ADMIN_USER} / ${GRAFANA_ADMIN_PASSWORD:-<from monitoring-secrets>}
MinIO:     ${MINIO_ROOT_USER} / ${MINIO_ROOT_PASSWORD:-<from platform-secrets>}
EOF

  printf '\n'
  kubectl get pods -A
}

main() {
  require_cmd sudo
  require_cmd curl
  require_cmd tar
  ensure_k3s
  configure_kubeconfig
  require_cmd kubectl
  disable_firewall_if_present
  create_namespaces
  create_runtime_secrets
  sync_public_endpoints
  build_and_import_images
  apply_core_manifests
  wait_for_core_rollouts
  bootstrap_minio_bucket
  hydrate_bootstrap_artifacts
  seed_model_storage
  seed_mlflow_registry
  print_summary
}

main "$@"
