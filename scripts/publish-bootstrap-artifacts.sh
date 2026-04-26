#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BUCKET_NAME="${BUCKET_NAME:-cuad-data-proj11-v2}"
BOOTSTRAP_PREFIX="${BOOTSTRAP_PREFIX:-bootstrap}"
MODEL_SOURCE_ROOT="${MODEL_SOURCE_ROOT:-${ROOT_DIR}/models}"
QUANTIZED_MODEL_DIR="${QUANTIZED_MODEL_DIR:-${MODEL_SOURCE_ROOT}/onnx_quantized_model}"
NER_BASE_MODEL_DIR="${NER_BASE_MODEL_DIR:-${MODEL_SOURCE_ROOT}/deadline-ner-bert_ner_v1}"
CLF_BASE_MODEL_DIR="${CLF_BASE_MODEL_DIR:-${MODEL_SOURCE_ROOT}/deadline-clf-roberta_clf_v6}"
BASELINE_DATA_DIR="${BASELINE_DATA_DIR:-${ROOT_DIR}/components/data/gx_quality/data}"
STAGING_DIR="${STAGING_DIR:-${ROOT_DIR}/.artifact-staging}"
MAKE_BUCKET_PUBLIC="${MAKE_BUCKET_PUBLIC:-1}"

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

has_files() {
  local dir="$1"
  [[ -d "${dir}" ]] && find "${dir}" -mindepth 1 -type f -print -quit | grep -q .
}

stage_dir_archive() {
  local source_dir="$1"
  local archive_path="$2"
  local parent
  local name

  parent="$(dirname "${source_dir}")"
  name="$(basename "${source_dir}")"
  tar -czf "${archive_path}" -C "${parent}" "${name}"
}

upload_file() {
  local local_path="$1"
  local object_name="$2"
  log "Uploading ${object_name}"
  openstack object create --name "${object_name}" "${BUCKET_NAME}" "${local_path}" >/dev/null
}

main() {
  require_cmd openstack
  require_cmd tar

  rm -rf "${STAGING_DIR}"
  mkdir -p "${STAGING_DIR}"

  log "Ensuring object storage container ${BUCKET_NAME} exists"
  openstack container create "${BUCKET_NAME}" >/dev/null || true
  if [[ "${MAKE_BUCKET_PUBLIC}" == "1" ]]; then
    openstack container set --property 'X-Container-Read=.r:*,.rlistings' "${BUCKET_NAME}"
  fi

  if [[ -f "${BASELINE_DATA_DIR}/train.jsonl" ]]; then
    upload_file "${BASELINE_DATA_DIR}/train.jsonl" "${BOOTSTRAP_PREFIX}/data/train.jsonl"
  else
    warn "Missing ${BASELINE_DATA_DIR}/train.jsonl"
  fi

  if [[ -f "${BASELINE_DATA_DIR}/test.jsonl" ]]; then
    upload_file "${BASELINE_DATA_DIR}/test.jsonl" "${BOOTSTRAP_PREFIX}/data/test.jsonl"
  else
    warn "Missing ${BASELINE_DATA_DIR}/test.jsonl"
  fi

  if has_files "${NER_BASE_MODEL_DIR}"; then
    stage_dir_archive "${NER_BASE_MODEL_DIR}" "${STAGING_DIR}/deadline-ner-bert_ner_v1.tar.gz"
    upload_file "${STAGING_DIR}/deadline-ner-bert_ner_v1.tar.gz" "${BOOTSTRAP_PREFIX}/models/deadline-ner-bert_ner_v1.tar.gz"
  else
    warn "Missing NER base model directory: ${NER_BASE_MODEL_DIR}"
  fi

  if has_files "${CLF_BASE_MODEL_DIR}"; then
    stage_dir_archive "${CLF_BASE_MODEL_DIR}" "${STAGING_DIR}/deadline-clf-roberta_clf_v6.tar.gz"
    upload_file "${STAGING_DIR}/deadline-clf-roberta_clf_v6.tar.gz" "${BOOTSTRAP_PREFIX}/models/deadline-clf-roberta_clf_v6.tar.gz"
  else
    warn "Missing classifier base model directory: ${CLF_BASE_MODEL_DIR}"
  fi

  if has_files "${QUANTIZED_MODEL_DIR}"; then
    stage_dir_archive "${QUANTIZED_MODEL_DIR}" "${STAGING_DIR}/onnx_quantized_model.tar.gz"
    upload_file "${STAGING_DIR}/onnx_quantized_model.tar.gz" "${BOOTSTRAP_PREFIX}/models/onnx_quantized_model.tar.gz"
  else
    warn "Missing quantized model directory: ${QUANTIZED_MODEL_DIR}"
  fi

  cat <<EOF

Bootstrap artifacts published.

Expected public URLs:
- https://chi.tacc.chameleoncloud.org/project/containers/container/${BUCKET_NAME}/${BOOTSTRAP_PREFIX}/data/train.jsonl
- https://chi.tacc.chameleoncloud.org/project/containers/container/${BUCKET_NAME}/${BOOTSTRAP_PREFIX}/data/test.jsonl
- https://chi.tacc.chameleoncloud.org/project/containers/container/${BUCKET_NAME}/${BOOTSTRAP_PREFIX}/models/deadline-ner-bert_ner_v1.tar.gz
- https://chi.tacc.chameleoncloud.org/project/containers/container/${BUCKET_NAME}/${BOOTSTRAP_PREFIX}/models/deadline-clf-roberta_clf_v6.tar.gz
- https://chi.tacc.chameleoncloud.org/project/containers/container/${BUCKET_NAME}/${BOOTSTRAP_PREFIX}/models/onnx_quantized_model.tar.gz
EOF
}

main "$@"
