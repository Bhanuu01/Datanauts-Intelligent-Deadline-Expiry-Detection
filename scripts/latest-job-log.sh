#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <namespace> <job-name-fragment> [tail-lines]"
  exit 1
fi

NAMESPACE="$1"
FRAGMENT="$2"
TAIL_LINES="${3:-120}"

JOB_NAME="$(
  kubectl get jobs -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o custom-columns=NAME:.metadata.name --no-headers \
    | grep "${FRAGMENT}" \
    | tail -n 1
)"

if [[ -z "${JOB_NAME}" ]]; then
  echo "No job found in namespace '${NAMESPACE}' matching '${FRAGMENT}'."
  exit 1
fi

echo "=== Latest job: ${JOB_NAME} ==="
kubectl logs -n "${NAMESPACE}" "job/${JOB_NAME}" --tail="${TAIL_LINES}"
