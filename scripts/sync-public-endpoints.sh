#!/bin/bash
set -euo pipefail

PUBLIC_HOST="${1:-${PUBLIC_HOST:-}}"

detect_public_host() {
  local candidate=""

  if command -v curl >/dev/null 2>&1; then
    candidate="$(curl -fsS https://api.ipify.org || true)"
    if [[ -n "${candidate}" ]]; then
      printf "%s" "${candidate}"
      return 0
    fi
  fi

  if command -v dig >/dev/null 2>&1; then
    candidate="$(dig +short myip.opendns.com @resolver1.opendns.com || true)"
    if [[ -n "${candidate}" ]]; then
      printf "%s" "${candidate}"
      return 0
    fi
  fi

  return 1
}

if [[ -z "${PUBLIC_HOST}" ]]; then
  PUBLIC_HOST="$(detect_public_host)" || {
    echo "Unable to detect a public IP automatically." >&2
    echo "Pass it explicitly: bash scripts/sync-public-endpoints.sh 129.114.x.x" >&2
    exit 1
  }
fi

PUBLIC_BASE_URL="http://${PUBLIC_HOST}"

for namespace in paperless; do
  kubectl create configmap public-endpoints \
    --namespace="${namespace}" \
    --from-literal=PUBLIC_HOST="${PUBLIC_HOST}" \
    --from-literal=PUBLIC_BASE_URL="${PUBLIC_BASE_URL}" \
    --dry-run=client -o yaml | kubectl apply -f -
done

echo "Public endpoint config synced:"
echo "  host: ${PUBLIC_HOST}"
echo "  base url: ${PUBLIC_BASE_URL}"
