#!/bin/bash
set -euo pipefail

IMAGE_FILTER="${1:-all}"
PRUNE_AFTER_IMPORT="${PRUNE_AFTER_IMPORT:-1}"

declare -A IMAGE_TO_DOCKERFILE=(
  ["ghcr.io/bhanuu01/datanauts-online-features:latest"]="components/data/online_features/Dockerfile"
  ["ghcr.io/bhanuu01/datanauts-data-monitoring:latest"]="components/data/evaluation_monitoring/Dockerfile"
  ["ghcr.io/bhanuu01/datanauts-platform-automation:latest"]="components/platform_automation/Dockerfile"
  ["ghcr.io/bhanuu01/datanauts-inference-runtime:latest"]="components/inference_service/Dockerfile.runtime"
  ["ghcr.io/bhanuu01/datanauts-onnx-serving:latest"]="components/serving/Dockerfile.onnx_quant"
)

declare -A IMAGE_TO_CONTEXT=(
  ["ghcr.io/bhanuu01/datanauts-online-features:latest"]="components/data/online_features"
  ["ghcr.io/bhanuu01/datanauts-data-monitoring:latest"]="components/data/evaluation_monitoring"
  ["ghcr.io/bhanuu01/datanauts-platform-automation:latest"]="."
  ["ghcr.io/bhanuu01/datanauts-inference-runtime:latest"]="."
  ["ghcr.io/bhanuu01/datanauts-onnx-serving:latest"]="."
)

build_and_import() {
  local image="$1"
  local dockerfile="$2"
  local context="$3"
  local tarball="/tmp/$(basename "${image%%:*}").tar"

  echo "=== Building ${image} ==="
  docker build -t "$image" -f "$dockerfile" "$context"

  echo "=== Importing ${image} into k3s ==="
  docker save "$image" -o "$tarball"
  sudo k3s ctr images import "$tarball"
  rm -f "$tarball"

  if [[ "$PRUNE_AFTER_IMPORT" == "1" ]]; then
    echo "=== Pruning Docker artifacts after importing ${image} ==="
    docker image rm -f "$image" >/dev/null 2>&1 || true
    sudo docker system prune -af >/dev/null 2>&1 || true
  fi
}

for image in "${!IMAGE_TO_DOCKERFILE[@]}"; do
  if [[ "$IMAGE_FILTER" != "all" && "$image" != *"$IMAGE_FILTER"* ]]; then
    continue
  fi

  build_and_import "$image" "${IMAGE_TO_DOCKERFILE[$image]}" "${IMAGE_TO_CONTEXT[$image]}"
done

echo
echo "=== Imported Datanauts images visible to k3s ==="
sudo k3s ctr images ls | grep datanauts || true
