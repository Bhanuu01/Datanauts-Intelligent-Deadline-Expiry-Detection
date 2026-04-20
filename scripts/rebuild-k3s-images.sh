#!/bin/bash
set -euo pipefail

IMAGE_FILTER="${1:-all}"

declare -A IMAGE_TO_DOCKERFILE=(
  ["ghcr.io/bhanuu01/datanauts-online-features:latest"]="components/data/online_features/Dockerfile"
  ["ghcr.io/bhanuu01/datanauts-data-monitoring:latest"]="components/data/evaluation_monitoring/Dockerfile"
  ["ghcr.io/bhanuu01/datanauts-platform-automation:latest"]="components/platform_automation/Dockerfile"
  ["ghcr.io/bhanuu01/datanauts-inference-runtime:latest"]="components/inference_service/Dockerfile.runtime"
)

build_and_import() {
  local image="$1"
  local dockerfile="$2"
  local tarball="/tmp/$(basename "${image%%:*}").tar"

  echo "=== Building ${image} ==="
  docker build -t "$image" -f "$dockerfile" .

  echo "=== Importing ${image} into k3s ==="
  docker save "$image" -o "$tarball"
  sudo k3s ctr images import "$tarball"
  rm -f "$tarball"
}

for image in "${!IMAGE_TO_DOCKERFILE[@]}"; do
  if [[ "$IMAGE_FILTER" != "all" && "$image" != *"$IMAGE_FILTER"* ]]; then
    continue
  fi

  build_and_import "$image" "${IMAGE_TO_DOCKERFILE[$image]}"
done

echo
echo "=== Imported Datanauts images visible to k3s ==="
sudo k3s ctr images ls | grep datanauts || true
echo
echo "=== Optional cleanup to reduce disk pressure ==="
echo "sudo docker system prune -af"
