#!/bin/bash
set -e

echo "=== Step 1: Install k3s ==="
curl -sfL https://get.k3s.io | sh -
sleep 15
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
echo "k3s installed."

echo "=== Step 2: Create namespaces ==="
kubectl apply -f k8s/namespace-paperless.yaml
kubectl apply -f k8s/namespace-platform.yaml

echo "=== Step 3: Create secrets (edit the script first!) ==="
bash scripts/create-secrets.sh

echo "=== Step 4: Deploy Paperless stack ==="
kubectl apply -f k8s/paperless/

echo "=== Step 5: Deploy Platform stack ==="
kubectl apply -f k8s/platform/

echo ""
echo "=== Waiting for pods to come up (this takes 2-3 minutes) ==="
kubectl rollout status deployment/paperless-ngx -n paperless --timeout=300s
kubectl rollout status deployment/mlflow -n platform --timeout=300s

echo ""
echo "=== All done! ==="
echo "Paperless-ngx: http://$(curl -s ifconfig.me)"
echo "MLflow:        http://$(curl -s ifconfig.me)/mlflow"
kubectl get pods -A