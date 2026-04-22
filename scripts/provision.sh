#!/bin/bash
set -e

echo "=== Step 1: Install k3s ==="
curl -sfL https://get.k3s.io | sh -
sleep 15
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
export KUBECONFIG=~/.kube/config
echo "k3s installed."

echo "=== Step 2: Create namespaces ==="
kubectl apply -f k8s/namespace-paperless.yaml
kubectl apply -f k8s/namespace-platform.yaml
kubectl apply -f k8s/namespace-ml.yaml
kubectl apply -f k8s/monitoring/namespace.yaml

echo "=== Step 3: Create runtime secrets ==="
bash scripts/create-secrets.sh

echo "=== Step 4: Deploy Paperless stack ==="
kubectl apply -f k8s/paperless/

echo "=== Step 5: Deploy Platform stack ==="
kubectl apply -f k8s/platform/

echo "=== Step 6: Deploy Monitoring stack ==="
kubectl apply -k k8s/monitoring/

echo ""
echo "=== Waiting for pods to come up (this takes 2-3 minutes) ==="
kubectl rollout status deployment/paperless-ngx -n paperless --timeout=300s
kubectl rollout status deployment/mlflow -n platform --timeout=300s
kubectl rollout status deployment/prometheus -n monitoring --timeout=300s
kubectl rollout status deployment/grafana -n monitoring --timeout=300s

echo ""
echo "=== All done! ==="
PUBLIC_IP="$(curl -s ifconfig.me || true)"
echo "Paperless-ngx: http://${PUBLIC_IP}"
echo "MLflow:        http://${PUBLIC_IP}:30500"
echo "MinIO Console: http://${PUBLIC_IP}:30901"
echo "Grafana:       http://${PUBLIC_IP}/grafana/login"
echo "Prometheus:    http://${PUBLIC_IP}/prometheus/graph"
kubectl get pods -A
