#!/bin/bash
set -euo pipefail

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required on the current machine."
  exit 1
fi

echo "=== Cluster ==="
kubectl get nodes
echo

echo "=== Namespaces ==="
kubectl get ns
echo

echo "=== Pods ==="
kubectl get pods -A
echo

echo "=== CronJobs ==="
kubectl get cronjobs -A
echo

echo "=== Recent Jobs ==="
kubectl get jobs -A
echo

echo "=== Services ==="
kubectl get svc -A
echo

echo "=== Ingress ==="
kubectl get ingress -A
echo

echo "=== Persistent Volume Claims ==="
kubectl get pvc -A
echo

echo "=== Resource Usage ==="
kubectl top nodes || true
kubectl top pods -A || true
echo

echo "=== Recent Events ==="
kubectl get events -A --sort-by=.lastTimestamp | tail -n 40 || true
