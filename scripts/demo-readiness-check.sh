#!/bin/bash
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

echo "=== Core Pods ==="
kubectl get pods -n paperless
kubectl get pods -n platform
kubectl get pods -n monitoring
kubectl get pods -n ml
echo

echo "=== Active Serving Deployments ==="
kubectl get deploy -n ml | grep onnx-serving || true
kubectl get hpa -n ml || true
echo

echo "=== Recent Pipeline Jobs ==="
kubectl get cronjobs -n ml
kubectl get jobs -n ml
echo

echo "=== Recent Traffic ==="
kubectl logs -n ml deploy/data-generator --tail=12 || true
echo

echo "=== Production Logs ==="
kubectl exec -n ml deploy/online-features -- sh -c 'tail -n 5 /data/production_ingest.jsonl' || true
echo "---"
kubectl exec -n ml deploy/online-features -- sh -c 'tail -n 5 /data/feedback_events.jsonl' || true
echo

echo "=== Decision Artifacts ==="
kubectl exec -n ml deploy/online-features -- sh -c 'test -f /data/retrain_decision.json && echo retrain_decision.json present || echo retrain_decision.json missing'
kubectl exec -n ml deploy/online-features -- sh -c 'test -f /data/promotion_decision.json && echo promotion_decision.json present || echo promotion_decision.json missing'
echo

echo "=== Demo Dashboards ==="
cat <<EOF
Open these in Grafana:
- Datanauts Overview
- Datanauts Serving
- Datanauts Data & Feedback
- Datanauts Platform Health
EOF
