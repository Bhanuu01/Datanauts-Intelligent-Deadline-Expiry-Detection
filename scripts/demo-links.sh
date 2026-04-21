#!/bin/bash
set -euo pipefail

PUBLIC_IP="${1:-129.114.27.190}"

cat <<EOF
Demo URLs
---------
Paperless:      http://${PUBLIC_IP}
MLflow:         http://${PUBLIC_IP}:30500
MinIO Console:  http://${PUBLIC_IP}:30901
Grafana:        http://${PUBLIC_IP}/grafana/login
Prometheus:     http://${PUBLIC_IP}/prometheus/graph

Logins
------
Paperless: admin / <paperless admin password from secret>
Grafana:   admin / admin
MinIO:     mlflow / <platform secret password>
EOF
