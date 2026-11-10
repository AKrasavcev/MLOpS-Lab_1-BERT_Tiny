#!/bin/bash
set -euo pipefail

kubectl apply -f k8s/namespace.yaml -f k8s/resourcequota.yaml -f k8s/configmap.yaml

# Optional local Secrets file (git-ignored). Without it, the Secrets are created below.
if [ -f k8s/secret.yaml ]; then
    kubectl apply -f k8s/secret.yaml
fi

# DB credentials live only in the cluster, never in git.
if ! kubectl -n mlops-l1 get secret mlflow-db >/dev/null 2>&1; then
    kubectl -n mlops-l1 create secret generic mlflow-db \
        --from-literal=POSTGRES_DB=mlflow \
        --from-literal=POSTGRES_USER=mlflow \
        --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)"
    echo "Created secret mlflow-db with a random password"
fi

# Connection details for MLflow clients (training Job).
if ! kubectl -n mlops-l1 get secret mlflow-credentials >/dev/null 2>&1; then
    kubectl -n mlops-l1 create secret generic mlflow-credentials \
        --from-literal=MLFLOW_TRACKING_URI=http://mlflow.mlops-l1.svc.cluster.local:5000
    echo "Created secret mlflow-credentials"
fi

kubectl apply -f k8s/postgres.yaml -f k8s/mlflow.yaml
