#!/bin/bash
set -euo pipefail

kubectl apply -f k8s/namespace.yaml -f k8s/resourcequota.yaml -f k8s/configmap.yaml

# DB credentials live only in the cluster, never in git.
if ! kubectl -n mlops-l1 get secret mlflow-db >/dev/null 2>&1; then
    kubectl -n mlops-l1 create secret generic mlflow-db \
        --from-literal=POSTGRES_DB=mlflow \
        --from-literal=POSTGRES_USER=mlflow \
        --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)"
    echo "Created secret mlflow-db with a random password"
fi

kubectl apply -f k8s/postgres.yaml -f k8s/mlflow.yaml
