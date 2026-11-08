#!/bin/bash
# Rotate the Postgres password used by MLflow.
# POSTGRES_PASSWORD only applies when the data dir is first created, so on an
# existing PVC we must change the password inside Postgres too.
set -euo pipefail
NS=mlops-l1
NEW_PASSWORD="$(openssl rand -hex 24)"
DB_USER="$(kubectl -n $NS get secret mlflow-db -o jsonpath='{.data.POSTGRES_USER}' 2>/dev/null | base64 -d || echo mlflow)"

kubectl -n $NS exec deploy/postgres -- \
    psql -U "$DB_USER" -d postgres -c "ALTER USER \"$DB_USER\" WITH PASSWORD '$NEW_PASSWORD';"

kubectl -n $NS create secret generic mlflow-db \
    --from-literal=POSTGRES_DB=mlflow \
    --from-literal=POSTGRES_USER="$DB_USER" \
    --from-literal=POSTGRES_PASSWORD="$NEW_PASSWORD" \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl -n $NS rollout restart deploy/postgres deploy/mlflow
kubectl -n $NS rollout status deploy/mlflow --timeout=300s
echo "Password rotated."
