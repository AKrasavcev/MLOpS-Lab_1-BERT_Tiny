#!/bin/bash
# Build the training image, load it into Minikube and run the training Job to completion.
#   scripts/train_job.sh            # TRAIN_MODE=train (one run + evaluate)
#   scripts/train_job.sh hpo        # Optuna HPO (nested runs)
#   SKIP_BUILD=1 scripts/train_job.sh   # reuse the image already in Minikube (e.g. after a ConfigMap edit)
set -euo pipefail
cd "$(dirname "$0")/.."

NS=mlops-l1
JOB=bert-tiny-train
MODE="${1:-train}"
IMAGE=bert-tiny-train:latest
COMMIT="$(git rev-parse --short=8 HEAD)"

if [ -z "${SKIP_BUILD:-}" ]; then
    docker build -f docker/Dockerfile.train --build-arg GIT_COMMIT="$COMMIT" -t "$IMAGE" .
    minikube image load "$IMAGE" --overwrite=true
fi

kubectl -n $NS get secret mlflow-credentials >/dev/null \
    || { echo "Secret mlflow-credentials missing: run scripts/k8_deploy.sh" >&2; exit 1; }
kubectl apply -f k8s/train-configmap.yaml
kubectl -n $NS delete job $JOB --ignore-not-found --wait=true

# Same manifest, with the mode and commit filled in.
sed -e "/name: TRAIN_MODE/{n;s/value: .*/value: \"$MODE\"/}" \
    -e "/name: GIT_COMMIT/{n;s/value: .*/value: \"$COMMIT\"/}" \
    k8s/train-job.yaml | kubectl apply -f -

kubectl -n $NS wait --for=condition=ready pod -l job-name=$JOB --timeout=5m || true
kubectl -n $NS logs -f job/$JOB || true

# Logs end when the container exits; then the Job is Complete or Failed (or retrying, backoffLimit 1).
for _ in $(seq 1 120); do
    STATUS="$(kubectl -n $NS get job $JOB -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type} {end}')"
    case "$STATUS" in
        *Complete*) kubectl -n $NS get job $JOB; exit 0 ;;
        *Failed*)   kubectl -n $NS get job $JOB; echo "Job failed" >&2; exit 1 ;;
    esac
    sleep 30
done
echo "Job still running; check: kubectl -n $NS get job $JOB" >&2
exit 1
