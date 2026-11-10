#!/bin/bash
# Build the training image, load it into Minikube and run the training Job to completion.
#   scripts/train_job.sh            # TRAIN_MODE=train (one run + evaluate)
#   scripts/train_job.sh hpo        # Optuna HPO (nested runs)
#   SKIP_BUILD=1 scripts/train_job.sh   # reuse the image already in Minikube (e.g. after a ConfigMap edit)
#   IMAGE=ghcr.io/akrasavcev/bert-tiny-train:sha-<12> SKIP_BUILD=1 scripts/train_job.sh   # CD: pinned image
# In train mode the run id and test metrics are saved to $OUT_DIR (default model/job/):
#   run_id.txt, metrics.json   (input for scripts/quality_gate.py and src/register.py)
set -euo pipefail
cd "$(dirname "$0")/.."

NS=mlops-l1
JOB=bert-tiny-train
MODE="${1:-train}"
IMAGE="${IMAGE:-bert-tiny-train:latest}"
OUT_DIR="${OUT_DIR:-model/job}"
COMMIT="${GIT_COMMIT:-$(git rev-parse --short=8 HEAD)}"

if [ -z "${SKIP_BUILD:-}" ]; then
    docker build -f docker/Dockerfile.train --build-arg GIT_COMMIT="$COMMIT" -t "$IMAGE" .
    minikube image load "$IMAGE" --overwrite=true
fi

kubectl -n $NS get secret mlflow-credentials >/dev/null \
    || { echo "Secret mlflow-credentials missing: run scripts/k8_deploy.sh" >&2; exit 1; }
kubectl apply -f k8s/train-configmap.yaml
kubectl -n $NS delete job $JOB --ignore-not-found --wait=true

# Same manifest, with the image, mode and commit filled in.
sed -e "s#^\([[:space:]]*image:\).*#\1 $IMAGE#" \
    -e "/name: TRAIN_MODE/{n;s/value: .*/value: \"$MODE\"/}" \
    -e "/name: GIT_COMMIT/{n;s/value: .*/value: \"$COMMIT\"/}" \
    k8s/train-job.yaml | kubectl apply -f -

kubectl -n $NS wait --for=condition=ready pod -l job-name=$JOB --timeout=5m || true
kubectl -n $NS logs -f job/$JOB || true

# Pick the run id and metrics out of the logs of the pod that succeeded (backoffLimit 1 may leave a failed one).
save_results() {
    local pod
    pod="$(kubectl -n $NS get pod -l job-name=$JOB --field-selector=status.phase=Succeeded \
            --sort-by=.metadata.creationTimestamp -o name | tail -1)"
    [ -n "$pod" ] || return 0
    mkdir -p "$OUT_DIR"
    local logs
    logs="$(kubectl -n $NS logs "$pod")"
    sed -n 's/^CD_RUN_ID=//p' <<<"$logs" | tail -1 > "$OUT_DIR/run_id.txt"
    sed -n 's/^CD_METRICS=//p' <<<"$logs" | tail -1 > "$OUT_DIR/metrics.json"
    if [ -s "$OUT_DIR/run_id.txt" ]; then
        echo "run_id=$(cat "$OUT_DIR/run_id.txt") -> $OUT_DIR/"
    else
        rm -f "$OUT_DIR/run_id.txt" "$OUT_DIR/metrics.json"   # e.g. hpo mode: no single run
    fi
}

# Logs end when the container exits; then the Job is Complete or Failed (or retrying, backoffLimit 1).
for _ in $(seq 1 120); do
    STATUS="$(kubectl -n $NS get job $JOB -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type} {end}')"
    case "$STATUS" in
        *Complete*) kubectl -n $NS get job $JOB; save_results; exit 0 ;;
        *Failed*)   kubectl -n $NS get job $JOB; echo "Job failed" >&2; exit 1 ;;
    esac
    sleep 30
done
echo "Job still running; check: kubectl -n $NS get job $JOB" >&2
exit 1
