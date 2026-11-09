#!/bin/bash
# R4 measurements in one go: cold start, image size, model size, then the load test.
# Writes docs/load-test.md and docs/load-test.json.
#   bash scripts/load_test.sh [requests] [concurrency]     (default 500 10)
# Needs: the API Deployment applied (k8s/deployment-cpu.yaml), MLflow running, a @champion alias.
set -euo pipefail
cd "$(dirname "$0")/.."

NS=mlops-l1
DEPLOY=mlops-l1-bert-tiny
PORT=${PORT:-8000}
N=${1:-500}
C=${2:-10}
PY=${PY:-.venv/bin/python}

IMAGE="$(kubectl -n $NS get deploy $DEPLOY -o jsonpath='{.spec.template.spec.containers[0].image}')"

# Image size from the local Docker copy (the same image is loaded into Minikube, which runs containerd).
IMAGE_BYTES="$(docker image inspect "$IMAGE" --format '{{.Size}}' 2>/dev/null || true)"
if [[ "$IMAGE_BYTES" =~ ^[0-9]+$ ]]; then IMAGE_SIZE="$((IMAGE_BYTES / 1000 / 1000)) MB"; else IMAGE_SIZE="n/a"; fi

# Cold start: new pod scheduled -> model downloaded from the registry -> /health ready.
echo "Restarting $DEPLOY to measure cold start..."
T0=$(date +%s.%N)
kubectl -n $NS rollout restart deployment/$DEPLOY
kubectl -n $NS rollout status deployment/$DEPLOY --timeout=5m
COLD_START="$(echo "$(date +%s.%N) - $T0" | bc | xargs printf '%.1f s')"

POD="$(kubectl -n $NS get pod -l app=$DEPLOY --field-selector=status.phase=Running \
        --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')"
MODEL_SIZE="$(kubectl -n $NS exec "$POD" -- du -sm /tmp/model | cut -f1) MB"

kubectl -n $NS port-forward svc/$DEPLOY "$PORT":8000 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
for _ in $(seq 1 20); do curl -sf "http://localhost:$PORT/health" >/dev/null && break; sleep 1; done

"$PY" scripts/load_test.py --url "http://localhost:$PORT" -n "$N" -c "$C" \
    --meta "image=\`$IMAGE\`" \
    --meta "image_size=$IMAGE_SIZE (uncompressed)" \
    --meta "model_artifact_size=$MODEL_SIZE (MLflow artifact, downloaded at start)" \
    --meta "cold_start=$COLD_START (rollout restart -> pod Ready, incl. model download)" \
    --json-out docs/load-test.json --md-out docs/load-test.md

# Peak memory of the pod (cgroup v2) after the test, to size the 1Gi limit. No metrics-server needed.
PEAK_MIB="$(kubectl -n $NS exec "$POD" -- cat /sys/fs/cgroup/memory.peak 2>/dev/null | awk '{printf "%d", $1/1048576}' || true)"
if [ -n "$PEAK_MIB" ]; then
    echo "| Pod memory peak | $PEAK_MIB MiB (limit 1Gi) |" >> docs/load-test.md
    echo "pod memory peak: $PEAK_MIB MiB"
fi
