#!/bin/bash
# Shared helpers for the CD / rollback scripts. Source it: `source scripts/k8s_lib.sh`.
NS="${NS:-mlops-l1}"
DEPLOY="${DEPLOY:-mlops-l1-bert-tiny}"
PY="${PY:-.venv/bin/python}"
_PF_PIDS=()

_cleanup_port_forwards() { for pid in "${_PF_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap _cleanup_port_forwards EXIT

# port_forward <svc> <local_port> <remote_port> <health_path>: background port-forward, wait until it answers.
port_forward() {
    local svc=$1 lport=$2 rport=$3 path=$4
    kubectl -n "$NS" port-forward "svc/$svc" "$lport:$rport" >/dev/null 2>&1 &
    _PF_PIDS+=($!)
    for _ in $(seq 1 30); do
        curl -s -o /dev/null "http://127.0.0.1:$lport$path" && return 0
        sleep 1
    done
    echo "port-forward to svc/$svc on :$lport did not come up" >&2
    return 1
}

# Use MLFLOW_TRACKING_URI if it answers, otherwise port-forward the in-cluster MLflow.
ensure_mlflow() {
    if [ -n "${MLFLOW_TRACKING_URI:-}" ] && curl -sf "${MLFLOW_TRACKING_URI%/}/health" >/dev/null; then
        return 0
    fi
    local port="${MLFLOW_PORT:-15000}"
    port_forward mlflow "$port" 5000 /health
    export MLFLOW_TRACKING_URI="http://127.0.0.1:$port"
    echo "MLflow via port-forward: $MLFLOW_TRACKING_URI"
}
