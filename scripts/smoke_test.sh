#!/bin/bash
# Smoke test of the deployed API: /health is ok (and serves the expected model version), /predict answers.
#   scripts/smoke_test.sh            # any version
#   scripts/smoke_test.sh 17         # fail unless /health reports model_version 17
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/k8s_lib.sh
EXPECTED="${1:-}"
PORT="${API_PORT:-18000}"

# Retry with a fresh port-forward: right after a rollout it may still land on a terminating old pod.
for attempt in $(seq 1 6); do
    port_forward "$DEPLOY" "$PORT" 8000 /health || true
    HEALTH="$(curl -s "http://127.0.0.1:$PORT/health" || true)"
    VERSION="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d["model_version"] if d.get("status")=="ok" else "")' \
               "$HEALTH" 2>/dev/null || true)"
    if [ -n "$VERSION" ] && { [ -z "$EXPECTED" ] || [ "$VERSION" = "$EXPECTED" ]; }; then
        break
    fi
    echo "attempt $attempt: /health -> ${HEALTH:-no answer} (want version ${EXPECTED:-any})"
    _cleanup_port_forwards; _PF_PIDS=()
    VERSION=""
    sleep 5
done
[ -n "$VERSION" ] || { echo "SMOKE FAIL: /health not ok or wrong model version" >&2; exit 1; }
echo "/health: $HEALTH"

PRED="$(curl -sf -X POST "http://127.0.0.1:$PORT/predict" -H 'Content-Type: application/json' \
        -d '{"text":"This movie is amazing and really enjoyable."}')" \
    || { echo "SMOKE FAIL: /predict failed" >&2; exit 1; }
echo "/predict: $PRED"
echo "SMOKE OK: model_version=$VERSION"
