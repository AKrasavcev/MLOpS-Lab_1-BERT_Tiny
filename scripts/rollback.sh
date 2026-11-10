#!/bin/bash
# Roll the API back to an older model version (and optionally the previous image).
#   scripts/rollback.sh                    # @champion -> the @previous version, rollout restart
#   scripts/rollback.sh 12                 # @champion -> v12, rollout restart
#   scripts/rollback.sh --undo-image [12]  # also `kubectl rollout undo` (previous image) instead of a restart
# Image-only rollback, aliases untouched:  kubectl -n mlops-l1 rollout undo deployment/mlops-l1-bert-tiny
# The rolled-back version becomes @previous, so running it twice returns to where you started.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/k8s_lib.sh

UNDO_IMAGE=""
VERSION=""
for arg in "$@"; do
    case "$arg" in
        --undo-image) UNDO_IMAGE=1 ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) VERSION="$arg" ;;
    esac
done

ensure_mlflow
OUT="$(mktemp)"
"$PY" src/register.py --rollback ${VERSION:+--to-version "$VERSION"} --output "$OUT"
CHAMPION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["champion"])' "$OUT")"
rm -f "$OUT"

if [ -n "$UNDO_IMAGE" ]; then
    kubectl -n "$NS" rollout undo "deployment/$DEPLOY"
else
    kubectl -n "$NS" rollout restart "deployment/$DEPLOY"
fi
kubectl -n "$NS" rollout status "deployment/$DEPLOY" --timeout=5m
kubectl -n "$NS" get deploy "$DEPLOY" -o jsonpath='image: {.spec.template.spec.containers[0].image}{"\n"}'

bash scripts/smoke_test.sh "$CHAMPION"
