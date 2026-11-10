#!/bin/sh
# Entrypoint of the training Job: download -> prepare -> train -> evaluate (or HPO).
#   TRAIN_MODE=train  one run, then evaluate_model.py (writes model/metrics.json)
#   TRAIN_MODE=hpo    Optuna study, one parent run with nested trial runs
# Hyperparameters come from /app/params.yaml (mounted from the ConfigMap).
set -eu

: "${MLFLOW_TRACKING_URI:?MLFLOW_TRACKING_URI must be set (Secret mlflow-credentials)}"
MODE="${TRAIN_MODE:-train}"
echo "mode=$MODE git_commit=${GIT_COMMIT:-unknown} tracking=$MLFLOW_TRACKING_URI"

# Fail fast with a clear message if MLflow is not reachable (up to about 2 min).
python - <<'EOF'
import os, sys, time, urllib.request
url = os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/health"
for attempt in range(24):
    try:
        urllib.request.urlopen(url, timeout=5)
        print(f"MLflow is up: {url}")
        sys.exit(0)
    except Exception as exc:
        print(f"waiting for MLflow ({exc})")
        time.sleep(5)
sys.exit(f"MLflow not reachable at {url}")
EOF

python src/create_raw_dataset.py
# Same hash `dvc add` stores for a single file, so the run can be matched to data/raw_dataset.csv.dvc.
DVC_RAW_MD5="$(md5sum data/raw_dataset.csv | cut -d' ' -f1)"
export DVC_RAW_MD5
echo "raw dataset md5=$DVC_RAW_MD5"
python src/prepare_dataset.py

case "$MODE" in
    train)
        python src/train.py
        python src/evaluate_model.py
        # One-line results for CD: scripts/train_job.sh picks these out of the pod logs.
        echo "CD_RUN_ID=$(cat model/run_id.txt)"
        python -c 'import json; print("CD_METRICS=" + json.dumps(json.load(open("model/metrics.json"))))'
        ;;
    hpo)
        python src/hpo.py
        ;;
    *)
        echo "unknown TRAIN_MODE=$MODE (expected train or hpo)" >&2
        exit 2
        ;;
esac
