"""Reproduce a historical MLflow run: same code (git), same data (dvc), same params -> re-train -> compare.

  time python reproduce.py <run_id> [--tolerance 0.005]

Prerequisites: a clean working tree on a branch, the data in the local DVC cache
(else `dvc pull` first), and MLFLOW_TRACKING_URI pointing at the server that has the run.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parent
TRAIN_KEYS = {
    "max_length": int, 
    "epochs": int, 
    "learning_rate": float, 
    "train_batch_size": int, 
    "eval_batch_size": int, 
    "weight_decay": float, 
    "seed": int
}


def sh(*cmd: str) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--tolerance", type=float, default=0.005)
    args = ap.parse_args()
    t0 = time.time()

    run = mlflow.get_run(args.run_id)                       # 0. the record
    tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
    commit = tags.get("mlflow.source.git.commit") or tags.get("git_commit")
    if not commit or commit == "unknown":
        sys.exit("run has no git commit tag - cannot reproduce")

    start_ref = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
    ).strip()
    ok = False
    try:
        sh("git", "checkout", "--quiet", commit)            # 1. code + .dvc pointers of that moment

        # 2. exact data bytes from the cache - only the .dvc pointer files: the historical
        #    commit predates its dvc.lock (the lock is committed after the run), so the
        #    pipeline outputs cannot be restored - they are rebuilt by `dvc repro` below.
        pointers = [
            str(p.relative_to(ROOT))
            for p in ROOT.rglob("*.dvc")
            if p.is_file() and p.name != ".dvc"
        ]
        if pointers:
            sh("dvc", "checkout", "--force", *pointers)

        cfg = yaml.safe_load((ROOT / "params.yaml").read_text())  # 3. same hyper-parameters
        cfg["train"].update({k: cast(params[k]) for k, cast in TRAIN_KEYS.items() if k in params})
        (ROOT / "params.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

        sh("dvc", "repro", "--force")                       # 4. re-run the DAG -> train.py logs a new run
        new_id = (ROOT / "model/run_id.txt").read_text().strip()
        new = mlflow.get_run(new_id)
        mlflow.MlflowClient().set_tag(new_id, "reproduces", args.run_id)

        ok = True                                           # 5. compare
        for key in ("accuracy", "f1"):
            delta = abs(new.data.metrics[key] - metrics[key])
            ok &= delta <= args.tolerance
            print(f"{key}: original={metrics[key]:.4f} reproduced={new.data.metrics[key]:.4f} delta={delta:.4f}")
    finally:
        sh("git", "checkout", "--quiet", "--force", start_ref)  # 6. leave the repo as we found it
        sh("dvc", "checkout", "--force")

    print(f"{'REPRODUCED' if ok else 'MISMATCH'} in {time.time() - t0:.0f}s -> new run {new_id}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
