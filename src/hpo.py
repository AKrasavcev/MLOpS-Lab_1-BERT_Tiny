"""Optuna HPO: one parent MLflow run, every trial a nested child run.

  python src/hpo.py [--n-trials 6]

Search space and trial count live in params.yaml -> hpo. The base config (epochs, max_length,
optimizer, ...) comes from params.yaml -> train. The best trial's run_id is written to
model/hpo_best.json and tagged on the parent run as best_run_id (input for register.py).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime

import mlflow
import optuna

from train import (
    DEFAULT_MODEL_DIR,
    config_from_params,
    git_commit,
    load_params,
    setup_mlflow,
    train_once,
)


def suggest(trial: optuna.Trial, space: dict) -> dict[str, object]:
    """Turn the params.yaml search space into Optuna suggestions.

    A list -> categorical. A dict with low/high -> float (log=True for log scale).
    """
    values: dict[str, object] = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            values[name] = trial.suggest_categorical(name, spec)
        else:
            values[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
    return values


def main() -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    params = load_params()
    hpo = params["hpo"]

    parser = argparse.ArgumentParser(description="Optuna HPO with nested MLflow runs.")
    parser.add_argument("--n-trials", type=int, default=int(hpo["n_trials"]))
    parser.add_argument("--tracking-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()

    metric = hpo.get("metric", "eval_f1")
    base = config_from_params(params)
    setup_mlflow(args.tracking_uri, base.experiment_name)

    sampler = optuna.samplers.TPESampler(seed=int(hpo.get("seed", 42)))
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="bert-tiny-hpo")
    parent_name = f"hpo-{datetime.now():%Y%m%d-%H%M}"

    with mlflow.start_run(run_name=parent_name, log_system_metrics=True) as parent:
        mlflow.set_tags({"run_type": "hpo_parent", "git_commit": git_commit(), "seed": str(base.seed)})
        mlflow.log_params({
            "n_trials": args.n_trials,
            "objective": metric,
            "sampler": "TPE",
            "search_space": json.dumps(hpo["search_space"]),
            **{f"base_{k}": v for k, v in base.hyperparams().items()},
        })

        def objective(trial: optuna.Trial) -> float:
            overrides = suggest(trial, hpo["search_space"])
            config = replace(base, run_name=f"{parent_name}-trial-{trial.number}", **overrides)
            run_id, metrics = train_once(
                config,
                nested=True,
                save_local=False,
                extra_tags={"run_type": "hpo_trial", "trial_number": str(trial.number)},
            )
            trial.set_user_attr("run_id", run_id)
            value = metrics[metric]
            mlflow.log_metric(f"trial_{metric}", value, step=trial.number)  # progress curve on the parent
            return value

        study.optimize(objective, n_trials=args.n_trials)

        best = study.best_trial
        best_run_id = best.user_attrs["run_id"]
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric(f"best_{metric}", best.value)
        mlflow.set_tags({"best_run_id": best_run_id, "best_trial": str(best.number)})

        trials_table = [
            {"trial": t.number, metric: t.value, "run_id": t.user_attrs.get("run_id"), **t.params}
            for t in study.trials
        ]
        mlflow.log_table(data={k: [row.get(k) for row in trials_table] for k in trials_table[0]},
                         artifact_file="hpo/trials.json")

    summary = {
        "parent_run_id": parent.info.run_id,
        "best_run_id": best_run_id,
        "best_trial": best.number,
        f"best_{metric}": best.value,
        "best_params": best.params,
    }
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_MODEL_DIR / "hpo_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
