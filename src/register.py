"""Register a trained run in the MLflow Model Registry and make it @champion.

  python src/register.py                     # run id from model/run_id.txt (dvc repro output)
  python src/register.py --run-id <id>       # any run that logged a "model" artifact
  python src/register.py --from-hpo          # best trial from model/hpo_best.json
  python src/register.py --run-id <id> --no-promote   # register only, keep aliases as they are
  python src/register.py --rollback                   # @champion -> the @previous version
  python src/register.py --rollback --to-version 12   # @champion -> v12

Alias logic: the version that is @champion now becomes @previous, and the new version becomes
@champion. The server loads models:/bert-tiny-sst2@champion, so moving the alias plus a
`kubectl rollout restart` is the whole deploy (and rollback is the same move in reverse).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "bert-tiny-sst2"
CHAMPION = "champion"
PREVIOUS = "previous"
RUN_ID_PATH = ROOT_DIR / "model" / "run_id.txt"
HPO_BEST_PATH = ROOT_DIR / "model" / "hpo_best.json"
# Run metrics copied onto the model version as tags, so the registry page shows the quality.
METRIC_TAGS = ("eval_f1", "eval_accuracy", "eval_precision", "eval_recall", "eval_loss")


def get_alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    """Version number behind an alias, or None if the alias (or the model) does not exist."""
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def promote(client: MlflowClient, name: str, version: str) -> dict[str, str | None]:
    """Make `version` the champion. The old champion (if different) becomes @previous."""
    old_champion = get_alias_version(client, name, CHAMPION)
    if old_champion is not None and old_champion != version:
        client.set_registered_model_alias(name, PREVIOUS, old_champion)
    client.set_registered_model_alias(name, CHAMPION, version)
    return {"champion": version, "previous": get_alias_version(client, name, PREVIOUS)}


def rollback(client: MlflowClient, name: str, to_version: str | None = None) -> dict[str, str | None]:
    """Move @champion back to `to_version` (default: @previous). The rolled-back version becomes @previous,
    so a second rollback undoes the first."""
    current = get_alias_version(client, name, CHAMPION)
    target = str(to_version) if to_version is not None else get_alias_version(client, name, PREVIOUS)
    if target is None:
        raise SystemExit(f"{name} has no @{PREVIOUS} alias; pass --to-version <n>.")
    if target == current:
        raise SystemExit(f"{name} v{target} is already @{CHAMPION}.")
    client.get_model_version(name, target)  # fails with a clear error if the version does not exist
    return {"model_name": name, "rolled_back_from": current, **promote(client, name, target)}


def version_tags(run: mlflow.entities.Run) -> dict[str, str]:
    tags = {
        "run_id": run.info.run_id,
        "git_commit": run.data.tags.get("git_commit", os.getenv("GIT_COMMIT", "unknown")),
        "dvc_raw_md5": run.data.tags.get("dvc_raw_md5", "unknown"),
        "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    for key in METRIC_TAGS:
        if key in run.data.metrics:
            tags[key] = f"{run.data.metrics[key]:.4f}"
    return tags


def register(
    run_id: str,
    name: str = MODEL_NAME,
    artifact_path: str = "model",
    promote_to_champion: bool = True,
    client: MlflowClient | None = None,
) -> dict[str, object]:
    """Register runs:/<run_id>/<artifact_path> as a new version of `name`, tag it, optionally promote."""
    client = client or MlflowClient()
    run = client.get_run(run_id)  # fails early with a clear error if the run id is wrong

    model_version = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", name)
    version = str(model_version.version)

    for key, value in version_tags(run).items():
        client.set_model_version_tag(name, version, key, value)
    f1 = run.data.metrics.get("eval_f1")
    client.update_model_version(
        name, version,
        description=f"run {run.info.run_name} ({run_id})" + (f", eval_f1={f1:.4f}" if f1 is not None else ""),
    )

    aliases = promote(client, name, version) if promote_to_champion else {
        "champion": get_alias_version(client, name, CHAMPION),
        "previous": get_alias_version(client, name, PREVIOUS),
    }
    return {"model_name": name, "version": version, "run_id": run_id, "eval_f1": f1, **aliases}


def resolve_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    if args.from_hpo:
        return json.loads(HPO_BEST_PATH.read_text(encoding="utf-8"))["best_run_id"]
    if RUN_ID_PATH.exists():
        return RUN_ID_PATH.read_text(encoding="utf-8").strip()
    raise SystemExit(f"No --run-id given and {RUN_ID_PATH} does not exist (run `dvc repro` first).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a run's model and move @champion to it.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-id", type=str, default=None)
    source.add_argument("--from-hpo", action="store_true", help=f"use best_run_id from {HPO_BEST_PATH.name}")
    source.add_argument("--rollback", action="store_true", help="move @champion back (no new version)")
    parser.add_argument("--to-version", type=str, default=None, help="with --rollback: target version")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--no-promote", action="store_true", help="register only, do not move aliases")
    parser.add_argument("--tracking-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--output", type=Path, default=None, help="also write the result JSON here (CD)")
    args = parser.parse_args()
    if args.to_version and not args.rollback:
        parser.error("--to-version needs --rollback")

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    if args.rollback:
        result = rollback(MlflowClient(), args.model_name, args.to_version)
    else:
        result = register(resolve_run_id(args), name=args.model_name, promote_to_champion=not args.no_promote)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
