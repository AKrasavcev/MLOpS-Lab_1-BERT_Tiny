"""Fine-tune BERT-Tiny on the prepared SST-2 CSVs and track everything in MLflow.

  python src/train.py [--learning-rate 3e-5 ...]      # one run, saves model/ for DVC
  python src/hpo.py                                   # many nested runs via train_once()

Tracking target comes from MLFLOW_TRACKING_URI (or --tracking-uri).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"
DEFAULT_MODEL_DIR = ROOT_DIR / "model"
DEFAULT_MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_EXPERIMENT_NAME = "bert-tiny-sst2"
DEFAULT_TRAIN_DATA_PATH = ROOT_DIR / "data" / "train.csv"
DEFAULT_TEST_DATA_PATH = ROOT_DIR / "data" / "test.csv"
RAW_DVC_POINTER = ROOT_DIR / "data" / "raw_dataset.csv.dvc"
MODEL_INPUT_COLUMNS = {"input_ids", "token_type_ids", "attention_mask", "label"}
SYSTEM_METRICS_INTERVAL_S = 5


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = "glue"
    dataset_config: str = "sst2"
    train_data_path: str = str(DEFAULT_TRAIN_DATA_PATH)
    test_data_path: str = str(DEFAULT_TEST_DATA_PATH)
    max_length: int = 128
    epochs: float = 1.0
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "linear"
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    seed: int = 42
    logging_steps: int = 20
    eval_steps: int = 100
    model_dir: str = str(DEFAULT_MODEL_DIR)
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    run_name: str | None = None

    def hyperparams(self) -> dict[str, object]:
        """What gets logged as MLflow params (names match params.yaml -> train)."""
        skip = {"train_data_path", "test_data_path", "model_dir", "experiment_name", "run_name"}
        return {k: v for k, v in asdict(self).items() if k not in skip}


def load_params() -> dict:
    return yaml.safe_load(PARAMS_PATH.read_text())


def config_from_params(params: dict) -> TrainingConfig:
    """Build a config from the params.yaml dict (casts, because YAML reads 2e-5 as a string)."""
    train = params["train"]
    download = params.get("download", {})
    casts = {
        "max_length": int, "epochs": float, "train_batch_size": int, "eval_batch_size": int,
        "learning_rate": float, "lr_scheduler_type": str, "optimizer": str,
        "weight_decay": float, "seed": int, "logging_steps": int, "eval_steps": int,
    }
    values = {key: cast(train[key]) for key, cast in casts.items() if key in train}
    if "dataset_name" in download:
        values["dataset_name"] = download["dataset_name"]
    if "dataset_config" in download:
        values["dataset_config"] = download["dataset_config"]
    return TrainingConfig(**values)


def parse_args() -> tuple[TrainingConfig, str | None]:
    base = config_from_params(load_params())
    parser = argparse.ArgumentParser(description="Fine-tune BERT-Tiny on CSV datasets and log to MLflow.")
    parser.add_argument("--max-length", type=int, default=base.max_length)
    parser.add_argument("--epochs", type=float, default=base.epochs)
    parser.add_argument("--train-batch-size", type=int, default=base.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=base.eval_batch_size)
    parser.add_argument("--learning-rate", type=float, default=base.learning_rate)
    parser.add_argument("--lr-scheduler-type", type=str, default=base.lr_scheduler_type)
    parser.add_argument("--optimizer", type=str, default=base.optimizer)
    parser.add_argument("--weight-decay", type=float, default=base.weight_decay)
    parser.add_argument("--seed", type=int, default=base.seed)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()

    config = replace(
        base,
        max_length=args.max_length,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        seed=args.seed,
        run_name=args.run_name,
    )
    return config, args.tracking_uri


def setup_mlflow(tracking_uri: str | None, experiment_name: str) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.set_system_metrics_sampling_interval(SYSTEM_METRICS_INTERVAL_S)


def git_commit() -> str:
    """Commit from git, or from the GIT_COMMIT env var (k8s Job images have no .git)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return os.getenv("GIT_COMMIT", "unknown")


def dvc_md5(dvc_file: Path) -> str:
    """md5 of the raw dataset as recorded by `dvc add` - the join key to dvc.lock."""
    try:
        return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]
    except Exception:
        return os.getenv("DVC_RAW_MD5", "untracked")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_pip_requirements() -> list[str]:
    """Pinned env for the logged model, so the server can match it exactly."""
    return [
        f"mlflow-skinny=={mlflow.__version__}",
        f"torch=={torch.__version__.split('+')[0]}",  # drop +cu130 / +cpu local tag
        f"transformers=={transformers.__version__}",
        f"numpy=={np.__version__}",
    ]


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


class MlflowStepCallback(TrainerCallback):
    """Send every Trainer log (train loss, lr, grad_norm, eval_*) to the active run with its step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        numeric = {
            ("train_step_loss" if key == "loss" else key): float(value)
            for key, value in logs.items()
            if isinstance(value, (int, float, np.floating)) and key != "epoch"
        }
        if "epoch" in logs:
            numeric["epoch"] = float(logs["epoch"])
        if numeric:
            mlflow.log_metrics(numeric, step=state.global_step)


def _tokenize(dataset, tokenizer, max_length: int):
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True
    )
    tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c not in MODEL_INPUT_COLUMNS])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


def train_once(
    config: TrainingConfig,
    nested: bool = False,
    save_local: bool = True,
    extra_tags: dict[str, str] | None = None,
) -> tuple[str, dict[str, float]]:
    """Train + evaluate one model inside its own MLflow run. Returns (run_id, final eval metrics).

    Expects setup_mlflow() to have been called. With nested=True the run becomes a child of
    the currently active run (used by hpo.py).
    """
    set_seed(config.seed)

    train_df = pd.read_csv(config.train_data_path)
    test_df = pd.read_csv(config.test_data_path)
    train_dataset = load_dataset("csv", data_files=config.train_data_path, split="train")
    eval_dataset = load_dataset("csv", data_files=config.test_data_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenized_train = _tokenize(train_dataset, tokenizer, config.max_length)
    tokenized_eval = _tokenize(eval_dataset, tokenizer, config.max_length)

    label_values = sorted(int(v) for v in train_df["label"].unique())
    label_to_id = {label: idx for idx, label in enumerate(label_values)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_values),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    model_dir = Path(config.model_dir)
    training_arguments = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optimizer,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        weight_decay=config.weight_decay,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        report_to=[],  # our callback does the MLflow logging
        seed=config.seed,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[MlflowStepCallback()],
    )

    run_name = config.run_name or (
        f"bt-{config.lr_scheduler_type}-lr{config.learning_rate:g}-bs{config.train_batch_size}"
    )
    dataset_id = f"{config.dataset_name}-{config.dataset_config}"

    with mlflow.start_run(run_name=run_name, nested=nested, log_system_metrics=True) as run:
        run_id = run.info.run_id
        mlflow.log_params(config.hyperparams())
        mlflow.log_params({
            "train_size": len(tokenized_train),
            "eval_size": len(tokenized_eval),
            "num_labels": len(label_values),
        })
        mlflow.set_tags({
            "git_commit": git_commit(),
            "dvc_raw_md5": dvc_md5(RAW_DVC_POINTER),
            "model_name": config.model_name,
            "seed": str(config.seed),
            **(extra_tags or {}),
        })
        mlflow.log_input(
            mlflow.data.from_pandas(train_df, source="data/train.csv", name=f"{dataset_id}-train"),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(test_df, source="data/test.csv", name=f"{dataset_id}-test"),
            context="evaluation",
        )

        train_result = trainer.train()
        final_eval = trainer.evaluate()  # also logged by the callback at the last step
        numeric = (int, float, np.floating)
        eval_metrics = {k: float(v) for k, v in final_eval.items() if isinstance(v, numeric)}
        train_metrics = {k: float(v) for k, v in train_result.metrics.items() if isinstance(v, numeric)}

        # Logged only - registration (and @champion) happens later in the gate/register step.
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            name="model",
            task="text-classification",
            pip_requirements=model_pip_requirements(),
        )

        report = {"run_id": run_id, "eval": eval_metrics, "train": train_metrics}
        mlflow.log_dict(report, "evaluation/evaluation.json")

        if save_local:
            local_model_dir = model_dir / "model"
            local_model_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(local_model_dir))
            tokenizer.save_pretrained(str(local_model_dir))
            (model_dir / "run_id.txt").write_text(run_id, encoding="utf-8")
            (model_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return run_id, eval_metrics


def main() -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    config, tracking_uri = parse_args()
    setup_mlflow(tracking_uri, config.experiment_name)
    run_id, metrics = train_once(config)
    print(json.dumps({"run_id": run_id, "eval_f1": metrics.get("eval_f1"), "model_dir": config.model_dir}, indent=2))


if __name__ == "__main__":
    main()
