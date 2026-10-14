from __future__ import annotations

import argparse
import json
import yaml
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import subprocess
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT_DIR / "model"
DEFAULT_MODEL_DIR = MODEL_ROOT
DEFAULT_MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_DATASET_NAME = "glue"
DEFAULT_DATASET_CONFIG = "sst2"
DEFAULT_EXPERIMENT_NAME = "bert-tiny-sst2"
DEFAULT_TRAIN_DATA_PATH = ROOT_DIR / "data" / "train.csv"
DEFAULT_TEST_DATA_PATH = ROOT_DIR / "data" / "test.csv"


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_config: str = DEFAULT_DATASET_CONFIG
    train_data_path: str = str(DEFAULT_TRAIN_DATA_PATH)
    test_data_path: str = str(DEFAULT_TEST_DATA_PATH)
    max_length: int = 128
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    weight_decay: float = 0.01
    seed: int = 42
    model_dir: str = str(DEFAULT_MODEL_DIR)
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    run_name: str | None = None
    tracking_uri: str | None = None


def parse_args() -> TrainingConfig:
    params = yaml.safe_load(Path(ROOT_DIR / "params.yaml").read_text())["train"]

    parser = argparse.ArgumentParser(description="Fine-tune a BERT Tiny classifier on CSV datasets and log to MLflow.")
    parser.add_argument("--max-length", type=int, default=params["max_length"])
    parser.add_argument("--epochs", type=int, default=params["epochs"])
    parser.add_argument("--learning-rate", type=float, default=params["learning_rate"])
    parser.add_argument("--train-batch-size", type=int, default=params["train_batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=params["eval_batch_size"])
    parser.add_argument("--weight-decay", type=float, default=params["weight_decay"])
    parser.add_argument("--seed", type=int, default=params["seed"])
    args = parser.parse_args()

    return TrainingConfig(
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def dvc_md5(dvc_file: Path) -> str:
    """md5 of the raw dataset as recorded by `dvc add` - the join key to dvc.lock."""
    try:
        return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]
    except Exception:
        return "untracked"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return compute_metrics


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = config.tracking_uri or "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    train = pd.read_csv(config.train_data_path)
    test = pd.read_csv(config.test_data_path)

    train_dataset = load_dataset("csv", data_files=config.train_data_path, split="train")
    eval_dataset = load_dataset("csv", data_files=config.test_data_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

    tokenized_train = train_dataset.map(tokenize_batch, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_batch, batched=True)

    label_values = sorted({int(row["label"]) for row in train_dataset})
    label_to_id = {label: idx for idx, label in enumerate(label_values)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenized_train = tokenized_train.remove_columns(
        [column for column in tokenized_train.column_names if column not in {"input_ids", "token_type_ids", "attention_mask", "label"}]
    )
    tokenized_eval = tokenized_eval.remove_columns(
        [column for column in tokenized_eval.column_names if column not in {"input_ids", "token_type_ids", "attention_mask", "label"}]
    )

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_eval = tokenized_eval.rename_column("label", "labels")
    tokenized_train.set_format("torch")
    tokenized_eval.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_values),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    training_arguments = TrainingArguments(
        output_dir=str(Path(config.model_dir) / "checkpoints"),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
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
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
    )

    run_name = config.run_name or f"{config.model_name}-csv"
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(asdict(config))
        mlflow.log_params(
            {
                "train_size": len(tokenized_train),
                "eval_size": len(tokenized_eval),
                "num_labels": len(label_values),
                "dataset_path": config.train_data_path,
            }
        )
        mlflow.set_tags({                                           # identities
            "git_commit": git_commit(),
            "dvc_raw_md5": dvc_md5(ROOT_DIR / "data/raw_dataset.csv.dvc"),
            "model_name": config.model_name,
        })
        mlflow.log_input(                                           # data provenance (name + digest)
            mlflow.data.from_pandas(
                train, source="data/train.csv", name=f"{config.dataset_name}-{config.dataset_config}-train"
            ),
            context="training",
        )

        train_result = trainer.train()
        # eval_result = trainer.evaluate()

        train_metrics = {key: float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float, np.floating))}
        # eval_metrics = {key: float(value) for key, value in eval_result.items() if isinstance(value, (int, float, np.floating))}
        mlflow.log_metrics(train_metrics)
        # mlflow.log_metrics(eval_metrics)

        local_model_dir = model_dir / "model"
        local_model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(local_model_dir))
        tokenizer.save_pretrained(str(local_model_dir))

        try:
            mlflow.log_artifacts(str(local_model_dir), artifact_path="model")
        except mlflow.exceptions.MlflowException:
            pass

        run_id = mlflow.active_run().info.run_id
        (model_dir / "run_id.txt").write_text(run_id, encoding="utf-8")

        predictions = trainer.predict(tokenized_eval)
        predicted_labels = np.argmax(predictions.predictions, axis=-1)
        report = {
            "accuracy": float(accuracy_score(predictions.label_ids, predicted_labels)),
            "f1": float(f1_score(predictions.label_ids, predicted_labels, zero_division=0)),
            # "metrics": {"train": train_metrics, "eval": eval_metrics},
            "metrics": {"train": train_metrics},
        }
        report_path = model_dir / "evaluation.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        try:
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        except mlflow.exceptions.MlflowException:
            pass

    print(json.dumps({"run_name": run_name, "run_id": run_id, "model_dir": str(local_model_dir)}, indent=2))


if __name__ == "__main__":
    main()
