from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
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


DEFAULT_MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_DATASET_NAME = "glue"
DEFAULT_DATASET_CONFIG = "sst2"
DEFAULT_EXPERIMENT_NAME = "bert-tiny-sst2"


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_config: str = DEFAULT_DATASET_CONFIG
    train_split: str = "train[:1000]"
    eval_split: str = "validation[:200]"
    max_length: int = 128
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    seed: int = 42
    output_dir: str = "artifacts/bert-tiny"
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    run_name: str | None = None
    tracking_uri: str | None = None


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Fine-tune prajjwal1/bert-tiny and log to MLflow.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-split", default="train[:1000]")
    parser.add_argument("--eval-split", default="validation[:200]")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/bert-tiny")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()
    return TrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
    )


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
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return compute_metrics


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split={"train": config.train_split, "eval": config.eval_split},
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    def tokenize_batch(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=config.max_length)

    tokenized_train = dataset["train"].map(tokenize_batch, batched=True)
    tokenized_eval = dataset["eval"].map(tokenize_batch, batched=True)

    label_list = dataset["train"].features["label"].names

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
        num_labels=len(label_list),
        id2label={index: label for index, label in enumerate(label_list)},
        label2id={label: index for index, label in enumerate(label_list)},
    )

    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
        seed=config.seed,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,
        disable_tqdm=False,
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

    run_name = config.run_name or f"{config.model_name}-sst2"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(asdict(config))
        mlflow.log_params(
            {
                "train_size": len(tokenized_train),
                "eval_size": len(tokenized_eval),
                "num_labels": len(label_list),
            }
        )

        train_result = trainer.train()
        eval_result = trainer.evaluate()

        train_metrics = {key: float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float, np.floating))}
        eval_metrics = {key: float(value) for key, value in eval_result.items() if isinstance(value, (int, float, np.floating))}
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(eval_metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(model_dir)
            tokenizer.save_pretrained(model_dir)
            mlflow.log_artifacts(str(model_dir), artifact_path="model")

            predictions = trainer.predict(tokenized_eval)
            predicted_labels = np.argmax(predictions.predictions, axis=-1)
            report = {
                "accuracy": accuracy_score(predictions.label_ids, predicted_labels),
                "f1": f1_score(predictions.label_ids, predicted_labels),
                "metrics": {"train": train_metrics, "eval": eval_metrics},
            }
            report_path = Path(temp_dir) / "evaluation.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")

    print(json.dumps({"run_name": run_name, "experiment": config.experiment_name, "tracking_uri": config.tracking_uri or "default"}, indent=2))


if __name__ == "__main__":
    main()
