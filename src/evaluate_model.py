from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT_DIR / "model"
DEFAULT_MODEL_DIR = MODEL_ROOT / "model"
DEFAULT_TEST_DATA_PATH = ROOT_DIR / "data" / "test.csv"
DEFAULT_RUN_ID_PATH = MODEL_ROOT / "run_id.txt"
DEFAULT_METRICS_PATH = MODEL_ROOT / "metrics.json"


def build_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
        }

    return compute_metrics


def evaluate_model() -> dict[str, float]:
    model_path = Path(DEFAULT_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    test_dataset = load_dataset("csv", data_files=str(DEFAULT_TEST_DATA_PATH), split="train")

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized_test = test_dataset.map(tokenize_batch, batched=True)
    tokenized_test = tokenized_test.remove_columns(
        [column for column in tokenized_test.column_names if column not in {"input_ids", "token_type_ids", "attention_mask", "label"}]
    )
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_test.set_format("torch")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_path.parent / "eval"),
            per_device_eval_batch_size=8,
            report_to=[],
            disable_tqdm=True,
        ),
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(),
    )

    eval_result = trainer.evaluate()
    metrics = {key: float(value) for key, value in eval_result.items() if isinstance(value, (int, float, np.floating))}

    metrics_output = Path(DEFAULT_METRICS_PATH)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_id = Path(DEFAULT_RUN_ID_PATH).read_text(encoding="utf-8").strip() if Path(DEFAULT_RUN_ID_PATH).exists() else None
    if run_id:                                   # lineage: evaluation lands on the training run
        try:
            client = mlflow.MlflowClient()
            for key, value in metrics.items():
                client.log_metric(run_id, f"eval/{key}", value)
        except Exception as exc:                               # best-effort: run may live on an unreachable server
            print(f"warning: eval metrics not attached to the training run ({exc})")

    return metrics


def main() -> None:
    metrics = evaluate_model()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
