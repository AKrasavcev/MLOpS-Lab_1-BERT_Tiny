# BERT-Tiny Training with MLflow

This repository contains a CPU-friendly fine-tuning script for `prajjwal1/bert-tiny` on the GLUE SST-2 sentiment task.

## What it does

- Downloads `prajjwal1/bert-tiny` from Hugging Face.
- Fine-tunes it on a small SST-2 subset.
- Logs parameters, metrics, and the trained model to MLflow.
- Uses `MLFLOW_TRACKING_URI` so it can log to a remote MLflow server, including the one deployed in Kubernetes.

## Install

```bash
install uv
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run locally

Run mlflow local server
```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--artifacts-destination ./mlartifacts \
--host 127.0.0.1 --port 5000
```

Run manualy
```bash
python src/create_raw_dataset.py    # mandatory for first time

export MLFLOW_TRACKING_URI=http://localhost:5000
python src/prepare_dataset.py
python src/train.py
python src/evaluate_model.py
```

or using 
```bash
dvc repro
```

## Run against the Kubernetes MLflow service

If the MLflow server is exposed through port-forwarding:

```bash
kubectl -n mlops port-forward svc/mlflow 5000:5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m src.train_bert_tiny
```

## Outputs

- MLflow run parameters and metrics.
- A logged model artifact under the `model/` artifact path.
- Evaluation JSON under the `evaluation/` artifact path.
