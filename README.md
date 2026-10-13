# BERT-Tiny Training with MLflow

This repository contains a CPU-friendly fine-tuning script for `prajjwal1/bert-tiny` on the GLUE SST-2 sentiment task.

## What it does

- Downloads `prajjwal1/bert-tiny` from Hugging Face.
- Fine-tunes it on a small SST-2 subset.
- Logs parameters, metrics, and the trained model to MLflow.
- Uses `MLFLOW_TRACKING_URI` so it can log to a remote MLflow server, including the one deployed in Kubernetes.

## Install

```bash
pip install -r requirements.txt
```

## Run locally

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.train_bert_tiny \
  --experiment-name bert-tiny-sst2 \
  --run-name local-test \
  --train-split "train[:1000]" \
  --eval-split "validation[:200]"
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
