# BERT-Tiny Training with MLflow

This repository contains a CPU-friendly fine-tuning script for `prajjwal1/bert-tiny` on the GLUE SST-2 sentiment task.

## What it does

- Downloads `prajjwal1/bert-tiny` from Hugging Face.
- Fine-tunes it on a small SST-2 subset.
- Logs parameters, metrics, and the trained model to MLflow.
- Uses `MLFLOW_TRACKING_URI` so it can log to a remote MLflow server, including the one deployed in Kubernetes.

## Install dependencies

```bash
install uv
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run locally on local MLflow server

Run **MLflow** local server
```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--artifacts-destination ./mlartifacts \
--host 127.0.0.1 --port 5000
```

Server accessible at [http://127.0.0.1:5000](http://127.0.0.1:5000)

Run training manualy
```bash
python src/create_raw_dataset.py    # mandatory for first time

export MLFLOW_TRACKING_URI=http://localhost:5000
python src/prepare_dataset.py
python src/train.py
python src/evaluate_model.py
```

or using **DVC**
```bash
dvc repro
```

Run parameters can be changed in `params.yaml`

To see metrics and parameters chenge:
```bash
dvc metrics diff HEAD~1
dvc params diff HEAD~1
```

To reproduce experiment **(don't forget to save all uncommitted changes)**:
```bash
python reproduce.py <experiment-id>
```

## Create Docker image

If you want to build image yourself and run security scans:

Install **Docker**: [https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/)

Build image

```bash
docker build -f docker/Dockerfile.cpu -t bert-tiny:latest .

docker run --rm -d --name bert-tiny-api -p 8000:8000 bert-tiny:latest
```

Install **Trivy**: [https://trivy.dev/docs/latest/getting-started/installation/](https://trivy.dev/docs/latest/getting-started/installation/)

Trivy security scan
```bash
trivy image --scanners vuln --severity HIGH,CRITICAL -f json -o docs/trivy-cpu.json bert-tiny:latest

trivy image -f spdx-json -o docs/sbom-cpu.json bert-tiny:latest
```

## Publish to GHCR

Built image accessible at [https://github.com/users/AKrasavcev/packages/container/package/bert-tiny](https://github.com/users/AKrasavcev/packages/container/package/bert-tiny)

## Run against the Kubernetes MLflow service

Install **Minikube**: [https://minikube.sigs.k8s.io/docs/start/](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fwindows%2Fx86-64%2Fstable%2F.exe+download)

Start Minikube cluster
```bash
minikube start --cpus=4 --memory=8192
bash scripts/k8_deploy.sh
```

or 

```bash
kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml \
    -f k8s/mlflow.yaml -f k8s/postgres.yaml \
    -f k8s/resourcequota.yaml
```


Access MLflow server through port-forwarding from your PC:

```bash
kubectl -n mlops-l1 port-forward svc/mlflow 5000:5000

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/train.py
python src/evaluate_model.py
```

You can also run the full pipeline with the same tracking URI:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
dvc repro
```

To see all running pods:
```bash
kubectl get po -A
```

To deploy trained model in pod:

```bash
kubectl apply -f k8s/deployment-cpu.yaml

kubectl -n mlops-l1 port-forward svc/mlops-l1-bert-tiny 8000:8000

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This movie is amazing and really enjoyable."}'
```

To delete minikube cluster:
```bash
minikube delete
```

## Outputs

- MLflow run parameters and metrics.
- A logged model artifact under the `model/` artifact path.
- Evaluation JSON under the `evaluation/` artifact path.
