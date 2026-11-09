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

Configure **DVC**:

```bash
mkdir -p ~/dvcstore
dvc remote add -d localremote ~/dvcstore
```

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
kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml -f k8s/resourcequota.yaml
kubectl -n mlops-l1 create secret generic mlflow-db \
    --from-literal=POSTGRES_DB=mlflow \
    --from-literal=POSTGRES_USER=mlflow \
    --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)"
kubectl -n mlops-l1 create secret generic mlflow-credentials \
    --from-literal=MLFLOW_TRACKING_URI=http://mlflow.mlops-l1.svc.cluster.local:5000
kubectl apply -f k8s/postgres.yaml -f k8s/mlflow.yaml
```

Database credentials live only in the `mlflow-db` Secret (see [k8s/secret.example.yaml](k8s/secret.example.yaml)); they are never committed. To change the password on an existing database, run `bash scripts/rotate_db_password.sh`.

The API Deployment downloads the model from the MLflow registry at startup (`MODEL_URI` in [k8s/configmap.yaml](k8s/configmap.yaml)); the image contains no model.


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

### Train inside the cluster (Kubernetes Job)

The Job ([k8s/train-job.yaml](k8s/train-job.yaml)) runs the training image ([docker/Dockerfile.train](docker/Dockerfile.train)) on CPU. It downloads SST-2 from Hugging Face, prepares it, trains, evaluates, and logs everything to the in-cluster MLflow.

- **Hyperparameters**: ConfigMap `mlops-l1-train-params` ([k8s/train-configmap.yaml](k8s/train-configmap.yaml)), mounted as `/app/params.yaml`. It has the same format as `params.yaml`.
- **Connection details**: Secret `mlflow-credentials` (key `MLFLOW_TRACKING_URI`). The Secret is created in the cluster by `scripts/k8_deploy.sh` and is never committed (template: [k8s/secret.example.yaml](k8s/secret.example.yaml)).
- **Lineage**: the run is tagged with `git_commit` (baked in at build time) and `dvc_raw_md5` (md5 of the downloaded raw CSV, comparable with `data/raw_dataset.csv.dvc`).

```bash
bash scripts/train_job.sh          # build image, minikube image load, run Job, follow logs, wait for Completed
bash scripts/train_job.sh hpo      # same, but runs Optuna HPO (nested runs)
kubectl -n mlops-l1 get job bert-tiny-train   # COMPLETIONS 1/1, STATUS Complete
```

To change a hyperparameter, edit `k8s/train-configmap.yaml` (for example `learning_rate`) and re-run with `SKIP_BUILD=1 bash scripts/train_job.sh`. The script applies the ConfigMap and re-creates the Job, because a Job's pod template can't be changed in place. A new run then appears in the MLflow UI under the `bert-tiny-sst2` experiment.

### Register a model and make it `@champion`

Training only logs the model. [src/register.py](src/register.py) registers a run as a new version of `bert-tiny-sst2`, moves the current `@champion` to `@previous` and puts `@champion` on the new version. It also copies the run's metrics and `git_commit` onto the version as tags.

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/register.py                    # run id from model/run_id.txt (after dvc repro)
python src/register.py --from-hpo         # best HPO trial from model/hpo_best.json
python src/register.py --run-id <run_id>  # any run
python src/register.py --run-id <run_id> --no-promote   # register only
```

The API loads `MODEL_URI=models:/bert-tiny-sst2@champion`. At startup it resolves the alias to a version (`MlflowClient.get_model_version_by_alias`), downloads exactly that version, and reports it in `/health` (`model_version`, `run_id`). Restart the pods to pick up a new champion:

```bash
kubectl -n mlops-l1 rollout restart deployment/mlops-l1-bert-tiny
```

To deploy trained model in pod:

```bash
kubectl apply -f k8s/configmap.yaml -f k8s/deployment-cpu.yaml

kubectl -n mlops-l1 port-forward svc/mlops-l1-bert-tiny 8000:8000

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This movie is amazing and really enjoyable."}'
```

The Deployment reads `MODEL_URI` from the ConfigMap `mlops-l1-config` and `MLFLOW_TRACKING_URI` from the Secret `mlflow-credentials` (created by `scripts/k8_deploy.sh` or by applying secret.yaml).

### Load test (inference NFRs)

[scripts/load_test.sh](scripts/load_test.sh) restarts the Deployment to time the cold start, reads the image and model sizes, then port-forwards the Service and runs [scripts/load_test.py](scripts/load_test.py): 500 `POST /predict` requests at concurrency 10 by default. It writes [docs/load-test.md](docs/load-test.md) and `docs/load-test.json`.

```bash
bash scripts/load_test.sh            # 500 requests, concurrency 10
bash scripts/load_test.sh 1000 20    # custom requests / concurrency

# or only the load test, against an already port-forwarded service:
python scripts/load_test.py --url http://localhost:8000 -n 500 -c 10
```

The output includes p50/p95/p99 latency, throughput, error rate, model version, image tag and size, model artifact size, cold start time and pod peak memory. The script exits 1 if more than 1% of requests fail.

To delete minikube cluster:
```bash
minikube delete
```

## Outputs

- MLflow run parameters and metrics.
- A logged model artifact under the `model/` artifact path.
- Evaluation JSON under the `evaluation/` artifact path.
