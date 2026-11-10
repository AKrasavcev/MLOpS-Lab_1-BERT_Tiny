# BERT-Tiny SST-2: a mini MLOps platform

An end-to-end MLOps project built around a small text classifier. It fine-tunes [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) on the GLUE **SST-2** sentiment task (positive / negative movie reviews) on CPU and covers the whole model life cycle:

- **Data and pipeline**: dataset download, a train/test split and a reproducible pipeline with **DVC** (`dvc repro`).
- **Experiment tracking**: parameters, step-wise metrics, the model, the git commit and the data hash are logged to **MLflow**.
- **Hyperparameter search**: **Optuna** HPO, with each trial logged as a nested MLflow run.
- **Reproducibility**: any historical run can be re-trained from its run id (same code, data and params) and compared.
- **Infrastructure**: a local **Kubernetes** cluster (Minikube) running MLflow backed by **Postgres**, with resource quotas and Secrets.
- **Training in the cluster**: a Kubernetes **Job** that downloads data, trains, evaluates and logs to the in-cluster MLflow.
- **Model registry**: the `bert-tiny-sst2` registered model with `@champion` / `@previous` aliases.
- **Serving**: a **FastAPI** inference API (`/health`, `/predict`) that pulls `@champion` from the registry at startup. The image contains no model.
- **CI/CD**: GitHub Actions for lint, tests and image builds to GHCR, then cluster training, a **quality gate**, registration, deployment, a smoke test and **automatic rollback** on a self-hosted runner.
- **Ops**: load testing (latency/throughput NFRs), Trivy vulnerability scans, an SBOM and DB password rotation.

Requirements: [docs/requirements.md](docs/requirements.md). Latest load-test results: [docs/load-test.md](docs/load-test.md).

---

## Contents

1. [Architecture](#architecture)
2. [Repository layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Quick start](#quick-start)
5. [Local development](#local-development)
6. [Kubernetes platform](#kubernetes-platform)
7. [Model registry](#model-registry-champion--previous)
8. [Inference API](#inference-api)
9. [Load test](#load-test)
10. [CI](#ci-lint-tests-images)
11. [CD](#cd-quality-gate-deploy-auto-rollback)
12. [Rollback](#rollback)
13. [Docker images, GHCR and security scans](#docker-images-ghcr-and-security-scans)
14. [Configuration reference](#configuration-reference)
15. [Troubleshooting](#troubleshooting)
16. [Cleanup](#cleanup)

---

## Architecture

```
                 GitHub                                   Local machine (WSL / Linux)
 ┌──────────────────────────────────┐        ┌────────────────────────────────────────────────────────┐
 │ push / PR                        │        │  Minikube cluster, namespace mlops-l1                  │
 │   │                              │        │                                                        │
 │   ▼                              │        │   ┌────────────┐     ┌─────────────────────────────┐   │
 │ CI (GitHub-hosted runner)        │        │   │ postgres   │◄────┤ mlflow (tracking + registry)│   │
 │   ruff + pytest                  │        │   │ (PVC)      │     │ :5000, NodePort 30500 (PVC) │   │
 │   build + push images ──► GHCR ──┼──pull──┼─┐ └────────────┘     └───────▲─────────▲───────────┘   │
 │   │ (master only)                │        │ │                           │ log     │ download      │
 │   ▼                              │        │ │  ┌──────────────────────┐ │         │ @champion     │
 │ CD (self-hosted runner) ─────────┼────────┼─┼─►│ Job bert-tiny-train  ├─┘         │               │
 │   train Job → quality gate       │        │ │  └──────────────────────┘ ┌─────────┴────────────┐  │
 │   → register @champion           │        │ └──────────────────────────►│ Deployment           │  │
 │   → deploy → smoke test          │        │                             │ mlops-l1-bert-tiny   │  │
 │   → auto-rollback on failure     │        │                             │ FastAPI :8000        │  │
 └──────────────────────────────────┘        │                             └──────────────────────┘  │
                                             └────────────────────────────────────────────────────────┘
```

**Data flow:** Hugging Face `glue/sst2` → `data/raw_dataset.csv` (DVC-tracked) → `data/train.csv` + `data/test.csv` → `train.py` (MLflow run + model) → `evaluate_model.py` (`model/metrics.json`) → `register.py` (registry version + `@champion`) → API.

---

## Repository layout

| Path | What it is |
|---|---|
| [src/create_raw_dataset.py](src/create_raw_dataset.py) | Downloads SST-2 from Hugging Face → `data/raw_dataset.csv` |
| [src/prepare_dataset.py](src/prepare_dataset.py) | Samples `dataset_size` rows and splits them into `data/train.csv` / `data/test.csv` |
| [src/train.py](src/train.py) | Fine-tunes BERT-Tiny, logs params/metrics/model to MLflow, writes `model/run_id.txt` |
| [src/evaluate_model.py](src/evaluate_model.py) | Evaluates on the test set, logs to the same run, writes `model/metrics.json` |
| [src/hpo.py](src/hpo.py) | Optuna HPO (nested MLflow runs), writes `model/hpo_best.json` |
| [src/reproduce.py](src/reproduce.py) | Re-trains a historical run (git checkout + `dvc checkout` + same params) and compares metrics |
| [src/register.py](src/register.py) | Registers a run as a model version, moves `@champion` / `@previous`, and does rollback |
| [src/server.py](src/server.py) | FastAPI inference service |
| [dvc.yaml](dvc.yaml), [params.yaml](params.yaml) | DVC pipeline (`prepare → train → evaluate`) and all hyperparameters |
| [docker/](docker/) | `Dockerfile.cpu` (API image), `Dockerfile.train` + `train-entrypoint.sh` (training Job image) |
| [k8s/](k8s/) | Namespace, ResourceQuota, ConfigMaps, Postgres, MLflow, training Job, API Deployment, Secret template |
| [scripts/](scripts/) | Deploy, train Job, smoke test, rollback, load test, runner setup, GHCR push, password rotation |
| [tests/](tests/) | Unit tests (no MLflow server or cluster needed) |
| [.github/workflows/](.github/workflows/) | `ci.yml`, `cd.yml`, `rollback.yml` |
| [docs/](docs/) | Requirements, U1 report, load-test results, Trivy report, SBOM |

Requirement files:

| File | Used for |
|---|---|
| `requirements.txt` | Local development: training, DVC, MLflow, Optuna, tests |
| `requirements-server.txt` | API image and CI (CPU torch, FastAPI, `mlflow-skinny`) |
| `requirements-train.txt` | Training Job image (pinned, CPU torch) |
| `requirements-dev.txt` | Lint and tests in CI, on top of `requirements-server.txt` |

`torch`, `transformers` and `numpy` are pinned to the same versions in the server and train images, so a model trained in the Job loads in the API.

---

## Prerequisites

| Tool | Needed for | Install |
|---|---|---|
| Python **3.12** | everything | `sudo apt install python3.12 python3.12-venv` |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | virtualenv and packages | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| git | code, lineage tags | `sudo apt install git` |
| [Docker](https://docs.docker.com/engine/install/ubuntu/) | building images, Minikube driver | Docker Engine (or Docker Desktop with WSL integration) |
| [Minikube](https://minikube.sigs.k8s.io/docs/start/) | local Kubernetes cluster | see link |
| kubectl | cluster access | [install](https://kubernetes.io/docs/tasks/tools/), or `minikube kubectl --` |
| openssl, curl | Secrets, scripts | `sudo apt install openssl curl` |
| [Trivy](https://trivy.dev/docs/latest/getting-started/installation/) *(optional)* | image scan / SBOM | see link |
| [GitHub CLI](https://cli.github.com/) *(optional)* | manual push to GHCR | `sudo apt install gh` |

Hardware: about 4 CPUs and 8 GB RAM free for Minikube. No GPU is needed.

The project is developed on **WSL2 (Ubuntu)** and plain Linux works the same way. See [Troubleshooting](#troubleshooting) for WSL specifics.

---

## Quick start

From a fresh clone to a deployed model API in the cluster:

```bash
git clone https://github.com/AKrasavcev/MLOpS-Lab_1-BERT_Tiny.git
cd MLOpS-Lab_1-BERT_Tiny

# 1. Python env
bash scripts/create_venv.sh            # uv venv .venv + requirements.txt
source .venv/bin/activate

# 2. Cluster + MLflow/Postgres
minikube start --cpus=4 --memory=8192
cp k8s/secret.example.yaml k8s/secret.yaml
sed -i "s/<change-me>/$(openssl rand -hex 24)/" k8s/secret.yaml   # git-ignored, never commit it
bash scripts/k8_deploy.sh
kubectl -n mlops-l1 rollout status deploy/mlflow

# 3. Train in the cluster (builds the image, runs the Job, saves the run id to model/job/)
bash scripts/train_job.sh

# 4. Register the model as @champion
kubectl -n mlops-l1 port-forward svc/mlflow 5000:5000 &
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/register.py --run-id "$(cat model/job/run_id.txt)"

# 5. Deploy the API and test it
kubectl apply -f k8s/deployment-cpu.yaml
kubectl -n mlops-l1 rollout status deploy/mlops-l1-bert-tiny
bash scripts/smoke_test.sh
```

MLflow UI: <http://127.0.0.1:5000> (while the port-forward runs).

---

## Local development

### 1. Install dependencies

```bash
bash scripts/create_venv.sh
source .venv/bin/activate
```

This is the same as `uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt`.

### 2. Start a local MLflow server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5000
```

UI: <http://127.0.0.1:5000>. In another terminal:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

You can point `MLFLOW_TRACKING_URI` at the in-cluster MLflow instead (see [Access MLflow](#access-mlflow)). If it is not set, runs go to a local `./mlruns` file store.

### 3. Configure DVC

`.dvc/config` has a local remote `localremote`. Point it at a folder on your machine:

```bash
mkdir -p ~/dvcstore
dvc remote modify localremote url ~/dvcstore     # or: dvc remote add -d localremote ~/dvcstore
```

The remote is local, so a fresh clone has no data to `dvc pull`. Create the raw dataset once:

```bash
python src/create_raw_dataset.py       # downloads glue/sst2 -> data/raw_dataset.csv
dvc add data/raw_dataset.csv           # only if you want to track a new version
dvc push
```

### 4. Run the pipeline

With DVC, which runs only the stages whose inputs changed:

```bash
dvc repro            # prepare -> train -> evaluate
```

Or step by step:

```bash
python src/prepare_dataset.py
python src/train.py              # CLI flags override params.yaml, e.g. --learning-rate 5e-5 --epochs 2
python src/evaluate_model.py
```

Outputs:

- An MLflow run in the `bert-tiny-sst2` experiment with params, train loss every `logging_steps`, eval metrics every `eval_steps`, final `eval_accuracy`, `eval_f1`, `eval_precision`, `eval_recall` and `eval_loss`, the model under the `model/` artifact path, and evaluation JSON under `evaluation/`.
- Tags `git_commit`, `dvc_raw_md5` and `seed` for lineage.
- `model/run_id.txt` and `model/metrics.json` (committed). `model/model/` is the DVC-cached model.

Hyperparameters live in [params.yaml](params.yaml). To compare against the previous commit:

```bash
dvc metrics diff HEAD~1
dvc params diff HEAD~1
```

### 5. Hyperparameter search (Optuna)

```bash
python src/hpo.py                 # n_trials and search space from params.yaml (hpo:)
python src/hpo.py --n-trials 10
```

This creates one parent run (`run_type=hpo_parent`) with a nested run per trial. The best trial is tagged `best_run_id` on the parent and written to `model/hpo_best.json`. Register it with `python src/register.py --from-hpo`.

### 6. Reproduce a historical run

```bash
python src/reproduce.py <run_id>                 # default metric tolerance 0.005
python src/reproduce.py <run_id> --tolerance 0.01
```

The script checks out the run's `git_commit`, restores the matching data with `dvc checkout`, applies the run's train params and runs `dvc repro --force`. It tags the new run `reproduces=<run_id>`, compares the eval metrics, and then force-checks-out your original branch. It exits 0 on `REPRODUCED` and 1 on `MISMATCH`.

**Before you start:** commit or stash all changes, because the final force checkout discards uncommitted work. The data must be in the local DVC cache (otherwise run `dvc pull`), and `MLFLOW_TRACKING_URI` must point at the server that has the run.

### 7. Lint and tests

```bash
ruff check .
pytest -q
```

---

## Kubernetes platform

All resources live in the `mlops-l1` namespace.

| Manifest | Creates |
|---|---|
| [namespace.yaml](k8s/namespace.yaml) | namespace `mlops-l1` |
| [resourcequota.yaml](k8s/resourcequota.yaml) | quota: requests 4 CPU / 8Gi, limits 8 CPU / 16Gi |
| [configmap.yaml](k8s/configmap.yaml) | `mlops-l1-config` with `MODEL_URI=models:/bert-tiny-sst2@champion` |
| [postgres.yaml](k8s/postgres.yaml) | Postgres 16 with a PVC: the MLflow backend store |
| [mlflow.yaml](k8s/mlflow.yaml) | MLflow 3.15.2 server with an artifact PVC, Service `mlflow` (:5000, NodePort 30500) |
| [train-configmap.yaml](k8s/train-configmap.yaml) | `mlops-l1-train-params`, mounted into the Job as `/app/params.yaml` |
| [train-job.yaml](k8s/train-job.yaml) | Job `bert-tiny-train` |
| [deployment-cpu.yaml](k8s/deployment-cpu.yaml) | API Deployment + Service `mlops-l1-bert-tiny` (:8000) |
| [secret.example.yaml](k8s/secret.example.yaml) | template for the Secrets `mlflow-db` and `mlflow-credentials` |

### Start the cluster and deploy MLflow

```bash
minikube start --cpus=4 --memory=8192

# Secrets file (once per clone). k8s/secret.yaml is git-ignored.
cp k8s/secret.example.yaml k8s/secret.yaml
sed -i "s/<change-me>/$(openssl rand -hex 24)/" k8s/secret.yaml

bash scripts/k8_deploy.sh
kubectl -n mlops-l1 get pods            # postgres and mlflow Running
```

`k8_deploy.sh` applies the namespace, quota, ConfigMap and Secrets. It creates the `mlflow-db` and `mlflow-credentials` Secrets if they are missing, with a random hex DB password, then applies Postgres and MLflow.

<details>
<summary>Manual equivalent</summary>

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
</details>

Credentials exist only in the cluster Secrets and are never committed. To rotate the DB password on a running database:

```bash
bash scripts/rotate_db_password.sh
```

### Access MLflow

```bash
kubectl -n mlops-l1 port-forward svc/mlflow 5000:5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Alternatively, use the NodePort: `minikube service -n mlops-l1 mlflow --url`.

Any local command then logs to the cluster, for example `python src/train.py`, `dvc repro` or `python src/hpo.py`.

### Train inside the cluster (Kubernetes Job)

The Job runs the training image ([docker/Dockerfile.train](docker/Dockerfile.train)) on CPU: wait for MLflow → download SST-2 → prepare → train → evaluate (or HPO). Everything is logged to the in-cluster MLflow.

```bash
bash scripts/train_job.sh                 # build image, minikube image load, run Job, follow logs
bash scripts/train_job.sh hpo             # Optuna HPO instead of a single run
SKIP_BUILD=1 bash scripts/train_job.sh    # reuse the image already in Minikube
kubectl -n mlops-l1 get job bert-tiny-train   # COMPLETIONS 1/1
```

- **Hyperparameters** come from the ConfigMap in [k8s/train-configmap.yaml](k8s/train-configmap.yaml), in the same format as `params.yaml`. To change one (for example `learning_rate`), edit the file and re-run with `SKIP_BUILD=1`. The script re-applies the ConfigMap and re-creates the Job, because a Job's pod template can't be changed in place.
- **Connection**: the Secret `mlflow-credentials` (`MLFLOW_TRACKING_URI`).
- **Lineage**: each run is tagged with `git_commit` (baked in at build time) and `dvc_raw_md5` (md5 of the downloaded CSV, comparable with `data/raw_dataset.csv.dvc`).
- **Results**: in train mode the run id and test metrics are saved to `model/job/run_id.txt` and `model/job/metrics.json` (set `OUT_DIR` to change the location).

---

## Model registry (`@champion` / `@previous`)

Training only logs a model. [src/register.py](src/register.py) creates a new version of the registered model `bert-tiny-sst2`, moves the old `@champion` to `@previous` and puts `@champion` on the new version. It also copies the run's metrics and `git_commit` onto the version as tags.

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/register.py                          # run id from model/run_id.txt (after dvc repro)
python src/register.py --run-id "$(cat model/job/run_id.txt)"   # run from the cluster Job
python src/register.py --from-hpo               # best HPO trial (model/hpo_best.json)
python src/register.py --run-id <id> --no-promote   # register only, don't move aliases
python src/register.py --rollback [--to-version 12] # move @champion back, no new version
```

---

## Inference API

[src/server.py](src/server.py) reads `MODEL_URI` (default `models:/bert-tiny-sst2@champion`). At startup it resolves the alias to a concrete version, downloads exactly that version from MLflow and reports it in `/health`. Supported URI forms: `models:/<name>@<alias>`, `models:/<name>/<version>`, `models:/<name>/latest`, `runs:/…` or a local path.

| Endpoint | Description |
|---|---|
| `GET /` | service info, model URI and version |
| `GET /health` | `status` (`ok` / `degraded`), `model_name`, `model_version`, `run_id`, … |
| `POST /predict` | body `{"text": "..."}` → `prediction`, `confidence`, `probabilities`, `model_version` |

If the model can't be loaded, `/health` reports `degraded` and `/predict` returns **503**.

### In the cluster

```bash
kubectl apply -f k8s/configmap.yaml -f k8s/deployment-cpu.yaml
kubectl -n mlops-l1 port-forward svc/mlops-l1-bert-tiny 8000:8000

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This movie is amazing and really enjoyable."}'
```

The Deployment takes `MODEL_URI` from the ConfigMap `mlops-l1-config` and `MLFLOW_TRACKING_URI` from the Secret `mlflow-credentials`. The image tag in `deployment-cpu.yaml` is updated by CD. After moving `@champion` by hand, restart the pods to load the new version:

```bash
kubectl -n mlops-l1 rollout restart deployment/mlops-l1-bert-tiny
bash scripts/smoke_test.sh        # or: bash scripts/smoke_test.sh <expected_version>
```

### Locally (without Kubernetes)

With Docker, using an MLflow server on the host:

```bash
docker build -f docker/Dockerfile.cpu -t bert-tiny:latest .
docker run --rm --network host \
  -e MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  bert-tiny:latest
```

Or directly from the venv:

```bash
uv pip install fastapi "uvicorn[standard]"
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uvicorn src.server:app --port 8000
```

---

## Load test

[scripts/load_test.sh](scripts/load_test.sh) restarts the Deployment to time the cold start and reads the image and model sizes. It then port-forwards the Service and runs [scripts/load_test.py](scripts/load_test.py), which sends 500 `POST /predict` requests at concurrency 10 by default. Results go to [docs/load-test.md](docs/load-test.md) and `docs/load-test.json`.

```bash
bash scripts/load_test.sh            # 500 requests, concurrency 10
bash scripts/load_test.sh 1000 20    # custom requests / concurrency

# load test only, against an already port-forwarded service:
python scripts/load_test.py --url http://localhost:8000 -n 500 -c 10
```

The report covers p50/p95/p99 latency, throughput, error rate, model version, image tag and size, model artifact size, cold start time and pod peak memory. The script exits 1 if more than 1% of requests fail.

---

## CI (lint, tests, images)

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on every push and pull request on a GitHub-hosted runner:

1. **lint-test**: installs `requirements-server.txt` + `requirements-dev.txt`, runs `ruff check .` (config in [pyproject.toml](pyproject.toml)) and `pytest`.
2. **build-push** (after lint-test passes): builds `docker/Dockerfile.cpu` and `docker/Dockerfile.train` and pushes `ghcr.io/akrasavcev/bert-tiny:sha-<12>` and `ghcr.io/akrasavcev/bert-tiny-train:sha-<12>` with `GITHUB_TOKEN`. On `master` both are also tagged `latest`. Pull requests only build, without pushing.

The tests in [tests/](tests/) need no MLflow server or cluster:

| File | Covers |
|---|---|
| `test_prepare.py` | `split_rows` sizes, no overlap, seed determinism, CSV round-trip |
| `test_gate.py` | [scripts/quality_gate.py](scripts/quality_gate.py) passes/fails around the threshold, exit codes, step summary |
| `test_server.py` | FastAPI `TestClient` with a fake model: `/health`, `/predict`, 503 without a model, alias → version resolution |
| `test_register.py` | `@champion` → `@previous` alias moves with a fake `MlflowClient` |

To run the same checks as CI locally:

```bash
uv pip install -r requirements-dev.txt
ruff check .
pytest -q
```

---

## CD (quality gate, deploy, auto-rollback)

[.github/workflows/cd.yml](.github/workflows/cd.yml) runs on a **self-hosted runner** on the Minikube machine, because GitHub-hosted runners can't reach the local cluster. It starts after CI is green on `master`, or by hand from *Actions → CD → Run workflow*.

1. **train-gate**: `pytest`, then loads the CI-built `bert-tiny-train:sha-<12>` image into Minikube ([scripts/minikube_image.sh](scripts/minikube_image.sh), which builds the image locally if it is not in GHCR). It then runs the training Job with hyperparameters from `k8s/train-configmap.yaml`. [scripts/quality_gate.py](scripts/quality_gate.py) checks the test metrics against `gate:` in [params.yaml](params.yaml) (`eval_f1 >= 0.70`). Below the threshold the job fails and nothing else runs.
2. **deploy**: `register.py` registers the run and moves `@champion` (the old champion becomes `@previous`). The workflow writes `bert-tiny:sha-<12>` into `k8s/deployment-cpu.yaml`, then runs `kubectl apply`, `rollout restart` and `rollout status`. [scripts/smoke_test.sh](scripts/smoke_test.sh) then checks that `/health` reports the new version and that `/predict` answers. **If the deploy or the smoke test fails, the workflow rolls back by itself**: the alias returns to the old champion and `rollout undo` restores the old image. On success the new image tag is committed to `master` with `[skip ci]`.

### Self-hosted runner setup (once)

1. GitHub → repo *Settings → Actions → Runners → New self-hosted runner* (Linux x64). Copy the registration token.
2. On the Minikube machine:

   ```bash
   bash scripts/setup_runner.sh <TOKEN>   # installs ~/actions-runner, label "minikube", systemd service
   ```

   The script also links `kubectl` onto `PATH` if it's missing and, on WSL, writes `WSL_DISTRO_NAME` into the runner's `.env`.
3. Before each CD run: `minikube start`, and the runner service must be running:

   ```bash
   sudo ~/actions-runner/svc.sh status
   bash scripts/restart_action_runner.sh   # restart and follow logs until "Listening for Jobs"
   ```

The runner keeps one venv per requirements hash, so torch is downloaded once ([.github/actions/runner-env](.github/actions/runner-env/action.yml)).

### Demo: a failing run

- *Actions → CD → Run workflow* with `min_value = 0.95`: the gate fails and nothing is deployed.
- Or train a bad model: set `learning_rate: 1e-7` in `k8s/train-configmap.yaml` and push.

---

## Rollback

From a terminal (it uses `MLFLOW_TRACKING_URI` if set, otherwise it port-forwards MLflow itself):

```bash
bash scripts/rollback.sh                 # @champion -> @previous version, rollout restart, smoke test
bash scripts/rollback.sh 12              # @champion -> v12
bash scripts/rollback.sh --undo-image    # alias rollback + kubectl rollout undo (previous image)
kubectl -n mlops-l1 rollout undo deployment/mlops-l1-bert-tiny   # image only, aliases untouched
```

Or from GitHub with *Actions → Rollback → Run workflow* ([rollback.yml](.github/workflows/rollback.yml)), which takes an optional `version` and `undo_image`.

The rolled-back version becomes `@previous`, so running `rollback.sh` a second time returns to the newer model. `/health` shows which version is serving.

---

## Docker images, GHCR and security scans

| Image | Dockerfile | Contents |
|---|---|---|
| `bert-tiny` | [docker/Dockerfile.cpu](docker/Dockerfile.cpu) | Python 3.12-slim, multi-stage, CPU torch, FastAPI server. **No model**: it is pulled from MLflow at startup |
| `bert-tiny-train` | [docker/Dockerfile.train](docker/Dockerfile.train) | CPU torch + training deps, base checkpoint baked in, non-root user |

Published images: <https://github.com/users/AKrasavcev/packages/container/package/bert-tiny>. CI pushes them, or you can push by hand:

```bash
bash scripts/to_ghcr.sh      # gh auth login, docker login ghcr.io, build + push bert-tiny:sha-<12>
```

Minikube uses containerd and can't see the host's Docker images. Load them explicitly:

```bash
minikube image load bert-tiny:latest
bash scripts/minikube_image.sh <image> <dockerfile>   # pull from GHCR, or build, then load
```

Vulnerability scan and SBOM with Trivy:

```bash
docker build -f docker/Dockerfile.cpu -t bert-tiny:latest .
trivy image --scanners vuln --severity HIGH,CRITICAL -f json -o docs/trivy-cpu.json bert-tiny:latest
trivy image -f spdx-json -o docs/sbom-cpu.json bert-tiny:latest
```

---

## Configuration reference

### `params.yaml`

| Section | Keys | Used by |
|---|---|---|
| `train` | `max_length`, `epochs`, `train_batch_size`, `eval_batch_size`, `learning_rate`, `lr_scheduler_type`, `optimizer`, `weight_decay`, `seed`, `logging_steps`, `eval_steps` | `train.py` |
| `download` | `dataset_name` (`glue`), `dataset_config` (`sst2`) | `create_raw_dataset.py` |
| `prepare` | `dataset_size` (10000), `test_size` (0.2), `seed` | `prepare_dataset.py` |
| `hpo` | `n_trials`, `seed`, `metric`, `search_space` (a list is categorical, `{low, high, log}` is a float range) | `hpo.py` |
| `gate` | `metric` (`eval_f1`), `min_value` (0.70) | `quality_gate.py` (CD) |

Inside the cluster the Job uses the copy in `k8s/train-configmap.yaml`. Keep the two in sync if you want local and cluster runs to match.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | – (local `./mlruns`) | MLflow server for training, registration and the API |
| `MODEL_URI` | `models:/bert-tiny-sst2@champion` | model the API loads |
| `MODEL_CACHE_DIR` | `/tmp/model` | where the API downloads the model |
| `TRAIN_MODE` | `train` | training Job mode: `train` or `hpo` |
| `GIT_COMMIT` | from git, else `unknown` | lineage tag when there is no `.git` (images) |
| `SKIP_BUILD`, `IMAGE`, `OUT_DIR` | – | options for `scripts/train_job.sh` |
| `NS`, `DEPLOY`, `PY`, `MLFLOW_PORT`, `API_PORT` | `mlops-l1`, `mlops-l1-bert-tiny`, `.venv/bin/python`, `15000`, `18000` | options for the rollback and smoke-test scripts |

---

## Troubleshooting

- **`k8_deploy.sh` fails on `k8s/secret.yaml`**: the file is git-ignored. Create it from `k8s/secret.example.yaml` (see [Start the cluster](#start-the-cluster-and-deploy-mlflow)). Use a hex password, because it goes into a `postgresql://` URI.
- **MLflow `403 Invalid Host header`**: add the host you connect with to `MLFLOW_SERVER_ALLOWED_HOSTS` in `k8s/mlflow.yaml`.
- **API `/health` is `degraded`**: there is no `@champion` yet (register a model) or MLflow is unreachable. Check with `kubectl -n mlops-l1 logs deploy/mlops-l1-bert-tiny`.
- **`ErrImagePull` for a local image**: load it into Minikube (`minikube image load …`). Minikube doesn't see host Docker images.
- **WSL: runner reports `kubeconfig: Misconfigured`**: `WSL_DISTRO_NAME` must be in `~/actions-runner/.env`, which `setup_runner.sh` adds. Restart with `bash scripts/restart_action_runner.sh`.
- **Stale kubeconfig after a reboot**: `minikube start` or `minikube update-context`.
- **Pods stuck `Pending`**: the namespace ResourceQuota or Minikube memory is exhausted. Run `kubectl -n mlops-l1 describe quota` and give Minikube more `--memory`.
- **See everything that runs**: `kubectl get po -A`.

---

## Cleanup

```bash
kubectl delete namespace mlops-l1      # all project resources, including PVCs (MLflow data!)
minikube stop                          # keep the cluster for later
minikube delete                        # remove the cluster completely
```

## License

See [LICENSE](LICENSE).
