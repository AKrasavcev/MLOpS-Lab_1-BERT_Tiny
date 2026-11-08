"""BERT-Tiny sentiment API. The model comes from the MLflow registry, not from the image.

MODEL_URI (default models:/bert-tiny-sst2@champion) is resolved once at startup:
  1. the alias is turned into a concrete version (MlflowClient.get_model_version_by_alias),
  2. exactly that version is downloaded (models:/<name>/<version>),
  3. /health reports model_name, model_version and run_id.
Resolving first means /health always describes what was actually loaded, even if the alias
moves later. To pick up a new @champion: `kubectl rollout restart deployment/mlops-l1-bert-tiny`.
"""
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_URI = os.getenv("MODEL_URI", "models:/bert-tiny-sst2@champion")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/model"))

# models:/<name>@<alias> | models:/<name>/<version> | models:/<name>/latest
_REGISTRY_URI = re.compile(r"^models:/(?P<name>[^/@]+)(?:@(?P<alias>[^/]+)|/(?P<version>[^/]+))$")

app = FastAPI(title="BERT Tiny Sentiment API")


@dataclass
class ModelInfo:
    model_uri: str                    # what was asked for (env)
    resolved_uri: str                 # what was downloaded
    model_name: str | None = None
    model_version: str | None = None
    alias: str | None = None
    run_id: str | None = None


def resolve_model_uri(model_uri: str, client: MlflowClient | None = None) -> ModelInfo:
    """Pin a registry URI to one concrete version. Non-registry URIs (runs:/, paths) pass through."""
    match = _REGISTRY_URI.match(model_uri)
    if not match:
        return ModelInfo(model_uri=model_uri, resolved_uri=model_uri)

    client = client or MlflowClient()
    name, alias, version = match["name"], match["alias"], match["version"]
    if alias:
        mv = client.get_model_version_by_alias(name, alias)
    elif version.lower() == "latest":
        mv = max(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
    else:
        mv = client.get_model_version(name, version)

    return ModelInfo(
        model_uri=model_uri,
        resolved_uri=f"models:/{name}/{mv.version}",
        model_name=name,
        model_version=str(mv.version),
        alias=alias,
        run_id=mv.run_id,
    )


def load_model(info: ModelInfo, dst: Path):
    """Download a model logged with mlflow.transformers and load it with plain transformers."""
    dst.mkdir(parents=True, exist_ok=True)
    local_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=info.resolved_uri, dst_path=str(dst)))
    tok = AutoTokenizer.from_pretrained(str(local_dir / "components" / "tokenizer"))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(local_dir / "model"))
    mdl.eval()
    return tok, mdl


tokenizer = None
model = None
MODEL_INFO = ModelInfo(model_uri=MODEL_URI, resolved_uri=MODEL_URI)
STARTUP_ERROR = ""
if os.getenv("SKIP_MODEL_LOAD") != "1":  # tests import the app without a registry
    try:
        MODEL_INFO = resolve_model_uri(MODEL_URI)
        tokenizer, model = load_model(MODEL_INFO, MODEL_CACHE_DIR)
    except Exception as exc:  # keep the process up so /health can say what went wrong
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"model load failed for {MODEL_URI}: {STARTUP_ERROR}", flush=True)


def model_ready() -> bool:
    return tokenizer is not None and model is not None


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    ready = model_ready()
    body = {
        "status": "ok" if ready else "degraded",
        **asdict(MODEL_INFO),
        "message": "Model loaded" if ready else (STARTUP_ERROR or "Model not loaded"),
    }
    # 503 keeps the pod out of the Service (readiness) until a model is loaded.
    return JSONResponse(body, status_code=200 if ready else 503)


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    if not model_ready():
        raise HTTPException(status_code=503, detail=f"Model not available: {STARTUP_ERROR or 'not loaded'}")

    encoded = tokenizer(
        payload.text,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_index = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0][predicted_index].item())

    label = model.config.id2label.get(predicted_index, str(predicted_index))
    probabilities = {
        str(key): round(float(value), 4)
        for key, value in zip(model.config.id2label.values(), probs[0].tolist())
    }

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "model_version": MODEL_INFO.model_version,
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "BERT Tiny Sentiment API",
        "endpoints": ["/health", "/predict"],
        "model_uri": MODEL_URI,
        "model_version": MODEL_INFO.model_version,
    }
