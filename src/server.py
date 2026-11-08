import os
from pathlib import Path

import mlflow
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The model comes from the MLflow registry, not from the image.
MODEL_URI = os.getenv("MODEL_URI", "models:/bert-tiny-sst2@champion")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/model"))

app = FastAPI(title="BERT Tiny Sentiment API")


def load_from_registry(model_uri: str, dst: Path):
    """Download a model logged with mlflow.transformers and load it with transformers."""
    dst.mkdir(parents=True, exist_ok=True)
    local_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dst)))
    tok = AutoTokenizer.from_pretrained(str(local_dir / "components" / "tokenizer"))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(local_dir / "model"))
    mdl.eval()
    return tok, mdl


STARTUP_ERROR = ""
try:
    tokenizer, model = load_from_registry(MODEL_URI, MODEL_CACHE_DIR)
    MODEL_READY = True
except Exception as exc:  # pragma: no cover - only for startup failures in container
    tokenizer = None
    model = None
    MODEL_READY = False
    STARTUP_ERROR = str(exc)


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if MODEL_READY else "degraded",
        "model_uri": MODEL_URI,
        "message": "Model loaded" if MODEL_READY else STARTUP_ERROR,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    if not MODEL_READY or tokenizer is None or model is None:
        raise RuntimeError(f"Model not available: {STARTUP_ERROR}")

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
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "BERT Tiny Sentiment API",
        "endpoints": ["/health", "/predict"],
        "model_uri": MODEL_URI,
    }
