import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

os.environ["SKIP_MODEL_LOAD"] = "1"  # import the app without a registry

from src import server  # noqa: E402


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[101, 102]]), "attention_mask": torch.tensor([[1, 1]])}


class FakeModel:
    config = SimpleNamespace(id2label={0: "negative", 1: "positive"})

    def __call__(self, **encoded):
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture
def loaded(monkeypatch):
    monkeypatch.setattr(server, "tokenizer", FakeTokenizer())
    monkeypatch.setattr(server, "model", FakeModel())
    monkeypatch.setattr(
        server,
        "MODEL_INFO",
        server.ModelInfo(
            model_uri="models:/bert-tiny-sst2@champion",
            resolved_uri="models:/bert-tiny-sst2/7",
            model_name="bert-tiny-sst2",
            model_version="7",
            alias="champion",
            run_id="run-7",
        ),
    )


def test_health_ok(client, loaded):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "7"
    assert body["run_id"] == "run-7"


def test_predict(client, loaded):
    resp = client.post("/predict", json={"text": "great movie"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "positive"
    assert body["confidence"] == pytest.approx(0.8808, abs=1e-4)
    assert set(body["probabilities"]) == {"negative", "positive"}
    assert body["model_version"] == "7"


def test_predict_validates_payload(client, loaded):
    assert client.post("/predict", json={}).status_code == 422


def test_health_503_without_model(client, monkeypatch):
    monkeypatch.setattr(server, "tokenizer", None)
    monkeypatch.setattr(server, "model", None)
    monkeypatch.setattr(server, "STARTUP_ERROR", "MlflowException: alias not found")
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
    assert "alias not found" in resp.json()["message"]


def test_predict_503_without_model(client, monkeypatch):
    monkeypatch.setattr(server, "tokenizer", None)
    monkeypatch.setattr(server, "model", None)
    assert client.post("/predict", json={"text": "x"}).status_code == 503


def test_resolve_alias_to_version():
    mlflow_client = MagicMock()
    mlflow_client.get_model_version_by_alias.return_value = SimpleNamespace(version=12, run_id="r12")
    info = server.resolve_model_uri("models:/bert-tiny-sst2@champion", mlflow_client)
    assert info.resolved_uri == "models:/bert-tiny-sst2/12"
    assert (info.model_version, info.alias, info.run_id) == ("12", "champion", "r12")


def test_resolve_latest_picks_highest_version():
    mlflow_client = MagicMock()
    mlflow_client.search_model_versions.return_value = [
        SimpleNamespace(version="2", run_id="a"),
        SimpleNamespace(version="10", run_id="b"),
    ]
    info = server.resolve_model_uri("models:/m/latest", mlflow_client)
    assert info.model_version == "10"


def test_resolve_non_registry_uri_passes_through():
    info = server.resolve_model_uri("runs:/abc/model")
    assert info.resolved_uri == "runs:/abc/model"
    assert info.model_version is None
