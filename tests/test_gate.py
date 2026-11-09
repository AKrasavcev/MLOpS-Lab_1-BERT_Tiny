import json

import pytest

from scripts.quality_gate import check, load_gate_config, main


@pytest.mark.parametrize(
    ("value", "passed"),
    [(0.69, False), (0.6999, False), (0.70, True), (0.75, True)],
)
def test_gate_around_threshold(value, passed):
    assert check({"eval_f1": value}, "eval_f1", 0.70).passed is passed


def test_missing_metric_fails():
    result = check({"eval_accuracy": 0.9}, "eval_f1", 0.70)
    assert not result.passed
    assert "not found" in result.message


def test_config_from_params(tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text("gate:\n  metric: eval_accuracy\n  min_value: 0.8\n")
    assert load_gate_config(params) == ("eval_accuracy", 0.8)


def test_config_defaults_without_gate_section(tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text("train:\n  epochs: 1\n")
    assert load_gate_config(params) == ("eval_f1", 0.70)


def test_main_exit_codes_and_summary(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"eval_f1": 0.745}))
    params = tmp_path / "params.yaml"
    params.write_text("gate:\n  metric: eval_f1\n  min_value: 0.70\n")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    args = ["--metrics", str(metrics), "--params", str(params)]
    assert main(args) == 0
    assert main([*args, "--min-value", "0.95"]) == 1
    text = summary.read_text()
    assert "pass" in text and "fail" in text
