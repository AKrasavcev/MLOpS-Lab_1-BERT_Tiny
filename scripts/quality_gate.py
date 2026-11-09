"""Quality gate: fail (exit 1) when the trained model is below the threshold.

  python scripts/quality_gate.py                      # model/metrics.json vs params.yaml -> gate
  python scripts/quality_gate.py --min-value 0.95     # override the threshold (CD red-run demo)

The threshold lives in params.yaml under `gate: {metric, min_value}`. Without that section the
defaults below are used. In GitHub Actions the verdict is also written to $GITHUB_STEP_SUMMARY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT_DIR / "model" / "metrics.json"
PARAMS_PATH = ROOT_DIR / "params.yaml"
DEFAULT_METRIC = "eval_f1"
DEFAULT_MIN_VALUE = 0.70


@dataclass
class GateResult:
    metric: str
    value: float | None
    min_value: float

    @property
    def passed(self) -> bool:
        return self.value is not None and self.value >= self.min_value

    @property
    def message(self) -> str:
        if self.value is None:
            return f"FAIL: metric '{self.metric}' not found in metrics"
        verdict = "PASS" if self.passed else "FAIL"
        sign = ">=" if self.passed else "<"
        return f"{verdict}: {self.metric} = {self.value:.4f} {sign} {self.min_value:.4f}"


def load_gate_config(params_path: Path = PARAMS_PATH) -> tuple[str, float]:
    params = yaml.safe_load(params_path.read_text(encoding="utf-8")) if params_path.exists() else None
    gate = (params or {}).get("gate") or {}
    return str(gate.get("metric", DEFAULT_METRIC)), float(gate.get("min_value", DEFAULT_MIN_VALUE))


def check(metrics: dict[str, float], metric: str, min_value: float) -> GateResult:
    value = metrics.get(metric)
    return GateResult(metric=metric, value=None if value is None else float(value), min_value=float(min_value))


def write_step_summary(result: GateResult, summary_path: str | None) -> None:
    if not summary_path:
        return
    status = "Passed" if result.passed else "Failed"
    value = "missing" if result.value is None else f"{result.value:.4f}"
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write(
            f"### ({status}) Quality gate\n\n"
            "| metric | value | threshold | result |\n|---|---|---|---|\n"
            f"| `{result.metric}` | {value} | {result.min_value:.4f} | {'pass' if result.passed else 'fail'} |\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail when the model metric is below the threshold.")
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--params", type=Path, default=PARAMS_PATH)
    parser.add_argument("--metric", type=str, default=None, help="override gate.metric")
    parser.add_argument("--min-value", type=float, default=None, help="override gate.min_value")
    args = parser.parse_args(argv)

    metric, min_value = load_gate_config(args.params)
    metric = args.metric or metric
    min_value = min_value if args.min_value is None else args.min_value

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = check(metrics, metric, min_value)
    print(result.message)
    write_step_summary(result, os.getenv("GITHUB_STEP_SUMMARY"))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
