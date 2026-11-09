"""Load test for the inference API (R4).

Sends N POST /predict requests at a fixed concurrency and reports latency percentiles,
throughput and error rate. Results go to JSON (machine-readable) and Markdown (for docs/).

    kubectl -n mlops-l1 port-forward svc/mlops-l1-bert-tiny 8000:8000 &
    python scripts/load_test.py --url http://localhost:8000 -n 500 -c 10

Extra measurements taken outside this script (image size, cold start, ...) can be added
to the report with --meta key=value. scripts/load_test.sh does all of it in one go.
"""
import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Mix of short and long, positive and negative sentences, so latency is not measured on one input.
TEXTS = [
    "a gorgeous, witty, seductive movie.",
    "the plot is nothing but boilerplate clichés from start to finish.",
    "it's a charming and often affecting journey.",
    "unflinchingly bleak and desperate",
    "an utterly incompetent, conceptually bankrupt mess of a film.",
    "the performances are uniformly excellent, and the script never talks down to its audience, "
    "which makes the two hours fly by.",
    "boring.",
    "a masterpiece four years in the making",
    "the movie is so thoroughly predictable that by the halfway point you will have stopped caring "
    "about any of the characters, and the ending only confirms that the filmmakers stopped caring too.",
    "funny, touching and smart",
]


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default) on an already sorted list."""
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def one_request(client: httpx.Client, text: str) -> tuple[float, int | str]:
    start = time.perf_counter()
    try:
        status: int | str = client.post("/predict", json={"text": text}).status_code
    except httpx.HTTPError as exc:
        status = type(exc).__name__
    return (time.perf_counter() - start) * 1000, status


def run(url: str, n: int, concurrency: int, warmup: int, timeout: float, seed: int) -> dict:
    rng = random.Random(seed)
    texts = [rng.choice(TEXTS) for _ in range(n)]
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    with httpx.Client(base_url=url, timeout=timeout, limits=limits) as client:
        health = client.get("/health")
        health.raise_for_status()  # do not benchmark a pod that has no model
        health = health.json()

        for i in range(warmup):  # first calls pay for lazy init; keep them out of the numbers
            one_request(client, TEXTS[i % len(TEXTS)])

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda t: one_request(client, t), texts))
        wall_s = time.perf_counter() - started

    latencies = sorted(ms for ms, status in results if status == 200)
    errors: dict[str, int] = {}
    for _, status in results:
        if status != 200:
            errors[str(status)] = errors.get(str(status), 0) + 1

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": url,
        "requests": n,
        "concurrency": concurrency,
        "warmup": warmup,
        "model_name": health.get("model_name"),
        "model_version": health.get("model_version"),
        "run_id": health.get("run_id"),
        "successful": len(latencies),
        "error_rate": round(sum(errors.values()) / n, 4),
        "errors": errors,
        "wall_time_s": round(wall_s, 2),
        "throughput_rps": round(n / wall_s, 1),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "min": round(latencies[0], 1) if latencies else None,
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
            "max": round(latencies[-1], 1) if latencies else None,
        },
    }


def to_markdown(r: dict) -> str:
    lat = r["latency_ms"]
    rows = [
        ("Date (UTC)", r["timestamp"]),
        ("Model", f'{r["model_name"]} v{r["model_version"]} (run `{r["run_id"]}`)'),
        *[(k.replace("_", " ").capitalize(), v) for k, v in r.get("meta", {}).items()],
        ("Requests / concurrency", f'{r["requests"]} / {r["concurrency"]} (+{r["warmup"]} warm-up, not counted)'),
        ("Latency p50 / p95 / p99", f'{lat["p50"]} / **{lat["p95"]}** / {lat["p99"]} ms'),
        ("Latency mean / min / max", f'{lat["mean"]} / {lat["min"]} / {lat["max"]} ms'),
        ("Throughput", f'{r["throughput_rps"]} req/s ({r["wall_time_s"]} s total)'),
        ("Error rate", f'{r["error_rate"]:.2%} {r["errors"] or ""}'.strip()),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (
        "# Inference load test\n\n"
        "Generated by `scripts/load_test.sh` (`scripts/load_test.py`): `POST /predict` through "
        "`kubectl port-forward` to the `mlops-l1-bert-tiny` Service, one pod (limit 1 CPU / 1Gi). "
        "Latency is measured on the client, so it includes the port-forward hop.\n\n"
        f"| Metric | Value |\n|---|---|\n{body}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("-n", "--requests", type=int, default=500)
    ap.add_argument("-c", "--concurrency", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                    help="extra measurement to include in the report (repeatable)")
    ap.add_argument("--json-out", type=Path, help="write results as JSON")
    ap.add_argument("--md-out", type=Path, help="write results as a Markdown table")
    ap.add_argument("--max-error-rate", type=float, default=0.01,
                    help="exit 1 if the error rate is above this (default 0.01)")
    args = ap.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        ap.error("--requests and --concurrency must be >= 1")

    result = run(args.url, args.requests, args.concurrency, args.warmup, args.timeout, args.seed)
    result["meta"] = dict(m.split("=", 1) for m in args.meta)

    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(to_markdown(result))

    if result["error_rate"] > args.max_error_rate:
        print(f"error rate {result['error_rate']:.2%} > {args.max_error_rate:.2%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
