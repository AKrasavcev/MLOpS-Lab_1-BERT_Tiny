from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw_dataset.csv"
DEFAULT_DATASET_NAME = "glue"
DEFAULT_DATASET_CONFIG = "sst2"



def build_raw_rows(dataset: dict[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for split_name in ("train", "validation"):
        for item in dataset.get(split_name, []):
            text = item.get("sentence") or item.get("text") or ""
            label = item.get("label")
            rows.append({"text": str(text), "label": int(label)})

    return rows


def write_csv_rows(path: Path | str, rows: Iterable[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["text", "label"]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def create_raw_dataset() -> Path:
    dataset = load_dataset(DEFAULT_DATASET_NAME, DEFAULT_DATASET_CONFIG)
    rows = build_raw_rows(dataset)
    return write_csv_rows(RAW_DATA_PATH, rows)


def main() -> None:
    output_path = create_raw_dataset()
    print(f"Saved raw dataset with {sum(1 for _ in output_path.open('r', encoding='utf-8')) - 1} rows to {output_path}")


if __name__ == "__main__":
    main()
