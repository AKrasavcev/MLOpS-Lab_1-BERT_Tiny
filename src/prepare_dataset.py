from __future__ import annotations

import csv
import random
import yaml
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw_dataset.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"


def split_rows(rows: list[dict[str, object]], test_size: float = 0.2, seed: int = 42) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rows:
        return [], []

    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)
    cutoff = max(1, int(len(shuffled_rows) * (1 - test_size)))
    return shuffled_rows[:cutoff], shuffled_rows[cutoff:]


def write_csv_rows(path: Path | str, rows: Iterable[dict[str, object]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["text", "label"]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized_row = {"text": row["text"], "label": int(row["label"])}
            writer.writerow(normalized_row)
    return output_path



def prepare_dataset(params: dict[str, float]) -> tuple[Path, Path]:
    raw_file = Path(RAW_DATA_PATH)
    with raw_file.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [{"text": row["text"], "label": int(row["label"])} for row in reader]

    train_rows, test_rows = split_rows(rows, test_size=params["test_size"], seed=int(params["seed"]))
    train_output = write_csv_rows(TRAIN_DATA_PATH, train_rows)
    test_output = write_csv_rows(TEST_DATA_PATH, test_rows)
    return train_output, test_output


def main() -> None:
    params = yaml.safe_load(Path(ROOT_DIR / "params.yaml").read_text())["prepare"]

    train_path, test_path = prepare_dataset(params)
    print(f"Prepared train set: {train_path}")
    print(f"Prepared test set: {test_path}")


if __name__ == "__main__":
    main()
