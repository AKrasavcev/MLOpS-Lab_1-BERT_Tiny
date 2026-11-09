import csv

from src.prepare_dataset import split_rows, write_csv_rows


def make_rows(n: int) -> list[dict[str, object]]:
    return [{"text": f"sentence {i}", "label": i % 2} for i in range(n)]


def test_split_sizes():
    train, test = split_rows(make_rows(100), dataset_size=50, test_size=0.2, seed=42)
    assert len(train) == 40
    assert len(test) == 10


def test_dataset_size_larger_than_data_keeps_all_rows():
    train, test = split_rows(make_rows(10), dataset_size=1000, test_size=0.2, seed=0)
    assert len(train) + len(test) == 10


def test_split_has_no_overlap():
    train, test = split_rows(make_rows(100), dataset_size=100, test_size=0.3, seed=1)
    assert {r["text"] for r in train}.isdisjoint({r["text"] for r in test})


def test_split_is_deterministic_for_seed():
    rows = make_rows(100)
    assert split_rows(rows, 100, 0.2, seed=7) == split_rows(rows, 100, 0.2, seed=7)
    assert split_rows(rows, 100, 0.2, seed=7) != split_rows(rows, 100, 0.2, seed=8)


def test_split_does_not_mutate_input():
    rows = make_rows(20)
    original = list(rows)
    split_rows(rows, 20, 0.2, seed=3)
    assert rows == original


def test_empty_input():
    assert split_rows([], dataset_size=10) == ([], [])


def test_csv_round_trip(tmp_path):
    rows = [{"text": 'he said "hi", then left', "label": 1}, {"text": "bad\nmovie", "label": "0"}]
    path = write_csv_rows(tmp_path / "nested" / "out.csv", rows)

    with path.open(newline="", encoding="utf-8") as fh:
        read_back = list(csv.DictReader(fh))

    assert [r["text"] for r in read_back] == [r["text"] for r in rows]
    assert [int(r["label"]) for r in read_back] == [1, 0]
