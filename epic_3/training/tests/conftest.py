from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def model_library_root() -> Path:
    return REPO_ROOT / "model_library"


def _write_csv(path: Path, X, y, feature_names, target_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*feature_names, target_name])
        for features, target in zip(X, y, strict=True):
            writer.writerow([*features, target])


@pytest.fixture
def iris_csv_splits(tmp_path: Path) -> tuple[Path, Path]:
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=42,
        stratify=dataset.target,
    )
    train_path = tmp_path / "iris_train.csv"
    test_path = tmp_path / "iris_test.csv"
    _write_csv(train_path, X_train, y_train, dataset.feature_names, "species")
    _write_csv(test_path, X_test, y_test, dataset.feature_names, "species")
    return train_path, test_path


@pytest.fixture
def regression_csv_splits(tmp_path: Path) -> tuple[Path, Path]:
    dataset = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=42,
    )
    train_path = tmp_path / "reg_train.csv"
    test_path = tmp_path / "reg_test.csv"
    _write_csv(train_path, X_train, y_train, dataset.feature_names, "target")
    _write_csv(test_path, X_test, y_test, dataset.feature_names, "target")
    return train_path, test_path
