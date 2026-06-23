from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.agents.training.data_loader import load_training_data
from backend.agents.training.errors import TrainingDataError


def test_loads_csv_and_infers_species_target(iris_csv_splits: tuple[Path, Path]) -> None:
    train_path, test_path = iris_csv_splits
    data = load_training_data(train_path, test_path)

    assert data.target_name == "species"
    assert data.X_train.shape[1] == 4
    assert data.X_test.shape[1] == 4
    assert data.y_train.ndim == 1


def test_loads_npz_with_split_specific_keys(tmp_path: Path) -> None:
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    np.savez(train, X_train=np.ones((4, 3)), y_train=np.array([0, 1, 0, 1]))
    np.savez(test, X_test=np.ones((2, 3)), y_test=np.array([0, 1]))

    data = load_training_data(train, test)
    assert data.X_train.shape == (4, 3)
    assert data.X_test.shape == (2, 3)


def test_rejects_missing_split(tmp_path: Path) -> None:
    with pytest.raises(TrainingDataError, match="does not exist"):
        load_training_data(tmp_path / "missing.csv", tmp_path / "also_missing.csv")
