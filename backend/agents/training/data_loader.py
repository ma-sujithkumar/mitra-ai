"""Load already-split, already-preprocessed data for the local trainer.

Epic-2 owns preprocessing and splitting.  This loader therefore accepts two
small, explicit hand-off formats:

* CSV: header required; features must be numeric.  The target column can be
  supplied explicitly, otherwise a conventional target name is used and the
  final column is the deterministic fallback.
* NPZ: each file must contain either ``X``/``y`` or split-specific keys such as
  ``X_train``/``y_train`` and ``X_test``/``y_test``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import TrainingDataError

_TARGET_CANDIDATES = ("target", "label", "species", "class", "y", "output")


@dataclass(frozen=True)
class LoadedTrainingData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str


def load_training_data(
    train_path: str | Path,
    test_path: str | Path,
    *,
    target_column: str | None = None,
) -> LoadedTrainingData:
    """Load and validate train/test artifacts."""

    train = Path(train_path).expanduser().resolve()
    test = Path(test_path).expanduser().resolve()
    for path, split in ((train, "train"), (test, "test")):
        if not path.is_file():
            raise TrainingDataError(f"{split} data file does not exist: {path}")

    train_suffix = train.suffix.lower()
    test_suffix = test.suffix.lower()
    if train_suffix != test_suffix:
        raise TrainingDataError(
            "train and test files must use the same format; "
            f"got '{train_suffix}' and '{test_suffix}'"
        )

    if train_suffix == ".csv":
        X_train, y_train, train_features, resolved_target = _load_csv(
            train, target_column=target_column
        )
        X_test, y_test, test_features, test_target = _load_csv(
            test, target_column=resolved_target
        )
        if train_features != test_features:
            raise TrainingDataError(
                "train/test feature columns differ: "
                f"train={list(train_features)}, test={list(test_features)}"
            )
        if resolved_target != test_target:
            raise TrainingDataError("train/test target columns differ")
        feature_names = train_features
        target_name = resolved_target
    elif train_suffix == ".npz":
        X_train, y_train = _load_npz(train, split="train")
        X_test, y_test = _load_npz(test, split="test")
        feature_names = tuple(f"feature_{index}" for index in range(X_train.shape[-1]))
        target_name = target_column or "y"
    else:
        raise TrainingDataError(
            f"unsupported data format '{train_suffix}'; expected .csv or .npz"
        )

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    _validate_arrays(X_train, y_train, split="train")
    _validate_arrays(X_test, y_test, split="test")
    if X_train.shape[1:] != X_test.shape[1:]:
        raise TrainingDataError(
            "train/test feature shapes differ: "
            f"train={X_train.shape[1:]}, test={X_test.shape[1:]}"
        )

    return LoadedTrainingData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        target_name=target_name,
    )


def _load_csv(
    path: Path,
    *,
    target_column: str | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = tuple(reader.fieldnames or ())
            if len(headers) < 2:
                raise TrainingDataError(
                    f"CSV must contain at least one feature and one target column: {path}"
                )
            target = _resolve_target(headers, target_column)
            features = tuple(name for name in headers if name != target)
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise TrainingDataError(f"CSV is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise TrainingDataError(f"failed to read CSV: {path}: {exc}") from exc

    if not rows:
        raise TrainingDataError(f"CSV contains no data rows: {path}")

    X_values: list[list[float]] = []
    y_values: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            feature_row = [float(row[name]) for name in features]
        except (TypeError, ValueError) as exc:
            raise TrainingDataError(
                f"non-numeric or missing feature in {path} at row {row_number}"
            ) from exc
        if not np.isfinite(feature_row).all():
            raise TrainingDataError(
                f"non-finite feature in {path} at row {row_number}"
            )
        target_value = row.get(target)
        if target_value is None or target_value == "":
            raise TrainingDataError(
                f"missing target '{target}' in {path} at row {row_number}"
            )
        X_values.append(feature_row)
        y_values.append(target_value)

    return (
        np.asarray(X_values, dtype=np.float64),
        _coerce_targets(y_values),
        features,
        target,
    )


def _resolve_target(headers: tuple[str, ...], requested: str | None) -> str:
    if requested is not None:
        if requested not in headers:
            raise TrainingDataError(
                f"target column '{requested}' is not present; columns={list(headers)}"
            )
        return requested

    lower_to_original = {name.lower(): name for name in headers}
    for candidate in _TARGET_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return headers[-1]


def _coerce_targets(values: list[str]) -> np.ndarray:
    try:
        integers = [int(value) for value in values]
        return np.asarray(integers, dtype=np.int64)
    except ValueError:
        pass

    try:
        floats = [float(value) for value in values]
        result = np.asarray(floats, dtype=np.float64)
        if not np.isfinite(result).all():
            raise TrainingDataError("target values contain NaN or infinity")
        return result
    except ValueError:
        return np.asarray(values, dtype=str)


def _load_npz(path: Path, *, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    try:
        with np.load(path, allow_pickle=False) as archive:
            x_key = f"X_{split}" if f"X_{split}" in archive else "X"
            y_key = f"y_{split}" if f"y_{split}" in archive else "y"
            if x_key not in archive or y_key not in archive:
                raise TrainingDataError(
                    f"NPZ {path} must contain X/y or X_{split}/y_{split} arrays"
                )
            return np.asarray(archive[x_key]), np.asarray(archive[y_key])
    except TrainingDataError:
        raise
    except (OSError, ValueError) as exc:
        raise TrainingDataError(f"failed to load NPZ {path}: {exc}") from exc


def _validate_arrays(X: np.ndarray, y: np.ndarray, *, split: str) -> None:
    if X.ndim < 2:
        raise TrainingDataError(f"{split} X must have at least 2 dimensions")
    if y.ndim != 1:
        raise TrainingDataError(f"{split} y must be one-dimensional")
    if X.shape[0] == 0:
        raise TrainingDataError(f"{split} split is empty")
    if X.shape[0] != y.shape[0]:
        raise TrainingDataError(
            f"{split} X/y row mismatch: {X.shape[0]} != {y.shape[0]}"
        )
    if np.issubdtype(X.dtype, np.number) and not np.isfinite(X).all():
        raise TrainingDataError(f"{split} X contains NaN or infinity")
