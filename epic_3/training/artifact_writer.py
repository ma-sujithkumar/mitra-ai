"""Artifact helpers for one local training execution."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .contracts import TrainingResult
from .errors import ArtifactWriteError

MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "train_metrics.json"


def prepare_output_directory(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactWriteError(f"failed to create output directory {path}: {exc}") from exc
    if not path.is_dir():
        raise ArtifactWriteError(f"output path is not a directory: {path}")
    return path


def model_artifact_path(output_dir: str | Path) -> Path:
    return prepare_output_directory(output_dir) / MODEL_FILENAME


def metrics_artifact_path(output_dir: str | Path) -> Path:
    return prepare_output_directory(output_dir) / METRICS_FILENAME


def write_training_result(
    result: TrainingResult,
    output_dir: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = result.model_dump(mode="json")
    if extra:
        payload.update(extra)
    path = metrics_artifact_path(output_dir)
    _atomic_write_json(path, payload)
    return path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass
        raise ArtifactWriteError(f"failed to write JSON artifact {path}: {exc}") from exc
