"""Prepare training jobs and persist the Epic-3 hand-off manifest.

Execution is deliberately out of scope for this work item.  Onkar's training
worker/Ray executor will consume the generated ``training_jobs.json`` file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import (
    OrchestratorMetadata,
    SelectedModelConfig,
    TrainingJobManifest,
)
from .errors import InvalidModelConfigError, MissingDataSplitError
from .model_router import ModelRouter


class TrainingOrchestrator:
    """Create a validated, deterministic queue of training jobs."""

    def __init__(self, model_library_root: str | Path) -> None:
        self.router = ModelRouter(model_library_root)

    def prepare(
        self,
        *,
        session_id: str,
        metadata_path: str | Path,
        model_config_path: str | Path,
        train_path: str | Path,
        test_path: str | Path,
        session_dir: str | Path,
        output_path: str | Path | None = None,
    ) -> TrainingJobManifest:
        """Validate inputs, create jobs, and atomically write a manifest."""
        if not session_id.strip():
            raise InvalidModelConfigError("session_id must not be empty")

        train = self._require_file(train_path, "train")
        test = self._require_file(test_path, "test")
        session_root = Path(session_dir).resolve()
        session_root.mkdir(parents=True, exist_ok=True)

        metadata_payload = self._read_json(metadata_path, "metadata.json")
        model_payload = self._read_json(model_config_path, "model_config.json")
        if not isinstance(model_payload, list):
            raise InvalidModelConfigError("model_config.json must contain a JSON array")

        try:
            metadata = OrchestratorMetadata.model_validate(metadata_payload)
            selected = [SelectedModelConfig.model_validate(item) for item in model_payload]
        except ValidationError as exc:
            raise InvalidModelConfigError(f"Invalid orchestrator input: {exc}") from exc

        jobs = self.router.route_all(
            selected_models=selected,
            metadata=metadata,
            train_path=train,
            test_path=test,
            session_dir=session_root,
        )
        for job in jobs:
            Path(job.output_dir).mkdir(parents=True, exist_ok=False)

        manifest = TrainingJobManifest(
            session_id=session_id,
            problem_type=metadata.problem_type,
            data_format=metadata.data_format,
            total_jobs=len(jobs),
            jobs=jobs,
        )
        destination = Path(output_path or session_root / "training_jobs.json").resolve()
        self._atomic_write_json(destination, manifest.model_dump(mode="json"))
        return manifest

    @staticmethod
    def _read_json(path: str | Path, label: str) -> Any:
        source = Path(path)
        if not source.is_file():
            raise InvalidModelConfigError(f"{label} not found: {source}")
        try:
            return json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidModelConfigError(f"Unable to read {label}: {exc}") from exc

    @staticmethod
    def _require_file(path: str | Path, split_name: str) -> Path:
        source = Path(path).resolve()
        if not source.is_file():
            raise MissingDataSplitError(
                f"Epic-2 {split_name} split not found: {source}"
            )
        return source

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
