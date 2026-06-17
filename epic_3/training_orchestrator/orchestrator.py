"""Prepare, execute, and aggregate Epic-3 local training jobs.

The router remains responsible for creating the stable ``TrainingJob`` hand-off.
This module now also integrates Onkar's local training worker, persists job
state transitions, isolates per-model failures, and writes
``training_summary.json``.  Ray execution remains a later work item.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from epic_3.training.contracts import TrainingResult

from .contracts import (
    OrchestratorMetadata,
    SelectedModelConfig,
    TrainingJob,
    TrainingJobManifest,
    TrainingSummary,
)
from .errors import (
    InvalidModelConfigError,
    MissingDataSplitError,
    TrainingExecutionError,
)
from .model_router import ModelRouter
from .result_aggregator import TrainingResultAggregator


class TrainingWorker(Protocol):
    """Minimal interface implemented by the local and future Ray workers."""

    def run(self, job: TrainingJob) -> TrainingResult:
        """Execute one job and return a structured result."""


class TrainingOrchestrator:
    """Create training jobs, execute them locally, and aggregate results."""

    def __init__(self, model_library_root: str | Path) -> None:
        self.model_library_root = Path(model_library_root).expanduser().resolve()
        self.router = ModelRouter(self.model_library_root)
        self.aggregator = TrainingResultAggregator()

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
        self._write_manifest(manifest, destination)
        return manifest

    def execute_local(
        self,
        manifest: TrainingJobManifest,
        *,
        worker: TrainingWorker | None = None,
        target_column: str | None = None,
        manifest_path: str | Path | None = None,
        summary_path: str | Path | None = None,
    ) -> TrainingSummary:
        """Execute all queued jobs through Onkar's local training worker.

        Job states are persisted after every transition.  Any exception from one
        model is converted into a failed ``TrainingResult`` so the remaining
        jobs continue to run.
        """

        session_root = self._session_root(manifest)
        manifest_destination = Path(
            manifest_path or session_root / "training_jobs.json"
        ).resolve()
        summary_destination = Path(
            summary_path or session_root / "training_summary.json"
        ).resolve()

        if worker is None:
            # Local import keeps the orchestration contracts usable without
            # importing estimator dependencies during model routing.
            from epic_3.training.trainer import LocalTrainingWorker

            worker = LocalTrainingWorker(
                self.model_library_root,
                target_column=target_column,
            )

        invalid_states = [
            f"{job.model_id}:{job.status}"
            for job in manifest.jobs
            if job.status != "queued"
        ]
        if invalid_states:
            raise TrainingExecutionError(
                "local execution requires queued jobs; invalid states: "
                + ", ".join(invalid_states)
            )

        results: list[TrainingResult] = []
        for job in manifest.jobs:
            job.status = "running"
            self._write_manifest(manifest, manifest_destination)

            result = self._execute_one(worker, job)
            job.status = result.status
            results.append(result)
            self._write_manifest(manifest, manifest_destination)

        summary = self.aggregator.build(manifest=manifest, results=results)
        self.aggregator.write(summary, summary_destination)
        return summary

    def prepare_and_execute_local(
        self,
        *,
        session_id: str,
        metadata_path: str | Path,
        model_config_path: str | Path,
        train_path: str | Path,
        test_path: str | Path,
        session_dir: str | Path,
        target_column: str | None = None,
        manifest_path: str | Path | None = None,
        summary_path: str | Path | None = None,
        worker: TrainingWorker | None = None,
    ) -> TrainingSummary:
        """Convenience entry point for prepare -> local train -> aggregate."""

        session_root = Path(session_dir).resolve()
        destination = Path(
            manifest_path or session_root / "training_jobs.json"
        ).resolve()
        manifest = self.prepare(
            session_id=session_id,
            metadata_path=metadata_path,
            model_config_path=model_config_path,
            train_path=train_path,
            test_path=test_path,
            session_dir=session_root,
            output_path=destination,
        )
        return self.execute_local(
            manifest,
            worker=worker,
            target_column=target_column,
            manifest_path=destination,
            summary_path=summary_path,
        )

    @staticmethod
    def _execute_one(worker: TrainingWorker, job: TrainingJob) -> TrainingResult:
        try:
            raw_result = worker.run(job)
            result = TrainingResult.model_validate(raw_result)
            if result.model_id != job.model_id:
                raise TrainingExecutionError(
                    f"worker returned model_id '{result.model_id}' for '{job.model_id}'"
                )
            if result.model_name != job.model_name:
                raise TrainingExecutionError(
                    f"worker returned model_name '{result.model_name}' for "
                    f"'{job.model_name}'"
                )
            return result
        except Exception as exc:
            result = TrainingResult(
                model_id=job.model_id,
                model_name=job.model_name,
                status="failed",
                metrics={},
                model_path=None,
                training_time_sec=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
            # The normal worker writes this artifact itself.  Persist a fallback
            # result when the worker raises before reaching its artifact writer.
            try:
                from epic_3.training.artifact_writer import write_training_result

                write_training_result(
                    result,
                    job.output_dir,
                    extra={
                        "task_type": job.task_type,
                        "data_format": job.data_format,
                        "trainer_type": job.trainer_type,
                    },
                )
            except Exception:
                # Preserve failure isolation even when the output filesystem is
                # also unavailable; the session summary still records the error.
                pass
            return result

    @staticmethod
    def _session_root(manifest: TrainingJobManifest) -> Path:
        roots = {Path(job.output_dir).expanduser().resolve().parent for job in manifest.jobs}
        if len(roots) != 1:
            raise TrainingExecutionError(
                "all training job output directories must share one session directory"
            )
        return roots.pop()

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

    @classmethod
    def _write_manifest(cls, manifest: TrainingJobManifest, path: Path) -> None:
        cls._atomic_write_json(path, manifest.model_dump(mode="json"))

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
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
