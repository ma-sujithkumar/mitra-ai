"""Prepare, execute, and aggregate Epic-3 training jobs.

The router remains responsible for creating the stable ``TrainingJob`` hand-off.
This module integrates both Onkar's local training worker and Ray executor,
persists job-state transitions, isolates per-model failures, and writes
``training_summary.json``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from inspect import signature
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from backend.orchestration.events import NullTrainingEventSink, TrainingEvent, TrainingEventSink
from backend.agents.training.contracts import TrainingResult

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
    """Minimal interface implemented by the local training worker."""

    def run(self, job: TrainingJob) -> TrainingResult:
        """Execute one job and return a structured result."""


class ParallelTrainingExecutor(Protocol):
    """Minimal interface implemented by Onkar's ``RayExecutor``.

    Ray is imported lazily so model routing and local execution remain usable
    in environments where the optional Ray dependency is unavailable.
    """

    def start(self) -> Any:
        """Initialize or connect to the parallel execution runtime."""

    def submit_all(
        self,
        jobs: Sequence[TrainingJob],
        *,
        resource_overrides: Mapping[str, Any] | None = None,
    ) -> Sequence[Any]:
        """Submit all jobs without waiting for completion."""

    def collect(
        self,
        handles: Sequence[Any],
        *,
        timeout_sec: float | None = None,
        on_result: Callable[[TrainingResult], None] | None = None,
    ) -> Sequence[TrainingResult]:
        """Collect one structured result for each submitted job."""

    def close(self) -> None:
        """Release executor-owned resources and cancel active work."""


class TrainingOrchestrator:
    """Create jobs, execute locally or through Ray, and aggregate results."""

    def __init__(
        self,
        model_library_root: str | Path,
        *,
        event_sink: TrainingEventSink | None = None,
    ) -> None:
        self.model_library_root = Path(model_library_root).expanduser().resolve()
        self.router = ModelRouter(self.model_library_root)
        self.aggregator = TrainingResultAggregator()
        self.event_sink = event_sink or NullTrainingEventSink()

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

        self._reset_event_stream(session_id)
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
        for job in manifest.jobs:
            self._emit_job_event(
                manifest=manifest,
                job=job,
                status="queued",
                pct=0,
                msg=f"{job.model_name} queued for training",
                details={
                    "priority": job.priority,
                    "rationale": job.rationale,
                    "trainer_type": job.trainer_type,
                    "data_format": job.data_format,
                },
            )
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
        """Execute all queued jobs through Onkar's local training worker."""

        session_root = self._session_root(manifest)
        manifest_destination = Path(
            manifest_path or session_root / "training_jobs.json"
        ).resolve()
        summary_destination = Path(
            summary_path or session_root / "training_summary.json"
        ).resolve()

        if worker is None:
            from backend.agents.training.trainer import LocalTrainingWorker

            worker = LocalTrainingWorker(
                self.model_library_root,
                target_column=target_column,
            )

        self._require_queued_jobs(manifest, execution_mode="local")

        results: list[TrainingResult] = []
        for job in manifest.jobs:
            job.status = "running"
            self._write_manifest(manifest, manifest_destination)
            self._emit_job_event(
                manifest=manifest,
                job=job,
                status="running",
                pct=25,
                msg=f"{job.model_name} local training started",
            )

            result = self._execute_one(worker, job)
            job.status = result.status
            results.append(result)
            self._write_manifest(manifest, manifest_destination)
            self._emit_result_event(manifest=manifest, job=job, result=result)

        summary = self.aggregator.build(manifest=manifest, results=results)
        self.aggregator.write(summary, summary_destination)
        self._emit_summary_event(summary)
        self._close_event_stream(manifest.session_id)
        return summary

    def execute_ray(
        self,
        manifest: TrainingJobManifest,
        *,
        executor: ParallelTrainingExecutor | None = None,
        target_column: str | None = None,
        manifest_path: str | Path | None = None,
        summary_path: str | Path | None = None,
        timeout_sec: float | None = None,
        resource_overrides: Mapping[str, Any] | None = None,
        close_executor: bool = True,
    ) -> TrainingSummary:
        """Execute all queued jobs in parallel through Onkar's Ray executor.

        The orchestrator owns session state, not Ray internals. It marks every
        job as running before submission, accepts completion-order results,
        maps them back by ``model_id``, persists final states, and always writes
        a complete session summary. Missing, duplicated, malformed, or
        executor-level results are converted into per-model failures so one
        bad task never prevents the remaining results from being recorded.
        """

        session_root = self._session_root(manifest)
        manifest_destination = Path(
            manifest_path or session_root / "training_jobs.json"
        ).resolve()
        summary_destination = Path(
            summary_path or session_root / "training_summary.json"
        ).resolve()

        self._require_queued_jobs(manifest, execution_mode="Ray")

        owns_executor = executor is None
        if executor is None:
            from backend.agents.ray_wrapper import RayExecutor

            executor = RayExecutor(
                self.model_library_root,
                target_column=target_column,
            )

        for job in manifest.jobs:
            job.status = "running"
        self._write_manifest(manifest, manifest_destination)

        results: list[TrainingResult]
        emitted_terminal: dict[str, tuple[str, str | None]] = {}

        def on_result(raw_result: TrainingResult) -> None:
            try:
                result = TrainingResult.model_validate(raw_result)
                job = next(
                    item for item in manifest.jobs if item.model_id == result.model_id
                )
                if result.model_name != job.model_name:
                    return
                self._emit_result_event(manifest=manifest, job=job, result=result)
                emitted_terminal[result.model_id] = (result.status, result.error)
            except Exception:
                # Event callbacks are observability only.  Result validation and
                # failure normalization remain the orchestrator's responsibility.
                return

        try:
            executor.start()
            handles = executor.submit_all(
                manifest.jobs,
                resource_overrides=resource_overrides,
            )
            for job in manifest.jobs:
                self._emit_job_event(
                    manifest=manifest,
                    job=job,
                    status="submitted",
                    pct=10,
                    msg=f"{job.model_name} submitted to Ray",
                )
                self._emit_job_event(
                    manifest=manifest,
                    job=job,
                    status="running",
                    pct=25,
                    msg=f"{job.model_name} training started on Ray",
                )
            raw_results = self._collect_parallel_results(
                executor=executor,
                handles=handles,
                timeout_sec=timeout_sec,
                on_result=on_result,
            )
            results = self._normalize_parallel_results(
                manifest=manifest,
                raw_results=raw_results,
            )
        except Exception as exc:
            results = [
                self._failed_result(
                    job,
                    f"Ray execution failed: {type(exc).__name__}: {exc}",
                )
                for job in manifest.jobs
            ]
        finally:
            if close_executor or owns_executor:
                try:
                    executor.close()
                except Exception:
                    # Cleanup must never prevent state/summary persistence.
                    pass

        by_id = {result.model_id: result for result in results}
        for job in manifest.jobs:
            result = by_id[job.model_id]
            job.status = result.status
            self._write_manifest(manifest, manifest_destination)
            signature_value = (result.status, result.error)
            if emitted_terminal.get(job.model_id) != signature_value:
                self._emit_result_event(manifest=manifest, job=job, result=result)

        summary = self.aggregator.build(manifest=manifest, results=results)
        self.aggregator.write(summary, summary_destination)
        self._emit_summary_event(summary)
        self._close_event_stream(manifest.session_id)
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

    def prepare_and_execute_ray(
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
        timeout_sec: float | None = None,
        resource_overrides: Mapping[str, Any] | None = None,
        executor: ParallelTrainingExecutor | None = None,
        close_executor: bool = True,
    ) -> TrainingSummary:
        """Convenience entry point for prepare -> Ray train -> aggregate."""

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
        return self.execute_ray(
            manifest,
            executor=executor,
            target_column=target_column,
            manifest_path=destination,
            summary_path=summary_path,
            timeout_sec=timeout_sec,
            resource_overrides=resource_overrides,
            close_executor=close_executor,
        )


    @staticmethod
    def _collect_parallel_results(
        *,
        executor: ParallelTrainingExecutor,
        handles: Sequence[Any],
        timeout_sec: float | None,
        on_result: Callable[[TrainingResult], None],
    ) -> Sequence[TrainingResult]:
        """Use live callbacks when the executor supports them.

        Older/fake executors remain compatible, which keeps the hand-off
        contract stable while Onkar's real Ray executor can stream completions.
        """

        parameters = signature(executor.collect).parameters
        if "on_result" in parameters:
            return executor.collect(
                handles,
                timeout_sec=timeout_sec,
                on_result=on_result,
            )
        return executor.collect(handles, timeout_sec=timeout_sec)

    def _emit_job_event(
        self,
        *,
        manifest: TrainingJobManifest,
        job: TrainingJob,
        status: str,
        pct: int,
        msg: str,
        level: str = "info",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            event = TrainingEvent(
                session_id=manifest.session_id,
                level=level,
                msg=msg,
                pct=pct,
                status=status,
                model_id=job.model_id,
                model_name=job.model_name,
                details=dict(details or {}),
            )
            self.event_sink.emit(event)
        except Exception:
            # SSE is observability. It must never change a training outcome.
            pass

    def _emit_result_event(
        self,
        *,
        manifest: TrainingJobManifest,
        job: TrainingJob,
        result: TrainingResult,
    ) -> None:
        timed_out = bool(result.error and "timed out" in result.error.lower())
        status = "timed_out" if timed_out else result.status
        level = "error" if result.status == "failed" else "info"
        if timed_out:
            message = f"{job.model_name} training timed out"
        elif result.status == "failed":
            message = f"{job.model_name} training failed"
        else:
            message = f"{job.model_name} training completed"

        details: dict[str, Any] = {
            "training_time_sec": result.training_time_sec,
        }
        if result.error:
            details["error"] = result.error
        if result.model_path:
            details["model_path"] = result.model_path
        if "validation_score" in result.metrics:
            details["validation_score"] = result.metrics["validation_score"]

        self._emit_job_event(
            manifest=manifest,
            job=job,
            status=status,
            pct=100,
            msg=message,
            level=level,
            details=details,
        )

    def _emit_summary_event(self, summary: TrainingSummary) -> None:
        try:
            self.event_sink.emit(
                TrainingEvent(
                    session_id=summary.session_id,
                    status="all_completed",
                    pct=100,
                    level="error" if summary.status == "failed" else "info",
                    msg=(
                        "All model training jobs finished: "
                        f"{summary.completed} completed, {summary.failed} failed"
                    ),
                    details={
                        "summary_status": summary.status,
                        "total_models": summary.total_models,
                        "completed": summary.completed,
                        "failed": summary.failed,
                    },
                )
            )
        except Exception:
            pass

    def _reset_event_stream(self, session_id: str) -> None:
        reset = getattr(self.event_sink, "reset_session", None)
        if callable(reset):
            try:
                reset(session_id, clear_history=True)
            except Exception:
                pass

    def _close_event_stream(self, session_id: str) -> None:
        try:
            self.event_sink.close_session(session_id)
        except Exception:
            pass

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
            result = TrainingOrchestrator._failed_result(
                job,
                f"{type(exc).__name__}: {exc}",
            )
            # The normal worker writes this artifact itself. Persist a fallback
            # result when the worker raises before reaching its artifact writer.
            try:
                from backend.agents.training.artifact_writer import write_training_result

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

    @classmethod
    def _normalize_parallel_results(
        cls,
        *,
        manifest: TrainingJobManifest,
        raw_results: Sequence[TrainingResult] | Sequence[Any],
    ) -> list[TrainingResult]:
        """Return exactly one valid result per job in manifest order."""

        expected = {job.model_id: job for job in manifest.jobs}
        accepted: dict[str, TrainingResult] = {}
        invalid_by_id: dict[str, str] = {}

        for raw_result in raw_results:
            try:
                result = TrainingResult.model_validate(raw_result)
            except Exception:
                # A malformed response cannot be safely associated with a job.
                # Missing expected IDs are converted to failures below.
                continue

            job = expected.get(result.model_id)
            if job is None:
                # Ignore unknown IDs; they must not pollute this session.
                continue
            if result.model_id in accepted or result.model_id in invalid_by_id:
                accepted.pop(result.model_id, None)
                invalid_by_id[result.model_id] = (
                    "Ray executor returned duplicate results for this model_id"
                )
                continue
            if result.model_name != job.model_name:
                invalid_by_id[result.model_id] = (
                    "Ray executor returned model_name "
                    f"'{result.model_name}' instead of '{job.model_name}'"
                )
                continue
            accepted[result.model_id] = result

        normalized: list[TrainingResult] = []
        for job in manifest.jobs:
            if job.model_id in invalid_by_id:
                normalized.append(cls._failed_result(job, invalid_by_id[job.model_id]))
            elif job.model_id in accepted:
                normalized.append(accepted[job.model_id])
            else:
                normalized.append(
                    cls._failed_result(
                        job,
                        "Ray executor returned no result for this training job",
                    )
                )
        return normalized

    @staticmethod
    def _failed_result(job: TrainingJob, error: str) -> TrainingResult:
        return TrainingResult(
            model_id=job.model_id,
            model_name=job.model_name,
            status="failed",
            metrics={},
            model_path=None,
            training_time_sec=0.0,
            error=error,
        )

    @staticmethod
    def _require_queued_jobs(
        manifest: TrainingJobManifest,
        *,
        execution_mode: str,
    ) -> None:
        invalid_states = [
            f"{job.model_id}:{job.status}"
            for job in manifest.jobs
            if job.status != "queued"
        ]
        if invalid_states:
            raise TrainingExecutionError(
                f"{execution_mode} execution requires queued jobs; invalid states: "
                + ", ".join(invalid_states)
            )

    @staticmethod
    def _session_root(manifest: TrainingJobManifest) -> Path:
        roots = {
            Path(job.output_dir).expanduser().resolve().parent
            for job in manifest.jobs
        }
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
