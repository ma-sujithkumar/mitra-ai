from __future__ import annotations

import json
import logging
import os
import time
import tempfile
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from typing import Any, Callable, Protocol

from backend.activity_log import ActivityLog
from backend.config_loader import ConfigLoader
from backend.schemas.training import ExecutionMode
from backend.schemas.training import TrainingModelState
from backend.schemas.training import TrainingStartRequest
from backend.schemas.training import TrainingStatusResponse
from backend.session import SessionManager
from backend.orchestration.events import TrainingEvent
from backend.orchestration.events import TrainingEventBus
from backend.orchestration.events import TrainingEventSink
from backend.agents.ray_wrapper import RayExecutor
from backend.agents.training_orchestrator import TrainingOrchestrator
from backend.agents.training_orchestrator.contracts import TrainingSummary
from backend.orchestration.eval_runner import EvalRunner
from backend.orchestration.judge_loop import EvalArtifacts, JudgeLoop
from backend.orchestration.plotting import PipelinePlotGenerator
from backend.services.training_fallback import FallbackTrainingArtifactBuilder
from backend.services.training_fallback import FallbackTrainingArtifactError

# Per-session advanced overrides file (written by PUT /api/config/advanced).
ADVANCED_OVERRIDES_FILENAME = "config_overrides.json"


logger = logging.getLogger(__name__)


class TrainingServiceError(Exception):
    """Base class for API-safe training service failures."""


class TrainingSessionNotFoundError(TrainingServiceError):
    pass


class TrainingRunNotFoundError(TrainingServiceError):
    pass


class TrainingRunConflictError(TrainingServiceError):
    pass


class TrainingArtifactError(TrainingServiceError):
    def __init__(self, message: str, missing_paths: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_paths = missing_paths or []


class TrainingCancellationError(TrainingServiceError):
    pass


class ParallelExecutor(Protocol):
    def cancel_all(self, *, force: bool = True) -> int:
        ...

    def close(self) -> None:
        ...




class CancellationAwareExecutor:
    """Blocks late submissions and delegates cancellation to Ray safely."""

    def __init__(self, delegate: Any, cancel_event: Event) -> None:
        self.delegate = delegate
        self.cancel_event = cancel_event
        self.lock = RLock()

    def start(self) -> Any:
        with self.lock:
            self._require_active()
            return self.delegate.start()

    def submit_all(self, jobs: Any, **kwargs: Any) -> Any:
        with self.lock:
            self._require_active()
            return self.delegate.submit_all(jobs, **kwargs)

    def collect(self, handles: Any, **kwargs: Any) -> Any:
        self._require_active()
        return self.delegate.collect(handles, **kwargs)

    def cancel_all(self, *, force: bool = True) -> int:
        with self.lock:
            return int(self.delegate.cancel_all(force=force))

    def close(self) -> None:
        self.delegate.close()

    def _require_active(self) -> None:
        if self.cancel_event.is_set():
            raise RuntimeError("Training cancellation was requested")


class OrchestratorLike(Protocol):
    def prepare_and_execute_ray(self, **kwargs: Any) -> TrainingSummary:
        ...

    def prepare_and_execute_local(self, **kwargs: Any) -> TrainingSummary:
        ...


class TrainingRunEventSink:
    """Mirror training events into persistent API state before forwarding SSE."""

    def __init__(
        self,
        *,
        delegate: TrainingEventBus,
        on_event: Callable[[TrainingEvent], None],
    ) -> None:
        self.delegate = delegate
        self.on_event = on_event

    def emit(self, event: TrainingEvent) -> object:
        try:
            self.on_event(event)
        except Exception:
            # Persistent status is best-effort; SSE and training must continue.
            pass
        return self.delegate.emit(event)

    def close_session(self, session_id: str) -> None:
        self.delegate.close_session(session_id)

    def reset_session(self, session_id: str, *, clear_history: bool = True) -> None:
        self.delegate.reset_session(session_id, clear_history=clear_history)


OrchestratorFactory = Callable[[Path, TrainingEventSink], OrchestratorLike]
ExecutorFactory = Callable[[Path, str | None], ParallelExecutor]


class ResolvedTrainingPaths:
    def __init__(
        self,
        *,
        session_path: Path,
        session_output_dir: Path,
        metadata_path: Path,
        model_config_path: Path,
        train_path: Path,
        test_path: Path,
        manifest_path: Path,
        summary_path: Path,
        status_path: Path,
    ) -> None:
        self.session_path = session_path
        self.session_output_dir = session_output_dir
        self.metadata_path = metadata_path
        self.model_config_path = model_config_path
        self.train_path = train_path
        self.test_path = test_path
        self.manifest_path = manifest_path
        self.summary_path = summary_path
        self.status_path = status_path


class TrainingService:
    """Starts training asynchronously and owns run status/cancellation state."""

    terminal_statuses = {
        "completed",
        "partial_failure",
        "failed",
        "cancelled",
    }
    terminal_model_statuses = {"completed", "failed", "timed_out", "cancelled"}

    def __init__(
        self,
        *,
        config_loader: ConfigLoader,
        session_manager: SessionManager,
        event_bus: TrainingEventBus,
        orchestrator_factory: OrchestratorFactory | None = None,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self.config_loader = config_loader
        self.session_manager = session_manager
        self.event_bus = event_bus
        self.orchestrator_factory = (
            orchestrator_factory or self._default_orchestrator_factory
        )
        self.executor_factory = executor_factory or self._default_executor_factory
        self.worker_pool = ThreadPoolExecutor(
            max_workers=config_loader.training_api.max_concurrent_runs,
            thread_name_prefix="mitra-training",
        )
        self.runs: dict[str, TrainingStatusResponse] = {}
        self.futures: dict[str, Future[None]] = {}
        self.executors: dict[str, ParallelExecutor] = {}
        self.cancel_events: dict[str, Event] = {}
        self.status_paths: dict[str, Path] = {}
        self.lock = RLock()

    def start(self, request: TrainingStartRequest) -> TrainingStatusResponse:
        paths = self.resolve_paths(request)
        execution_mode = self.resolve_execution_mode(request.execution_mode)

        with self.lock:
            existing = self.runs.get(request.session_id)
            if existing is not None or paths.status_path.is_file():
                raise TrainingRunConflictError(
                    f"Training already exists for session: {request.session_id}"
                )

            now = self._utc_now()
            state = TrainingStatusResponse(
                session_id=request.session_id,
                status="created",
                execution_mode=execution_mode,
                created_at=now,
                cancellation_requested=False,
                cancelled_jobs=0,
                manifest_path=str(paths.manifest_path),
                summary_path=str(paths.summary_path),
            )
            self.runs[request.session_id] = state
            self.cancel_events[request.session_id] = Event()
            self.status_paths[request.session_id] = paths.status_path
            self.event_bus.reset_session(request.session_id, clear_history=True)
            self._persist_state(state, paths.status_path)
            future = self.worker_pool.submit(
                self._run_training,
                request,
                paths,
                execution_mode,
            )
            self.futures[request.session_id] = future
            return state.model_copy(deep=True)

    def get_status(self, session_id: str) -> TrainingStatusResponse:
        with self.lock:
            state = self.runs.get(session_id)
            if state is not None:
                return state.model_copy(deep=True)

        session_path = self.session_manager.get_session_path(session_id=session_id)
        status_path = (
            session_path
            / self.config_loader.training_api.session_output_dir
            / self.config_loader.training_api.run_status_filename
        )
        if not status_path.is_file():
            raise TrainingRunNotFoundError(
                f"Training run not found for session: {session_id}"
            )
        return TrainingStatusResponse.model_validate_json(
            status_path.read_text(encoding="utf-8")
        )

    def cancel(self, session_id: str) -> TrainingStatusResponse:
        with self.lock:
            state = self.runs.get(session_id)
            if state is None:
                raise TrainingRunNotFoundError(
                    f"Training run not found for session: {session_id}"
                )
            if state.status in self.terminal_statuses:
                raise TrainingCancellationError(
                    f"Training run is already terminal: {state.status}"
                )

            cancel_event = self.cancel_events[session_id]
            cancel_event.set()
            executor = self.executors.get(session_id)
            cancelled_jobs = executor.cancel_all(force=True) if executor else 0
            cancelled_at = self._utc_now()
            cancelled_state = state.model_copy(
                update={
                    "status": "cancelled",
                    "cancellation_requested": True,
                    "cancelled_jobs": cancelled_jobs,
                    "finished_at": cancelled_at,
                    **self._cancelled_state_updates(state, cancelled_at),
                }
            )
            self.runs[session_id] = cancelled_state
            self._persist_state(cancelled_state, self.status_paths[session_id])

        self._emit_cancelled_session(session_id, cancelled_jobs)
        return cancelled_state.model_copy(deep=True)

    def reset_run(self, session_id: str) -> None:
        """Clear out any existing training runs from memory and disk so it can be re-run."""
        with self.lock:
            self.runs.pop(session_id, None)
            self.futures.pop(session_id, None)
            self.executors.pop(session_id, None)
            self.cancel_events.pop(session_id, None)
            status_path = self.status_paths.pop(session_id, None)
            
            session_path = self.session_manager.get_session_path(session_id=session_id)
            if status_path is None:
                status_path = (
                    session_path
                    / self.config_loader.training_api.session_output_dir
                    / self.config_loader.training_api.run_status_filename
                )
            
            if status_path.is_file():
                try:
                    status_path.unlink()
                except Exception:
                    pass
            
            # Also clear the judge decision and training summary report files if they exist.
            reports_dir = session_path / self.config_loader.training_api.session_output_dir
            for filename in ("judge_decision.json", "training_summary.json"):
                report_file = reports_dir / filename
                if report_file.is_file():
                    try:
                        report_file.unlink()
                    except Exception:
                        pass
            
            self.event_bus.reset_session(session_id, clear_history=True)

    def shutdown(self) -> None:
        """Cancel active Ray work and stop accepting background jobs."""

        with self.lock:
            active_executors = list(self.executors.values())
            for cancel_event in self.cancel_events.values():
                cancel_event.set()
        for executor in active_executors:
            try:
                executor.cancel_all(force=True)
                executor.close()
            except Exception:
                pass
        self.worker_pool.shutdown(wait=True, cancel_futures=True)

    def resolve_paths(self, request: TrainingStartRequest) -> ResolvedTrainingPaths:
        try:
            session_path = self.session_manager.get_session_path(
                session_id=request.session_id
            ).resolve()
        except ValueError as exc:
            raise TrainingSessionNotFoundError(
                f"Invalid session id: {request.session_id}"
            ) from exc
        if not session_path.is_dir():
            raise TrainingSessionNotFoundError(
                f"Session not found: {request.session_id}"
            )

        training_config = self.config_loader.training_api
        metadata_path = self._resolve_input_path(
            session_path=session_path,
            explicit_path=request.metadata_path,
            candidates=training_config.metadata_candidates,
            label="metadata",
        )
        model_config_path = self._resolve_input_path(
            session_path=session_path,
            explicit_path=request.model_config_path,
            candidates=training_config.model_config_candidates,
            label="model_config",
        )
        train_path = self._resolve_input_path(
            session_path=session_path,
            explicit_path=request.train_path,
            candidates=training_config.train_candidates,
            label="train",
        )
        test_path = self._resolve_input_path(
            session_path=session_path,
            explicit_path=request.test_path,
            candidates=training_config.test_candidates,
            label="test",
        )

        required_paths = [
            metadata_path,
            model_config_path,
            train_path,
            test_path,
        ]
        missing_paths = [str(path) for path in required_paths if not path.is_file()]
        if missing_paths and request.allow_fallback_artifacts:
            try:
                self._create_fallback_artifacts(
                    request=request,
                    session_path=session_path,
                    metadata_path=metadata_path,
                    model_config_path=model_config_path,
                    train_path=train_path,
                    test_path=test_path,
                )
            except FallbackTrainingArtifactError as exc:
                logger.info(
                    "=> fallback training artifact creation skipped: session=%s reason=%s",
                    request.session_id,
                    exc,
                )
            missing_paths = [str(path) for path in required_paths if not path.is_file()]
        if missing_paths:
            raise TrainingArtifactError(
                "Required training artifacts are missing",
                missing_paths=missing_paths,
            )

        # Epic 1 metadata.json may use problem_type=supervised plus a
        # problem_subtype and can omit output_cols even when a target column is
        # known. Epic 3 routing requires the legacy classification/regression/
        # unsupervised value plus output_cols. Normalize at this boundary so
        # routing always receives a complete Epic-3-compatible metadata file.
        metadata_path = self._write_epic3_metadata(
            metadata_path=metadata_path,
            session_path=session_path,
            target_column=request.target_column,
            train_path=train_path,
        )

        session_output_dir = (
            session_path / training_config.session_output_dir
        ).resolve()
        session_output_dir.mkdir(parents=True, exist_ok=True)
        return ResolvedTrainingPaths(
            session_path=session_path,
            session_output_dir=session_output_dir,
            metadata_path=metadata_path,
            model_config_path=model_config_path,
            train_path=train_path,
            test_path=test_path,
            manifest_path=session_output_dir / training_config.manifest_filename,
            summary_path=session_output_dir / training_config.summary_filename,
            status_path=session_output_dir / training_config.run_status_filename,
        )


    def _create_fallback_artifacts(
        self,
        *,
        request: TrainingStartRequest,
        session_path: Path,
        metadata_path: Path,
        model_config_path: Path,
        train_path: Path,
        test_path: Path,
    ) -> None:
        """Create minimal artifacts so Epic-3 can run when upstream LLM stages fail."""

        problem_type = request.problem_type
        if problem_type == "auto":
            problem_type = None
        builder = FallbackTrainingArtifactBuilder(
            train_fraction=self.config_loader.pipeline.train_test_split,
        )
        result = builder.ensure(
            session_path=session_path,
            metadata_path=metadata_path,
            model_config_path=model_config_path,
            train_path=train_path,
            test_path=test_path,
            target_column=request.target_column,
            problem_type=problem_type,
        )
        if result.created_paths:
            ActivityLog(session_path=session_path).record(
                stage="training",
                level="WARNING",
                message=(
                    "Created fallback Epic-3 training artifacts because one or more "
                    "upstream metadata/model-selection/split artifacts were missing "
                    f"(problem_type={result.problem_type}, target={result.target_column}, "
                    f"train_rows={result.train_rows}, test_rows={result.test_rows})."
                ),
            )
            logger.info(
                "=> created fallback training artifacts: session=%s paths=%s",
                request.session_id,
                [str(path) for path in result.created_paths],
            )

    def _write_epic3_metadata(
        self,
        *,
        metadata_path: Path,
        session_path: Path,
        target_column: str | None,
        train_path: Path,
    ) -> Path:
        """Write an Epic-3-compatible metadata view and return its path.

        Epic-1 metadata can be valid for the setup page but still incomplete for
        Epic-3 routing. The router requires a legacy problem type
        (classification/regression/unsupervised) and, for supervised tasks, at
        least one output column. When the target is available from the UI,
        metadata, run_config, or train split, fill these fields deterministically
        before the orchestrator prepares jobs. This prevents pre-routing failures
        that otherwise stop ``training_jobs.json`` and ``training_summary.json``
        from being generated.
        """

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        translated_payload = dict(payload)
        legacy_problem_type = self._legacy_problem_type(payload=payload)
        if legacy_problem_type is not None:
            translated_payload["problem_type"] = legacy_problem_type
        else:
            legacy_problem_type = translated_payload.get("problem_type")

        if legacy_problem_type in {"classification", "regression"}:
            target = self._metadata_target_column(
                payload=translated_payload,
                session_path=session_path,
                explicit_target_column=target_column,
                train_path=train_path,
            )
            if target:
                translated_payload["target_column"] = target
                translated_payload["target_col"] = target
                output_cols = translated_payload.get("output_cols")
                if not isinstance(output_cols, list) or not output_cols:
                    translated_payload["output_cols"] = [target]
                if not translated_payload.get("target_col_type"):
                    translated_payload["target_col_type"] = (
                        "numeric" if legacy_problem_type == "regression" else "categorical"
                    )
                input_cols = translated_payload.get("input_cols")
                if not isinstance(input_cols, list) or not input_cols:
                    inferred_input_cols = self._metadata_input_columns(
                        target_column=target,
                        train_path=train_path,
                    )
                    if inferred_input_cols:
                        translated_payload["input_cols"] = inferred_input_cols

        if translated_payload == payload:
            return metadata_path

        epic3_metadata_path = metadata_path.parent / "metadata_epic3.json"
        epic3_metadata_path.write_text(
            json.dumps(translated_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return epic3_metadata_path

    @staticmethod
    def _metadata_target_column(
        *,
        payload: dict[str, Any],
        session_path: Path,
        explicit_target_column: str | None,
        train_path: Path,
    ) -> str | None:
        if explicit_target_column:
            return explicit_target_column
        for key in ("target_column", "target_col"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        output_cols = payload.get("output_cols")
        if isinstance(output_cols, list) and output_cols:
            return str(output_cols[0])
        run_config_path = session_path / "reports" / "run_config.json"
        if run_config_path.is_file():
            try:
                run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                run_config = {}
            for key in ("target_col", "target_column"):
                value = run_config.get(key)
                if isinstance(value, str) and value:
                    return value
        columns = TrainingService._csv_header(train_path)
        if len(columns) >= 2:
            return columns[-1]
        return None

    @staticmethod
    def _metadata_input_columns(
        *,
        target_column: str,
        train_path: Path,
    ) -> list[str]:
        return [column for column in TrainingService._csv_header(train_path) if column != target_column]

    @staticmethod
    def _csv_header(path: Path) -> list[str]:
        if not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                header = handle.readline().strip()
        except OSError:
            return []
        if not header:
            return []
        return [column.strip() for column in header.split(",") if column.strip()]

    @staticmethod
    def _legacy_problem_type(payload: dict[str, Any]) -> str | None:
        # Maps supervised/unsupervised (+ subtype) onto the Epic 3 enum. Returns
        # None when the payload already uses the legacy enum (nothing to do).
        problem_type = payload.get("problem_type")
        if problem_type == "unsupervised":
            return "unsupervised"
        if problem_type == "supervised":
            subtype = payload.get("problem_subtype")
            if subtype in {"classification", "regression"}:
                return subtype
            # Fall back to the target column type when subtype is absent.
            if payload.get("target_col_type") == "numeric":
                return "regression"
            return "classification"
        return None

    def resolve_execution_mode(
        self,
        requested_mode: ExecutionMode | None,
    ) -> ExecutionMode:
        mode = requested_mode or self.config_loader.training_api.default_execution_mode
        if mode not in {"ray", "local"}:
            raise TrainingArtifactError(f"Unsupported execution mode: {mode}")
        return mode

    def _run_training(
        self,
        request: TrainingStartRequest,
        paths: ResolvedTrainingPaths,
        execution_mode: ExecutionMode,
    ) -> None:
        cancel_event = self.cancel_events[request.session_id]
        if cancel_event.is_set():
            return

        self._update_state(
            request.session_id,
            status="running",
            started_at=self._utc_now(),
        )
        status_event_sink = TrainingRunEventSink(
            delegate=self.event_bus,
            on_event=self._record_training_event,
        )
        orchestrator = self.orchestrator_factory(
            self.config_loader.training_api.model_library_root,
            status_event_sink,
        )
        executor: ParallelExecutor | None = None
        # Set to the TrainingSummary only on a clean, non-cancelled success so
        # the post-training eval (run in finally) knows training really finished.
        completed_summary: TrainingSummary | None = None

        try:
            if cancel_event.is_set():
                return

            common_arguments: dict[str, Any] = {
                "session_id": request.session_id,
                "metadata_path": paths.metadata_path,
                "model_config_path": paths.model_config_path,
                "train_path": paths.train_path,
                "test_path": paths.test_path,
                "session_dir": paths.session_output_dir,
                "target_column": request.target_column,
                "manifest_path": paths.manifest_path,
                "summary_path": paths.summary_path,
            }
            if execution_mode == "ray":
                raw_executor = self.executor_factory(
                    self.config_loader.training_api.model_library_root,
                    request.target_column,
                )
                executor = CancellationAwareExecutor(raw_executor, cancel_event)
                with self.lock:
                    self.executors[request.session_id] = executor
                summary = orchestrator.prepare_and_execute_ray(
                    **common_arguments,
                    timeout_sec=(
                        request.timeout_sec
                        or self.config_loader.training_api.ray_timeout_sec
                    ),
                    executor=executor,
                    close_executor=False,
                )
            else:
                summary = orchestrator.prepare_and_execute_local(
                    **common_arguments,
                )

            if cancel_event.is_set():
                return
            self._write_training_summary_artifacts(paths, summary)
            summary_updates = self._summary_state_updates(summary)
            self._update_state(
                request.session_id,
                status=summary.status,
                finished_at=self._utc_now(),
                completed_models=summary.completed,
                failed_models=summary.failed,
                error=None,
                **summary_updates,
            )
            # Defer eval until after the executor is cleaned up (in finally) so
            # the executor is closed promptly once training is marked complete.
            completed_summary = summary
        except Exception as exc:
            if cancel_event.is_set():
                return
            message = f"{type(exc).__name__}: {exc}"
            failure_updates = self._failure_state_updates(
                request.session_id,
                message,
            )
            self._update_state(
                request.session_id,
                status="failed",
                finished_at=self._utc_now(),
                error=message,
                **failure_updates,
            )
            self._emit_failed_session(request.session_id, message)
        finally:
            if executor is not None:
                try:
                    executor.close()
                except Exception:
                    pass
            with self.lock:
                self.executors.pop(request.session_id, None)
            if cancel_event.is_set():
                self.event_bus.close_session(request.session_id)

        # Post-training evaluation runs only after the executor is cleaned up so
        # training is fully settled first. Runs SHAP + overfitting + HPT + judge
        # to populate the leaderboard/verdict. Non-fatal: training stays
        # "completed" even if evaluation fails (artifacts simply stay absent).
        has_eval = (
            completed_summary is not None
            and completed_summary.completed > 0
            and self.config_loader.pipeline.run_post_training_eval
            and not cancel_event.is_set()
        )
        if has_eval:
            self._run_post_training_evaluation(
                request,
                paths,
                completed_summary,
                orchestrator,
                execution_mode,
                executor,
                cancel_event,
            )
        else:
            if not cancel_event.is_set():
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=request.session_id,
                        status="all_completed",
                        level="info",
                        pct=100,
                        msg="Training completed successfully.",
                    )
                )
                self.event_bus.close_session(request.session_id)


    def _write_training_summary_artifacts(
        self,
        paths: ResolvedTrainingPaths,
        summary: TrainingSummary,
    ) -> None:
        """Persist the canonical training summary before optional evaluation.

        The orchestrator writes ``training/training_summary.json``, but the
        backend service is the UI/API boundary. Re-writing it here guarantees
        the file is complete and also mirrors the same payload into
        ``reports/training_summary.json`` so Leaderboard can render a
        training-only result even when Epic-4/Judge is disabled or skipped.
        """

        if hasattr(summary, "model_dump"):
            payload = summary.model_dump(mode="json")
        else:
            payload = {
                "session_id": getattr(summary, "session_id", paths.session_path.name),
                "status": getattr(summary, "status", "failed"),
                "total_models": getattr(summary, "total_models", None),
                "completed": getattr(summary, "completed", 0),
                "failed": getattr(summary, "failed", 0),
                "models": [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
                    for item in getattr(summary, "models", [])
                ],
            }
        for destination in (
            paths.summary_path,
            paths.session_path / "reports" / "training_summary.json",
        ):
            self._atomic_write_json(destination, payload)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
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

    def _run_post_training_evaluation(
        self,
        request: TrainingStartRequest,
        paths: ResolvedTrainingPaths,
        summary: TrainingSummary,
        orchestrator: OrchestratorLike,
        execution_mode: ExecutionMode,
        executor: ParallelExecutor | None,
        cancel_event: Event,
    ) -> None:
        """Run eval (SHAP/overfitting) + judge feedback loop after a successful training.

        Mirrors the headless run_pipeline stages 4-5 for the UI path. Any
        failure here is logged and swallowed so the training run stays green;
        the leaderboard simply reports a training-only result until artifacts
        appear.
        """
        try:
            self.event_bus.emit(
                TrainingEvent(
                    session_id=request.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg="[POST-TRAINING EVAL] Starting evaluation (SHAP, Overfitting) and model selection ranking.",
                    pct=10,
                )
            )
            task_type = self._read_task_type(paths.metadata_path)
            # Eval/judge/plot artifacts must land under the session ROOT (the
            # same base the evaluation router reads from), not the training
            # subdir, so the leaderboard/verdict/plot endpoints can find them.
            session_dir = paths.session_path

            # Mirror the training summary into reports/ so the leaderboard can
            # merge per-model metrics with the judge ranking.
            reports_dir = session_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "training_summary.json").write_text(
                summary.model_dump_json(indent=2),
                encoding="utf-8",
            )

            # Prefer the engineered dataset; fall back to the training split.
            engineered_csv = session_dir / "data" / "engineered_dataset.csv"
            if not engineered_csv.exists():
                engineered_csv = paths.train_path

            eval_runner = EvalRunner(
                session_id=request.session_id,
                session_dir=session_dir,
                task_type=task_type,
                target_column=request.target_column,
                event_bus=self.event_bus,
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=request.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg="[POST-TRAINING EVAL] Running parallel evaluation workers (SHAP explainers, overfitting analyzer)...",
                    pct=20,
                )
            )
            
            # Run overfitting and SHAP, excluding HPT from post-training evaluation per user instructions
            eval_output = eval_runner.run(
                training_summary=summary,
                engineered_dataset_path=engineered_csv,
                run_hpt=False,
            )

            self.event_bus.emit(
                TrainingEvent(
                    session_id=request.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg="[POST-TRAINING EVAL] Initial model evaluation completed. Starting Judge Agent multi-turn feedback trials...",
                    pct=40,
                )
            )
            metadata = self._read_json_or_none(paths.metadata_path)
            # Honour per-session advanced overrides (page-1 advanced settings)
            # over the config.ini default for the judge feedback loop length.
            max_judge_turns = self._resolve_max_judge_turns(request.session_id)
            judge_loop = JudgeLoop(
                task_type=task_type,
                max_turns=max_judge_turns,
                event_bus=self.event_bus,
                session_id=request.session_id,
            )

            # Re-train callback called by the Judge Agent feedback loop on model candidate exclusions
            def training_callback(excluded_names: list[str]) -> Any:
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=request.session_id,
                        stage="evaluation",
                        level="info",
                        status="running",
                        msg=f"[JUDGE FEEDBACK] Models rejected by Judge. Selecting new candidates excluding: {excluded_names}...",
                        pct=50,
                    )
                )

                if cancel_event.is_set():
                    raise RuntimeError("Cancellation requested during judge feedback loop")

                common_arguments = {
                    "session_id": request.session_id,
                    "metadata_path": paths.metadata_path,
                    "model_config_path": paths.model_config_path,
                    "train_path": paths.train_path,
                    "test_path": paths.test_path,
                    "session_dir": paths.session_output_dir,
                    "target_column": request.target_column,
                    "manifest_path": paths.manifest_path,
                    "summary_path": paths.summary_path,
                }
                
                # Overwrites model_config.json and triggers training on the new configuration candidates
                if execution_mode == "ray":
                    new_summary = orchestrator.prepare_and_execute_ray(
                        **common_arguments,
                        timeout_sec=(
                            request.timeout_sec
                            or self.config_loader.training_api.ray_timeout_sec
                        ),
                        executor=executor,
                        close_executor=False,
                    )
                else:
                    new_summary = orchestrator.prepare_and_execute_local(
                        **common_arguments,
                    )
                
                self._write_training_summary_artifacts(paths, new_summary)

                self.event_bus.emit(
                    TrainingEvent(
                        session_id=request.session_id,
                        stage="evaluation",
                        level="info",
                        status="running",
                        msg="[JUDGE FEEDBACK] Next candidate models trained. Running SHAP and Overfitting check...",
                        pct=60,
                    )
                )

                # Re-run evaluation without HPT for the new candidates
                nonlocal eval_output
                eval_output = eval_runner.run(
                    training_summary=new_summary,
                    engineered_dataset_path=engineered_csv,
                    run_hpt=False,
                )
                return new_summary, eval_output

            decision = judge_loop.run_with_feedback(
                eval_artifacts=EvalArtifacts(
                    shap_dirs=eval_output["shap_dirs"],
                    overfitting_dirs=eval_output["overfitting_dirs"],
                    hpt_results_path=None,
                ),
                training_summary=summary,
                session_dir=session_dir,
                training_callback=training_callback,
                metadata_path=paths.metadata_path,
                feature_selection_path=session_dir / "reports" / "feature_selection.json",
                mini_data_path=session_dir / "data" / "mini_dataset.csv",
                model_library_root=self.config_loader.training_api.model_library_root,
                max_models=10,
                dataset_id=request.session_id,
                metadata=metadata,
            )
            logger.info(
                "=> post-training eval complete: session=%s selected=%s",
                request.session_id,
                decision.selected_model,
            )

            # Ensure final training summary is mirrored to reports/ after feedback trials
            final_summary = None
            if paths.summary_path.is_file():
                try:
                    final_summary = TrainingSummary.model_validate_json(paths.summary_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if final_summary is None:
                final_summary = summary

            (reports_dir / "training_summary.json").write_text(
                final_summary.model_dump_json(indent=2),
                encoding="utf-8",
            )

            self.event_bus.emit(
                TrainingEvent(
                    session_id=request.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg=f"[POST-TRAINING EVAL] Judge loop complete. Selected best model: {decision.selected_model or 'None'}. Generating final plots...",
                    pct=80,
                )
            )
            # Generate on-demand visualizations for the UI plot popups; never
            # let a plotting failure affect the (already green) training run.
            self._generate_plots(session_dir, request.session_id)
            self.event_bus.emit(
                TrainingEvent(
                    session_id=request.session_id,
                    stage="evaluation",
                    level="info",
                    status="all_completed",
                    msg="[POST-TRAINING EVAL] Post-training evaluation chain fully complete. Leaderboard updated!",
                    pct=100,
                )
            )
            # Emit final top-level completion event to close the stream
            self.event_bus.emit(
                TrainingEvent(
                    session_id=request.session_id,
                    status="all_completed",
                    level="info",
                    pct=100,
                    msg="Pipeline run completed successfully.",
                )
            )

        except Exception as eval_exc:  # noqa: BLE001 - eval must never fail training
            logger.warning(
                "=> post-training evaluation skipped for session=%s: %s: %s",
                request.session_id,
                type(eval_exc).__name__,
                eval_exc,
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=request.session_id,
                    stage="evaluation",
                    level="warn",
                    status="failed",
                    msg=f"[POST-TRAINING EVAL] Evaluation skipped or failed: {eval_exc}",
                    pct=100,
                )
            )
        finally:
            self.event_bus.close_session(request.session_id)

    def _resolve_max_judge_turns(self, session_id: str) -> int:
        """Return the judge-turn count, preferring a per-session override."""
        default_turns = self.config_loader.pipeline.max_judge_turns
        session_root = self.session_manager.get_session_path(session_id=session_id)
        overrides_path = session_root / ADVANCED_OVERRIDES_FILENAME
        if not overrides_path.is_file():
            return default_turns
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        override_value = overrides.get("pipeline.max_judge_turns")
        return int(override_value) if override_value is not None else default_turns

    def _generate_plots(self, session_dir: Path, session_id: str) -> None:
        """Dump on-demand visualizations; swallow failures (non-fatal)."""
        try:
            plot_summary = PipelinePlotGenerator(session_dir=session_dir).generate_all()
            total_plots = sum(len(files) for files in plot_summary.values())
            logger.info(
                "=> plots generated: session=%s count=%d stages=%d",
                session_id,
                total_plots,
                len(plot_summary),
            )
        except Exception as plot_exc:  # noqa: BLE001 - plots must never fail training
            logger.warning(
                "=> plot generation skipped for session=%s: %s: %s",
                session_id,
                type(plot_exc).__name__,
                plot_exc,
            )

    @staticmethod
    def _read_task_type(metadata_path: Path) -> str:
        """Read the task type from metadata, accepting task_type or problem_type."""
        if not metadata_path.is_file():
            return "classification"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("task_type") or metadata.get("problem_type") or "classification"

    @staticmethod
    def _read_json_or_none(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _record_training_event(self, event: TrainingEvent) -> None:
        """Persist per-model lifecycle state without coupling training to SSE."""

        with self.lock:
            current = self.runs.get(event.session_id)
            status_path = self.status_paths.get(event.session_id)
            if current is None or status_path is None:
                return
            if current.status == "cancelled" and event.status != "cancelled":
                return

            if event.model_id is None:
                details = event.details
                updates: dict[str, Any] = {}
                if event.status == "all_completed":
                    updates = {
                        "total_models": self._optional_int(details.get("total_models")),
                        "completed_models": self._optional_int(details.get("completed")),
                        "failed_models": self._optional_int(details.get("failed")),
                    }
                    updates = {
                        key: value for key, value in updates.items() if value is not None
                    }
                if not updates:
                    return
                updated = current.model_copy(update=updates)
                self.runs[event.session_id] = updated
                self._persist_state(updated, status_path)
                return

            previous_by_id = {
                item.model_id: item for item in current.model_states
            }
            previous = previous_by_id.get(event.model_id)
            details = event.details
            previous_validation_score = (
                previous.validation_score if previous is not None else None
            )
            previous_model_path = previous.model_path if previous is not None else None
            previous_training_time = (
                previous.training_time_sec if previous is not None else None
            )
            previous_error = previous.error if previous is not None else None

            model_state = TrainingModelState(
                model_id=event.model_id,
                model_name=event.model_name or (
                    previous.model_name if previous is not None else event.model_id
                ),
                status=event.status,
                pct=event.pct,
                updated_at=event.ts,
                validation_score=self._optional_float(
                    details.get("validation_score"),
                    fallback=previous_validation_score,
                ),
                model_path=self._optional_string(
                    details.get("model_path"),
                    fallback=previous_model_path,
                ),
                training_time_sec=self._optional_float(
                    details.get("training_time_sec"),
                    fallback=previous_training_time,
                ),
                error=self._optional_string(
                    details.get("error"),
                    fallback=previous_error,
                ),
            )
            previous_by_id[event.model_id] = model_state
            model_states = sorted(
                previous_by_id.values(),
                key=lambda item: item.model_id,
            )
            status_counts = Counter(item.status for item in model_states)
            updated = current.model_copy(
                update={
                    "total_models": len(model_states),
                    "completed_models": status_counts.get("completed", 0),
                    "failed_models": (
                        status_counts.get("failed", 0)
                        + status_counts.get("timed_out", 0)
                    ),
                    "job_status_counts": dict(sorted(status_counts.items())),
                    "model_states": model_states,
                }
            )
            self.runs[event.session_id] = updated
            self._persist_state(updated, status_path)

    def _summary_state_updates(self, summary: TrainingSummary) -> dict[str, Any]:
        summary_models = getattr(summary, "models", None)
        if not summary_models:
            total_models = getattr(summary, "total_models", None)
            return {"total_models": total_models} if total_models is not None else {}

        now = self._utc_now()
        model_states = [
            TrainingModelState(
                model_id=item.model_id,
                model_name=item.model_name,
                status=(
                    "timed_out"
                    if item.status == "failed"
                    and item.error
                    and "timed out" in item.error.lower()
                    else item.status
                ),
                pct=100,
                updated_at=now,
                validation_score=item.validation_score,
                model_path=item.model_path,
                training_time_sec=item.training_time_sec,
                error=item.error,
            )
            for item in summary_models
        ]
        status_counts = Counter(item.status for item in model_states)
        return {
            "total_models": len(model_states),
            "job_status_counts": dict(sorted(status_counts.items())),
            "model_states": model_states,
        }

    def _failure_state_updates(
        self,
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        with self.lock:
            current = self.runs.get(session_id)
            if current is None or not current.model_states:
                return {}
            now = self._utc_now()
            failed_states = [
                item
                if item.status in self.terminal_model_statuses
                else item.model_copy(
                    update={
                        "status": "failed",
                        "pct": 100,
                        "updated_at": now,
                        "error": message,
                    }
                )
                for item in current.model_states
            ]
            status_counts = Counter(item.status for item in failed_states)
            return {
                "total_models": len(failed_states),
                "completed_models": status_counts.get("completed", 0),
                "failed_models": (
                    status_counts.get("failed", 0)
                    + status_counts.get("timed_out", 0)
                ),
                "job_status_counts": dict(sorted(status_counts.items())),
                "model_states": failed_states,
            }

    def _cancelled_state_updates(
        self,
        state: TrainingStatusResponse,
        timestamp: datetime,
    ) -> dict[str, Any]:
        cancelled_states = [
            item
            if item.status in self.terminal_model_statuses
            else item.model_copy(
                update={
                    "status": "cancelled",
                    "pct": 100,
                    "updated_at": timestamp,
                    "error": "Training cancellation was requested",
                }
            )
            for item in state.model_states
        ]
        status_counts = Counter(item.status for item in cancelled_states)
        return {
            "total_models": len(cancelled_states) or state.total_models,
            "completed_models": status_counts.get("completed", 0),
            "failed_models": (
                status_counts.get("failed", 0)
                + status_counts.get("timed_out", 0)
            ),
            "job_status_counts": dict(sorted(status_counts.items())),
            "model_states": cancelled_states,
        }

    @staticmethod
    def _optional_float(value: Any, fallback: float | None = None) -> float | None:
        if value is None:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_string(value: Any, fallback: str | None = None) -> str | None:
        if value is None:
            return fallback
        return str(value)

    def _update_state(self, session_id: str, **updates: Any) -> None:
        with self.lock:
            current = self.runs[session_id]
            if current.status == "cancelled" and updates.get("status") != "cancelled":
                return
            updated = current.model_copy(update=updates)
            self.runs[session_id] = updated
            self._persist_state(updated, self.status_paths[session_id])

    def _resolve_input_path(
        self,
        *,
        session_path: Path,
        explicit_path: str | None,
        candidates: list[str],
        label: str,
    ) -> Path:
        if explicit_path:
            raw_path = Path(explicit_path).expanduser()
            possible_paths = [raw_path] if raw_path.is_absolute() else [
                session_path / raw_path,
                self.config_loader.repo_root / raw_path,
            ]
            selected = next(
                (path for path in possible_paths if path.is_file()),
                possible_paths[0],
            ).resolve()
        else:
            if not candidates:
                raise TrainingArtifactError(
                    f"No configured path candidates for {label}"
                )
            possible_paths = [(session_path / candidate).resolve() for candidate in candidates]
            selected = next(
                (path for path in possible_paths if path.is_file()),
                possible_paths[0],
            )

        try:
            selected.relative_to(session_path)
        except ValueError as exc:
            raise TrainingArtifactError(
                f"{label} path must stay inside the session directory: {selected}"
            ) from exc
        return selected

    def _emit_cancelled_session(self, session_id: str, cancelled_jobs: int) -> None:
        self.event_bus.emit(
            TrainingEvent(
                session_id=session_id,
                status="cancelled",
                level="warn",
                pct=100,
                msg="Training cancellation requested",
                details={"cancelled_jobs": cancelled_jobs},
            )
        )
        self.event_bus.emit(
            TrainingEvent(
                session_id=session_id,
                status="all_completed",
                level="warn",
                pct=100,
                msg="Training session cancelled",
                details={
                    "summary_status": "cancelled",
                    "total_models": 0,
                    "completed": 0,
                    "failed": cancelled_jobs,
                },
            )
        )
        self.event_bus.close_session(session_id)

    def _emit_failed_session(self, session_id: str, message: str) -> None:
        self.event_bus.emit(
            TrainingEvent(
                session_id=session_id,
                status="all_completed",
                level="error",
                pct=100,
                msg="Training session failed before completion",
                details={
                    "summary_status": "failed",
                    "total_models": 0,
                    "completed": 0,
                    "failed": 0,
                    "error": message,
                },
            )
        )
        self.event_bus.close_session(session_id)

    @staticmethod
    def _persist_state(state: TrainingStatusResponse, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    state.model_dump(mode="json"),
                    handle,
                    indent=2,
                    sort_keys=True,
                )
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

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _default_orchestrator_factory(
        model_library_root: Path,
        event_bus: TrainingEventBus,
    ) -> TrainingOrchestrator:
        return TrainingOrchestrator(
            model_library_root,
            event_sink=event_bus,
        )

    @staticmethod
    def _default_executor_factory(
        model_library_root: Path,
        target_column: str | None,
    ) -> RayExecutor:
        return RayExecutor(
            model_library_root,
            target_column=target_column,
        )

    def run_hpt(self, session_id: str) -> None:
        """Run HPT asynchronously in the background and publish SSE events."""
        with self.lock:
            if not hasattr(self, "_active_hpt_runs"):
                self._active_hpt_runs = set()
            if session_id in self._active_hpt_runs:
                return
            self._active_hpt_runs.add(session_id)
            
        self.event_bus.reset_session(session_id, clear_history=False)
        self.worker_pool.submit(self._execute_hpt, session_id)

    def _execute_hpt(self, session_id: str) -> None:
        try:
            self.event_bus.emit(
                TrainingEvent(
                    session_id=session_id,
                    stage="hpt",
                    level="info",
                    status="running",
                    msg="[HPT TUNING] Starting hyperparameter optimization for the top-1 model selected by Judge Agent...",
                    pct=10,
                )
            )
            session_path = self.session_manager.get_session_path(session_id=session_id)
            
            # 1. Read top 5 model names from judge_decision.json
            judge_decision_path = session_path / "reports" / "judge_decision.json"
            top_model_names = []
            if judge_decision_path.is_file():
                try:
                    decision_data = json.loads(judge_decision_path.read_text(encoding="utf-8"))
                    ranked = decision_data.get("ranked_models") or []
                    # Only tune the single top-ranked model (rank 1) per user requirement
                    top_model_names = [m.get("model_name") for m in ranked if m.get("model_name")][:1]
                except Exception as exc:
                    logger.warning("=> failed to load judge_decision for HPT filtering: %s", exc)
            
            # If no judge decision, read from model_config.json
            if not top_model_names:
                try:
                    model_config_path = session_path / "model_config.json"
                    if model_config_path.is_file():
                        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
                        # Fallback: pick only the first model when judge hasn't run yet
                        top_model_names = [m.get("name") for m in model_config if m.get("name")][:1]
                except Exception:
                    pass
            
            if not top_model_names:
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=session_id,
                        stage="hpt",
                        level="warn",
                        status="failed",
                        msg="[HPT TUNING] No candidate models found for hyperparameter optimization.",
                        pct=100,
                    )
                )
                return

            self.event_bus.emit(
                TrainingEvent(
                    session_id=session_id,
                    stage="hpt",
                    level="info",
                    status="running",
                    msg=f"[HPT TUNING] Tuning top-1 model: {top_model_names[0]}",
                    pct=20,
                )
            )

            # 2. Run HyperparameterTuningAgent
            from backend.agents.evaluation.hpt.agent import HyperparameterTuningAgent
            hpt_agent = HyperparameterTuningAgent(
                session_id=session_id,
                verbose=True,
            )
            
            # Restrict to top-1 model only; still run 5 Optuna trials for that model
            hpt_agent.model_config = [m for m in hpt_agent.model_config if m.get("name") in top_model_names]
            hpt_agent.model_config_sorted = sorted(hpt_agent.model_config, key=lambda x: x.get('priority', 999))
            
            # 5 Optuna trials for the single top-1 model
            hpt_agent.hpt_config['MAX_HPT_TRIALS'] = 5
            
            # Setup data splits
            X, y, metadata = hpt_agent.data_loader.load_train_data()
            val_ratio = float(hpt_agent.hpt_config.get('VAL_SPLIT_RATIO', 0.2))
            X_train, X_val, y_train, y_val = hpt_agent.data_loader.create_validation_split(
                X, y, hpt_agent.problem_type, val_ratio, 
                random_state=hpt_agent.hpt_config.get('OPTUNA_SEED', 42)
            )
            data_bundle = hpt_agent.data_loader.create_databundle(X_train, y_train, X_val, y_val)
            
            # Define Optuna trial callback to emit live status/progress and logs
            def optuna_trial_callback(study, trial) -> None:
                trial_num = trial.number + 1
                max_trials = 5
                val_score = trial.value
                try:
                    best_score = study.best_value
                except ValueError:
                    best_score = val_score if val_score is not None else 0.0
                
                if val_score is not None:
                    msg = f"[HPT] Trial {trial_num}/{max_trials} completed | {hpt_agent.primary_metric}={val_score:.4f} | Best {hpt_agent.primary_metric}: {best_score:.4f}"
                else:
                    msg = f"[HPT] Trial {trial_num}/{max_trials} completed | Pruned or failed | Best {hpt_agent.primary_metric}: {best_score:.4f}"
                
                # Calculate granular progress pct (20% to 80% span)
                trial_pct = 20 + int((trial_num / max_trials) * 60)
                
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=session_id,
                        stage="hpt",
                        level="info",
                        status="running",
                        msg=msg,
                        pct=trial_pct,
                        details={
                            "trial_number": trial_num,
                            "total_trials": max_trials,
                            "best_score": best_score,
                            "trial_score": val_score if val_score is not None else 0.0,
                            "trial_state": trial.state.name,
                            "params": trial.params
                        }
                    )
                )

            hpt_agent.results = []
            total_models = len(hpt_agent.model_config_sorted)
            
            for idx, model_entry in enumerate(hpt_agent.model_config_sorted, 1):
                model_name = model_entry.get('name')
                pct = 20 + int((idx - 1) / total_models * 60)
                
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=session_id,
                        stage="hpt",
                        level="info",
                        status="running",
                        msg=f"[HPT TUNING] Tuning model {idx}/{total_models}: {model_name} (5 Optuna trials)...",
                        pct=pct,
                    )
                )
                
                try:
                    result = hpt_agent.tune_model(model_entry, data_bundle, trial_callback=optuna_trial_callback)
                    if result:
                        hpt_agent.results.append(result)
                except Exception as e:
                    hpt_agent.failed_models.append(model_name)
                    logger.error("HPT failed for %s: %s", model_name, e)
            
            # Inject primary_metric into each result so the leaderboard endpoint
            # can surface it without knowing the agent's internal state.
            enriched_results = []
            for res in hpt_agent.results:
                enriched_res = dict(res)
                enriched_res.setdefault("primary_metric", hpt_agent.primary_metric)
                enriched_results.append(enriched_res)

            # Write results to canonical evaluation/hpt/hpt_results.json path
            hpt_output_path = session_path / "evaluation" / "hpt" / "hpt_results.json"
            hpt_output_path.parent.mkdir(parents=True, exist_ok=True)
            hpt_output_path.write_text(
                json.dumps({"hpt_results": enriched_results}, indent=2), encoding="utf-8"
            )
            
            # Determine best score from the top-1 tuned result
            best_score = None
            if enriched_results:
                val_metrics = enriched_results[0].get("val_metrics") or {}
                best_score = (
                    val_metrics.get("accuracy")
                    or val_metrics.get("r2")
                    or val_metrics.get("f1")
                    or next(iter(val_metrics.values()), None)
                )
            
            top1_model_name = top_model_names[0] if top_model_names else "top-1 model"
            best_score_str = f" | Best {hpt_agent.primary_metric}: {best_score:.4f}" if best_score is not None else ""
            self.event_bus.emit(
                TrainingEvent(
                    session_id=session_id,
                    stage="hpt",
                    level="info",
                    status="all_completed",
                    msg=f"[HPT TUNING] Hyperparameter tuning completed for {top1_model_name}.{best_score_str} Best params stored in leaderboard.",
                    pct=100,
                )
            )
        except Exception as exc:
            logger.exception("=> HPT execution failed: %s", exc)
            self.event_bus.emit(
                TrainingEvent(
                    session_id=session_id,
                    stage="hpt",
                    level="error",
                    status="failed",
                    msg=f"[HPT TUNING] Hyperparameter tuning failed: {exc}",
                    pct=100,
                )
            )
        finally:
            with self.lock:
                if hasattr(self, "_active_hpt_runs"):
                    self._active_hpt_runs.discard(session_id)
            self.event_bus.close_session(session_id)
