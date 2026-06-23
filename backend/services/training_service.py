from __future__ import annotations

import json
import logging
import multiprocessing
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
from backend.agents.training_orchestrator.contracts import TrainingSummary, TrainingSummaryItem
from backend.orchestration.eval_runner import EvalRunner
from backend.orchestration.eval_runner import EvaluationRestartRequested
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

    # Feature-engineering-phase stages: pipeline_prep.py runs these once,
    # before training starts, and training never re-emits them. Their event-bus
    # history must survive a training (re)start the same way model_config.json
    # -- their file artifact -- already does (see reset_run()), or the Live
    # Training page's Dataset2Vec/Model Selection cards stay stuck on "pending"
    # forever once a clear_history reset drops their only events.
    _prep_phase_stages = {"d2v", "model_selection"}

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
        # Manager().Event() (not a raw multiprocessing.Event): the overfitting
        # branch is dispatched via ProcessPoolExecutor.submit() to an already-
        # running pool worker, so the restart signal must be picklable through
        # that call queue. A raw multiprocessing.Event only supports being
        # inherited by a process at fork/spawn time, not pickled to an
        # existing worker -- doing so raises "Condition objects should only be
        # shared between processes through inheritance". A Manager-backed
        # proxy is built for exactly this cross-process, post-fork use case.
        self._turn_restart_manager = multiprocessing.Manager()
        self.turn_restart_events: dict[str, Any] = {}
        self.status_paths: dict[str, Path] = {}
        self.lock = RLock()

    def _reset_session_preserving_prep_stages(self, session_id: str) -> None:
        """Reset the event bus, but carry forward d2v/model_selection events.

        Those FE-phase stages only ever run once, so a full clear_history wipe
        permanently loses their only events -- nothing downstream re-emits them.
        """
        preserved_events = [
            event for event in self.event_bus.history(session_id)
            if event.stage in self._prep_phase_stages
        ]
        self.event_bus.reset_session(session_id, clear_history=True)
        for event in preserved_events:
            self.event_bus.emit(event)

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
            self.turn_restart_events[request.session_id] = self._turn_restart_manager.Event()
            self.status_paths[request.session_id] = paths.status_path
            self._reset_session_preserving_prep_stages(request.session_id)
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
        persisted_state = TrainingStatusResponse.model_validate_json(
            status_path.read_text(encoding="utf-8")
        )
        # No in-memory run AND a non-terminal status on disk means the worker
        # thread that owned this run no longer exists (the server process was
        # restarted mid-run, e.g. while the judge feedback loop was training
        # or evaluating new candidates). Without this check the run looks
        # "running" forever even though nothing is left to ever finish it --
        # individual models can already show validation_score=completed while
        # the top-level status is stuck on "running" indefinitely.
        if persisted_state.status in self.terminal_statuses:
            return persisted_state
        orphaned_state = self._mark_orphaned_run(persisted_state)
        self._persist_state(orphaned_state, status_path)
        return orphaned_state

    def _mark_orphaned_run(self, state: TrainingStatusResponse) -> TrainingStatusResponse:
        """Convert a non-terminal run with no backing worker thread into a terminal failure."""
        orphaned_message = (
            "Training run was interrupted by a server restart and cannot resume. "
            "Please re-run training."
        )
        orphaned_model_states = [
            model_state
            if model_state.status in self.terminal_model_statuses
            else model_state.model_copy(
                update={"status": "failed", "pct": 100, "error": orphaned_message}
            )
            for model_state in state.model_states
        ]
        status_counts = Counter(model_state.status for model_state in orphaned_model_states)
        return state.model_copy(
            update={
                "status": "failed",
                "finished_at": self._utc_now(),
                "error": orphaned_message,
                "completed_models": status_counts.get("completed", 0),
                "failed_models": status_counts.get("failed", 0) + status_counts.get("timed_out", 0),
                "job_status_counts": dict(sorted(status_counts.items())),
                "model_states": orphaned_model_states,
            }
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

    def request_turn_restart(self, session_id: str) -> TrainingStatusResponse:
        """Signal a mid-turn restart: kill the current turn's running SHAP/overfitting
        subprocesses and redo that SAME judge turn from scratch, leaving previously
        completed turns and the rest of the run untouched (unlike cancel(), which
        aborts the whole run).
        """
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
            restart_event = self.turn_restart_events.get(session_id)
            if restart_event is None:
                raise TrainingCancellationError(
                    f"No active judge turn to restart for session: {session_id}"
                )
            restart_event.set()
            return state.model_copy(deep=True)

    def reset_run(self, session_id: str) -> None:
        """Clear out any existing training runs from memory and disk so it can be re-run."""
        with self.lock:
            self.runs.pop(session_id, None)
            self.futures.pop(session_id, None)
            self.executors.pop(session_id, None)
            self.cancel_events.pop(session_id, None)
            self.turn_restart_events.pop(session_id, None)
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
            
            # Clear training/eval OUTPUT report files so the pipeline re-runs clean.
            # NOTE: model_config.json is deliberately preserved -- it is a feature-
            # engineering/model-selection INPUT that training consumes, not a
            # training output. Deleting it left a re-run with no model config and
            # the start call failed with TRAINING_ARTIFACTS_INVALID.
            reports_dir = session_path / "reports"
            for filename in ("judge_decision.json", "training_summary.json", "judge_status.json"):
                report_file = reports_dir / filename
                if report_file.is_file():
                    try:
                        report_file.unlink()
                    except Exception:
                        pass
            
            # Clear evaluation outputs
            eval_dir = session_path / "evaluation"
            if eval_dir.is_dir():
                import shutil
                try:
                    shutil.rmtree(eval_dir)
                except Exception:
                    pass
            
            self._reset_session_preserving_prep_stages(session_id)

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
        self._turn_restart_manager.shutdown()

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
        """Normalize metadata.json in place for Epic-3 routing and return its path.

        Epic-1 metadata can be valid for the setup page but still incomplete for
        Epic-3 routing. The router requires a legacy problem type
        (classification/regression/unsupervised) and, for supervised tasks, at
        least one output column. When the target is available from the UI,
        metadata, run_config, or train split, fill these fields deterministically
        before the orchestrator prepares jobs. This prevents pre-routing failures
        that otherwise stop ``training_jobs.json`` and ``training_summary.json``
        from being generated.

        The normalized fields are written back into the same ``metadata.json``
        (no separate ``metadata_epic3.json`` artifact is produced) so every
        downstream consumer reads a single metadata file.
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

        # Write the normalized fields back into the same metadata.json instead of
        # emitting a duplicate metadata_epic3.json artifact.
        metadata_path.write_text(
            json.dumps(translated_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return metadata_path

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
            # Keep status as "running" here — the full pipeline (eval + judge turns) is still
            # in progress. Final terminal status is set after _run_post_training_evaluation()
            # so the frontend status poll never sees "completed" while judge turns are pending.
            self._update_state(
                request.session_id,
                status="running",
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
            # Eval (including all judge turns) is complete — update to the final terminal status
            # so the frontend status poll reflects the true pipeline outcome.
            if completed_summary is not None and not cancel_event.is_set():
                self._update_state(
                    request.session_id,
                    status=completed_summary.status,
                    finished_at=self._utc_now(),
                )
        else:
            if not cancel_event.is_set():
                # Eval skipped — mark terminal status now so the status poll stops.
                if completed_summary is not None:
                    self._update_state(
                        request.session_id,
                        status=completed_summary.status,
                        finished_at=self._utc_now(),
                    )
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

            turn_restart_event = self.turn_restart_events.get(request.session_id)
            eval_runner = EvalRunner(
                session_id=request.session_id,
                session_dir=session_dir,
                task_type=task_type,
                target_column=request.target_column,
                event_bus=self.event_bus,
                shap_timeout_sec=self.config_loader.pipeline.shap_timeout_sec,
                overfitting_timeout_sec=self.config_loader.pipeline.overfitting_timeout_sec,
                shap_skip_model_classes=self.config_loader.pipeline.shap_skip_model_classes,
                restart_event=turn_restart_event,
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
            
            # Run overfitting and SHAP, excluding HPT from post-training evaluation per user instructions.
            # A user-requested restart (turn_restart_event set while this is running) kills the
            # in-flight SHAP/overfitting subprocesses and raises EvaluationRestartRequested; redo
            # this same (pre-turn-1) evaluation from scratch rather than failing the whole run.
            while True:
                try:
                    eval_output = eval_runner.run(
                        training_summary=summary,
                        engineered_dataset_path=engineered_csv,
                        run_hpt=False,
                    )
                    break
                except EvaluationRestartRequested:
                    if turn_restart_event is not None:
                        turn_restart_event.clear()
                    logger.info(
                        "=> post-training eval: restart requested by user, redoing initial evaluation."
                    )
                    self.event_bus.emit(
                        TrainingEvent(
                            session_id=request.session_id,
                            stage="evaluation",
                            level="info",
                            status="running",
                            msg="[POST-TRAINING EVAL] Restart requested by user. Re-running evaluation from scratch...",
                            pct=20,
                        )
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

            # Accumulated pool of all models and eval artifacts across all judge turns.
            # Starts with the initial training run; each callback call merges in new models.
            accumulated_summary_items: list[TrainingSummaryItem] = list(summary.models)
            accumulated_shap_dirs: dict[str, Any] = dict(eval_output.get("shap_dirs", {}))
            accumulated_overfitting_dirs: dict[str, Any] = dict(eval_output.get("overfitting_dirs", {}))

            # Re-train callback called by the Judge Agent feedback loop on model candidate exclusions.
            # model_config.json is already rewritten by _reselect_models() before this is called.
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

                # Filter model_config.json to exclude already-trained models (approved from
                # previous turns). The reselection agent includes approved models in its output
                # so the judge sees them in the next turn, but we must NOT re-train them.
                model_config_path = paths.session_path / "reports" / "model_config.json"
                already_trained_names: set[str] = {item.model_name for item in accumulated_summary_items}
                if model_config_path.is_file():
                    all_model_configs = json.loads(model_config_path.read_text(encoding="utf-8"))
                    new_only_configs = [cfg for cfg in all_model_configs if cfg.get("model_name") not in already_trained_names]
                    if new_only_configs:
                        # Re-number priorities so new candidates continue after
                        # already-trained models instead of reusing priority 1, 2, ...
                        # The model selection agent always starts from priority=1,
                        # so without renumbering each feedback turn would emit SSE events
                        # with the same priority values (e.g. #8, #9) as the previous
                        # turn's new models, causing duplicate card numbers in the UI.
                        priority_offset = len(accumulated_summary_items)
                        for new_cfg_index, new_cfg in enumerate(new_only_configs):
                            new_cfg["priority"] = priority_offset + new_cfg_index + 1
                        model_config_path.write_text(json.dumps(new_only_configs, indent=2), encoding="utf-8")
                        logger.info(
                            "=> training_callback: %d new models to train (filtered %d already-trained from config, priorities offset by %d)",
                            len(new_only_configs),
                            len(all_model_configs) - len(new_only_configs),
                            priority_offset,
                        )
                    else:
                        # All reselected models already trained — build merged summary from
                        # accumulated data and skip re-training to avoid wasted work.
                        logger.info("=> training_callback: all reselected models already trained, skipping re-train turn.")
                        total_count = len(accumulated_summary_items)
                        completed_count = sum(1 for item in accumulated_summary_items if item.status == "completed")
                        failed_count = sum(1 for item in accumulated_summary_items if item.status == "failed")
                        all_completed = completed_count == total_count
                        all_failed = failed_count == total_count
                        status_map: dict[tuple[bool, bool], str] = {(True, False): "completed", (False, True): "failed"}
                        skip_merged_summary = TrainingSummary(
                            session_id=request.session_id,
                            status=status_map.get((all_completed, all_failed), "partial_failure"),
                            total_models=total_count,
                            completed=completed_count,
                            failed=failed_count,
                            models=list(accumulated_summary_items),
                        )
                        skip_merged_eval_output: dict[str, Any] = {
                            "shap_dirs": dict(accumulated_shap_dirs),
                            "overfitting_dirs": dict(accumulated_overfitting_dirs),
                            "hpt_results_path": None,
                        }
                        return skip_merged_summary, skip_merged_eval_output

                common_arguments = {
                    "session_id": request.session_id,
                    "metadata_path": paths.metadata_path,
                    "model_config_path": model_config_path,
                    "train_path": paths.train_path,
                    "test_path": paths.test_path,
                    "session_dir": paths.session_output_dir,
                    "target_column": request.target_column,
                    "manifest_path": paths.manifest_path,
                    "summary_path": paths.summary_path,
                    # Assign IDs starting after all already-trained models so SSE events
                    # use unique model IDs (model_011, model_012 etc.) that don't conflict
                    # with existing model cards in the frontend.
                    "model_id_start": len(accumulated_summary_items) + 1,
                }

                # Notify the judge card that it is waiting for re-training before it can
                # resume evaluation. Without this event the judge card appears frozen because
                # no stage="judge" SSE events are emitted during the training_callback phase.
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=request.session_id,
                        stage="judge",
                        level="info",
                        status="running",
                        msg=f"[JUDGE] Rejected models excluded. Training {len(new_only_configs)} new candidate(s) before next evaluation turn...",
                        pct=40,
                    )
                )

                # Signal the frontend that a new training wave is beginning so the training
                # stage card and model list update even though initial training is "done".
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=request.session_id,
                        stage="training",
                        level="info",
                        status="running",
                        msg=f"[JUDGE FEEDBACK] Starting training for {len(new_only_configs)} new candidate(s)...",
                        pct=5,
                    )
                )

                # Train only the truly new models (model_config.json was filtered above).
                retrain_started = time.monotonic()
                if execution_mode == "ray":
                    # Pass executor=None so orchestrator creates a fresh Ray executor
                    # for this feedback turn. The original executor was closed in
                    # _run_training's finally block before _run_post_training_evaluation
                    # was called, so reusing it here would raise an already-closed error.
                    new_summary = orchestrator.prepare_and_execute_ray(
                        **common_arguments,
                        timeout_sec=(
                            request.timeout_sec
                            or self.config_loader.training_api.ray_timeout_sec
                        ),
                        executor=None,
                        close_executor=True,
                    )
                else:
                    new_summary = orchestrator.prepare_and_execute_local(
                        **common_arguments,
                    )
                logger.info(
                    "=> training_callback: retraining %d new candidate(s) (mode=%s) took %.1fs",
                    len(new_only_configs),
                    execution_mode,
                    time.monotonic() - retrain_started,
                )

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
                # Keep the judge card active while SHAP + overfitting analysis runs.
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=request.session_id,
                        stage="judge",
                        level="info",
                        status="running",
                        msg="[JUDGE] New candidates trained. Running SHAP + overfitting analysis before next judge turn...",
                        pct=50,
                    )
                )

                # Re-run evaluation without HPT for the new candidates only.
                eval_started = time.monotonic()
                new_eval_output = eval_runner.run(
                    training_summary=new_summary,
                    engineered_dataset_path=engineered_csv,
                    run_hpt=False,
                )
                logger.info(
                    "=> training_callback: SHAP + overfitting re-eval for %d new candidate(s) took %.1fs",
                    len(new_only_configs),
                    time.monotonic() - eval_started,
                )

                # Validate renumbered IDs match what the orchestrator emitted via SSE.
                # model_id_start = len(accumulated_summary_items) + 1, so the formula below
                # produces the same IDs the orchestrator assigned (model_011, model_012, ...).
                new_items_renumbered: list[TrainingSummaryItem] = []
                for new_item_index, new_item in enumerate(new_summary.models):
                    renumbered_slot = len(accumulated_summary_items) + new_item_index + 1
                    new_item_data = new_item.model_dump()
                    new_item_data["model_id"] = f"model_{renumbered_slot:03d}"
                    new_items_renumbered.append(TrainingSummaryItem.model_validate(new_item_data))

                accumulated_summary_items.extend(new_items_renumbered)

                # Merge eval artifact directories (later turns overwrite on collision by name).
                accumulated_shap_dirs.update(new_eval_output.get("shap_dirs", {}))
                accumulated_overfitting_dirs.update(new_eval_output.get("overfitting_dirs", {}))

                # Build a merged TrainingSummary satisfying the strict Pydantic validator.
                total_count = len(accumulated_summary_items)
                completed_count = sum(1 for item in accumulated_summary_items if item.status == "completed")
                failed_count = sum(1 for item in accumulated_summary_items if item.status == "failed")
                all_completed = completed_count == total_count
                all_failed = failed_count == total_count
                # Map (all_completed, all_failed) -> status string; default to partial_failure.
                status_lookup: dict[tuple[bool, bool], str] = {
                    (True, False): "completed",
                    (False, True): "failed",
                }
                merged_status = status_lookup.get((all_completed, all_failed), "partial_failure")

                merged_summary = TrainingSummary(
                    session_id=request.session_id,
                    status=merged_status,
                    total_models=total_count,
                    completed=completed_count,
                    failed=failed_count,
                    models=list(accumulated_summary_items),
                )

                self._write_training_summary_artifacts(paths, merged_summary)

                # Correct the training summary count on the frontend.
                # execute_local() emits an all_completed event with only the
                # per-turn model count (e.g. 2 new models), but the leaderboard
                # needs the merged totals across all judge turns (e.g. 7 models).
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=request.session_id,
                        stage="training",
                        level="info",
                        status="all_completed",
                        pct=100,
                        msg=f"All model training jobs finished: {completed_count} completed, {failed_count} failed",
                        details={
                            "summary_status": merged_status,
                            "total_models": total_count,
                            "completed": completed_count,
                            "failed": failed_count,
                        },
                    )
                )

                # Notify the judge card that re-training + re-evaluation are done and the
                # Judge Agent is about to resume its next evaluation turn.
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=request.session_id,
                        stage="judge",
                        level="info",
                        status="running",
                        msg="[JUDGE] New candidates trained and evaluated. Judge Agent resuming next evaluation turn...",
                        pct=60,
                    )
                )

                merged_eval_output: dict[str, Any] = {
                    "shap_dirs": dict(accumulated_shap_dirs),
                    "overfitting_dirs": dict(accumulated_overfitting_dirs),
                    "hpt_results_path": None,
                }
                return merged_summary, merged_eval_output

            # metadata_model_selection.json uses the flat string list format required by
            # MetadataInput (model selection agent schema). metadata_epic3.json has input_cols
            # as dicts which fails Pydantic validation in select_models(), silently killing
            # the multi-turn feedback loop after turn 1.
            model_selection_metadata_path = self._resolve_model_selection_metadata_path(
                session_dir=session_dir,
                fallback_path=paths.metadata_path,
            )
            decision = judge_loop.run_with_feedback(
                eval_artifacts=EvalArtifacts(
                    shap_dirs=eval_output["shap_dirs"],
                    overfitting_dirs=eval_output["overfitting_dirs"],
                    hpt_results_path=None,
                ),
                training_summary=summary,
                session_dir=session_dir,
                training_callback=training_callback,
                metadata_path=model_selection_metadata_path,
                feature_selection_path=self._resolve_feature_selection_path(session_dir),
                mini_data_path=self._resolve_mini_data_path(session_dir),
                model_library_root=self.config_loader.training_api.model_library_root,
                max_models=10,
                dataset_id=request.session_id,
                metadata=metadata,
                turn_restart_event=turn_restart_event,
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

    def _resolve_model_selection_metadata_path(self, session_dir: Path, fallback_path: Path) -> Path:
        """Return metadata_model_selection.json if present; else fall back to fallback_path.

        PipelinePrep normalizes the raw metadata into metadata_model_selection.json, which
        has input_cols as a flat list of strings and a canonical problem_type value.
        The model selection agent (MetadataInput schema) requires this exact format.
        Using metadata_epic3.json (fallback_path) causes a ValidationError because its
        input_cols entries are dicts, not strings.
        """
        normalized_path = session_dir / "reports" / "metadata_model_selection.json"
        return normalized_path if normalized_path.is_file() else fallback_path

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

    def _resolve_mini_data_path(self, session_dir: Path) -> Path:
        """Return the path to the mini dataset CSV, trying known filenames in order."""
        mini_data_candidates = ["data/mini_data.csv", "data/mini_dataset.csv", "data/data.csv"]
        for candidate in mini_data_candidates:
            candidate_path = session_dir / candidate
            if candidate_path.is_file():
                return candidate_path
        # Return the first candidate even if missing (select_models handles None gracefully)
        return session_dir / "data" / "mini_data.csv"

    def _resolve_feature_selection_path(self, session_dir: Path) -> Path:
        """Return the feature_selection.json path, creating a minimal stub if missing.

        feature_selection.json is normally written by PipelinePrep during the
        feature engineering step. In fallback runs (no full pipeline), it may not
        exist. We write a minimal valid stub so model selection can still proceed.
        """
        feature_selection_candidates = [
            session_dir / "reports" / "feature_selection.json",
            session_dir / "feature_selection.json",
        ]
        existing = next((path for path in feature_selection_candidates if path.is_file()), None)
        if existing:
            return existing
        # Write a minimal stub that satisfies FeatureSelectionInput validation.
        stub_path = session_dir / "reports" / "feature_selection.json"
        stub_path.parent.mkdir(parents=True, exist_ok=True)
        stub_path.write_text(
            json.dumps({"keep": [], "drop": [], "engineered": [], "rationale": {}}),
            encoding="utf-8",
        )
        logger.info("=> wrote minimal feature_selection.json stub: %s", stub_path)
        return stub_path

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

    def run_hpt(self, session_id: str, top_n: int = 3, num_trials: int = 5) -> None:
        """Run HPT asynchronously in the background and publish SSE events.

        Args:
            session_id: Session to tune.
            top_n: Number of top Judge-ranked models to tune (default 3).
            num_trials: Optuna trials per model (default 5).
        """
        with self.lock:
            if not hasattr(self, "_active_hpt_runs"):
                self._active_hpt_runs = set()
            if session_id in self._active_hpt_runs:
                return
            self._active_hpt_runs.add(session_id)

        # Delete stale results from a prior tune synchronously, before this
        # call returns. /hpt reports status="complete" purely from the
        # existence of hpt_results.json (it has no notion of "a new run is in
        # flight"). Without this, a re-tune's first poll tick reads the OLD
        # file within ~0-2s of starting, immediately flips hptStatus back to
        # 'complete' with stale data, and kills the live SSE progress view
        # before any new trial events ever arrive.
        session_path = self.session_manager.get_session_path(session_id=session_id)
        hpt_eval_dir = session_path / "evaluation" / "hpt"
        for stale_filename in ("hpt_results.json", "hpt_summary.json"):
            stale_path = hpt_eval_dir / stale_filename
            if stale_path.is_file():
                stale_path.unlink()

        self.event_bus.reset_session(session_id, clear_history=False)
        self.worker_pool.submit(self._execute_hpt, session_id, top_n, num_trials)

    def _execute_hpt(self, session_id: str, top_n: int = 3, num_trials: int = 5) -> None:
        try:
            self.event_bus.emit(
                TrainingEvent(
                    session_id=session_id,
                    stage="hpt",
                    level="info",
                    status="running",
                    msg=f"[HPT TUNING] Starting hyperparameter optimization for the top-{top_n} model(s) selected by Judge Agent...",
                    pct=10,
                )
            )
            session_path = self.session_manager.get_session_path(session_id=session_id)

            # 1. Read top-N model names from judge_decision.json
            judge_decision_path = session_path / "reports" / "judge_decision.json"
            top_model_names = []
            if judge_decision_path.is_file():
                try:
                    decision_data = json.loads(judge_decision_path.read_text(encoding="utf-8"))
                    ranked = decision_data.get("ranked_models") or []
                    top_model_names = [m.get("model_name") for m in ranked if m.get("model_name")][:top_n]
                except Exception as exc:
                    logger.warning("=> failed to load judge_decision for HPT filtering: %s", exc)

            # If no judge decision, read from model_config.json
            if not top_model_names:
                try:
                    model_config_path = session_path / "model_config.json"
                    if model_config_path.is_file():
                        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
                        # Fallback: pick the first top_n models when judge hasn't run yet
                        top_model_names = [m.get("name") for m in model_config if m.get("name")][:top_n]
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
                    msg=f"[HPT TUNING] Tuning top-{len(top_model_names)} model(s): {', '.join(top_model_names)}",
                    pct=20,
                )
            )

            # 2. Run HyperparameterTuningAgent
            from backend.agents.evaluation.hpt.agent import HyperparameterTuningAgent
            hpt_agent = HyperparameterTuningAgent(
                session_id=session_id,
                verbose=True,
            )

            # Restrict to the top-N Judge-ranked models; run num_trials Optuna trials each.
            #
            # model_config.json is overwritten on every judge-loop turn with only that
            # turn's small re-selection batch (judge_loop.py::_reselect_models), so by the
            # time HPT runs it may no longer contain models the Judge ranked highly in an
            # earlier turn. Without this fallback, filtering by top_model_names silently
            # produces an empty list and HPT reports "completed for 0 model(s)" while still
            # naming the (un-tuned) Judge picks. Synthesize a minimal entry -- carrying the
            # built-in default hp_space -- for any Judge-ranked model missing from the file.
            existing_by_name = {m.get("name"): m for m in hpt_agent.model_config}
            restricted_model_config = []
            for model_name in top_model_names:
                model_entry = existing_by_name.get(model_name)
                if model_entry is None:
                    model_entry = {
                        "name": model_name,
                        "model_name": model_name,
                        "family": "unknown",
                        "priority": 999,
                        "default_hyperparameters": {},
                        "hp_space": hpt_agent._default_hp_spaces.get(model_name, {}),
                    }
                    logger.warning(
                        "=> HPT: %s ranked by Judge but absent from model_config.json; "
                        "using synthesized entry with default hp_space.",
                        model_name,
                    )
                restricted_model_config.append(model_entry)

            hpt_agent.model_config = restricted_model_config
            hpt_agent.model_config_sorted = sorted(hpt_agent.model_config, key=lambda x: x.get('priority', 999))

            hpt_agent.hpt_config['MAX_HPT_TRIALS'] = num_trials
            
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
                max_trials = num_trials
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
                        msg=f"[HPT TUNING] Tuning model {idx}/{total_models}: {model_name} ({num_trials} Optuna trials)...",
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
            
            # Determine the best-scoring model across all tuned results (not just
            # the first one — with top_n > 1 multiple models may have been tuned).
            def _result_score(res: dict) -> float:
                val_metrics = res.get("val_metrics") or {}
                value = (
                    val_metrics.get(hpt_agent.primary_metric)
                    or val_metrics.get("accuracy")
                    or val_metrics.get("r2")
                    or val_metrics.get("f1")
                    or next(iter(val_metrics.values()), None)
                )
                return value if value is not None else float("-inf")

            best_result = max(enriched_results, key=_result_score, default=None)
            best_score = _result_score(best_result) if best_result else None
            best_score = best_score if best_score != float("-inf") else None
            best_model_name = best_result.get("name") if best_result else None

            tuned_names = ", ".join(top_model_names) if top_model_names else "model(s)"
            best_score_str = (
                f" | Best {hpt_agent.primary_metric} ({best_model_name}): {best_score:.4f}"
                if best_score is not None
                else ""
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=session_id,
                    stage="hpt",
                    level="info",
                    status="all_completed",
                    msg=f"[HPT TUNING] Hyperparameter tuning completed for {len(enriched_results)} model(s) ({tuned_names}).{best_score_str} Best params stored in leaderboard.",
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
            # Deliberately do NOT close_session() here. The training pipeline
            # already closed this session before HPT ever runs (HPT only starts
            # from the Leaderboard, after training is fully done), so this call
            # was redundant on the first tune. On every re-tune it re-closed the
            # session right after finishing, which raced the next subscribe():
            # reset_session() (called at the top of run_hpt) un-closes it, but
            # if a run finished fast, this close_session() flipped it closed
            # again before/while the frontend's SSE viewer tried to subscribe,
            # producing an immediate empty stream with no live trial events.
            # Leaving the session open after HPT lets every subsequent re-tune
            # subscribe cleanly.
