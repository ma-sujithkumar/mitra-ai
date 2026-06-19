from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from typing import Any, Callable, Protocol

from backend.config_loader import ConfigLoader
from backend.schemas.training import ExecutionMode
from backend.schemas.training import TrainingStartRequest
from backend.schemas.training import TrainingStatusResponse
from backend.session import SessionManager
from epic_3.events import TrainingEvent
from epic_3.events import TrainingEventBus
from epic_3.ray_wrapper import RayExecutor
from epic_3.training_orchestrator import TrainingOrchestrator
from epic_3.training_orchestrator.contracts import TrainingSummary


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


OrchestratorFactory = Callable[[Path, TrainingEventBus], OrchestratorLike]
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
            cancelled_state = state.model_copy(
                update={
                    "status": "cancelled",
                    "cancellation_requested": True,
                    "cancelled_jobs": cancelled_jobs,
                    "finished_at": self._utc_now(),
                }
            )
            self.runs[session_id] = cancelled_state
            self._persist_state(cancelled_state, self.status_paths[session_id])

        self._emit_cancelled_session(session_id, cancelled_jobs)
        return cancelled_state.model_copy(deep=True)

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
        self.worker_pool.shutdown(wait=False, cancel_futures=True)

    def resolve_paths(self, request: TrainingStartRequest) -> ResolvedTrainingPaths:
        session_path = self.session_manager.get_session_path(
            session_id=request.session_id
        ).resolve()
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
        if missing_paths:
            raise TrainingArtifactError(
                "Required training artifacts are missing",
                missing_paths=missing_paths,
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
        orchestrator = self.orchestrator_factory(
            self.config_loader.training_api.model_library_root,
            self.event_bus,
        )
        executor: ParallelExecutor | None = None

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
            self._update_state(
                request.session_id,
                status=summary.status,
                finished_at=self._utc_now(),
                completed_models=summary.completed,
                failed_models=summary.failed,
                error=None,
            )
        except Exception as exc:
            if cancel_event.is_set():
                return
            message = f"{type(exc).__name__}: {exc}"
            self._update_state(
                request.session_id,
                status="failed",
                finished_at=self._utc_now(),
                error=message,
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
