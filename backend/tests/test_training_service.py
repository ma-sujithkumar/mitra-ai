from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from backend.config_loader import ConfigLoader
from backend.schemas.training import TrainingStartRequest
from backend.services.training_service import TrainingArtifactError
from backend.services.training_service import TrainingRunConflictError
from backend.services.training_service import TrainingService
from backend.session import SessionManager
from epic_3.events import TrainingEventBus


class FakeExecutor:
    def __init__(self) -> None:
        self.cancelled = 0
        self.closed = False

    def cancel_all(self, *, force: bool = True) -> int:
        del force
        self.cancelled += 2
        return 2

    def close(self) -> None:
        self.closed = True


class FakeOrchestrator:
    def __init__(
        self,
        *,
        started: Event | None = None,
        release: Event | None = None,
        summary_status: str = "completed",
    ) -> None:
        self.started = started
        self.release = release
        self.summary_status = summary_status

    def prepare_and_execute_ray(self, **kwargs: object) -> SimpleNamespace:
        return self._run(kwargs)

    def prepare_and_execute_local(self, **kwargs: object) -> SimpleNamespace:
        return self._run(kwargs)

    def _run(self, kwargs: dict[str, object]) -> SimpleNamespace:
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=3)

        summary_path = Path(str(kwargs["summary_path"]))
        manifest_path = Path(str(kwargs["manifest_path"]))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"status": self.summary_status}),
            encoding="utf-8",
        )
        manifest_path.write_text(json.dumps({"jobs": []}), encoding="utf-8")
        completed = 0 if self.summary_status == "failed" else 1
        failed = 1 if self.summary_status in {"failed", "partial_failure"} else 0
        return SimpleNamespace(
            status=self.summary_status,
            completed=completed,
            failed=failed,
        )


def create_training_artifacts(config_loader: ConfigLoader, session_id: str) -> Path:
    session_path = config_loader.paths.workspace_root / session_id
    (session_path / "reports").mkdir(parents=True, exist_ok=True)
    (session_path / "data").mkdir(parents=True, exist_ok=True)
    (session_path / "reports" / "metadata.json").write_text("{}", encoding="utf-8")
    (session_path / "model_config.json").write_text("[]", encoding="utf-8")
    (session_path / "data" / "train.csv").write_text("x,y\n1,a\n", encoding="utf-8")
    (session_path / "data" / "test.csv").write_text("x,y\n2,b\n", encoding="utf-8")
    return session_path


def wait_for_terminal(service: TrainingService, session_id: str) -> str:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = service.get_status(session_id).status
        if status in service.terminal_statuses:
            return status
        time.sleep(0.01)
    raise AssertionError("training did not reach a terminal state")


def test_start_runs_in_background_and_persists_summary(
    test_config_loader: ConfigLoader,
) -> None:
    session_id = "session_background"
    session_path = create_training_artifacts(test_config_loader, session_id)
    event_bus = TrainingEventBus()
    fake_executor = FakeExecutor()
    service = TrainingService(
        config_loader=test_config_loader,
        session_manager=SessionManager(test_config_loader.paths.workspace_root),
        event_bus=event_bus,
        orchestrator_factory=lambda model_root, bus: FakeOrchestrator(),
        executor_factory=lambda model_root, target: fake_executor,
    )

    state = service.start(
        TrainingStartRequest(
            session_id=session_id,
            target_column="y",
            execution_mode="ray",
        )
    )

    assert state.status in {"created", "running"}
    assert wait_for_terminal(service, session_id) == "completed"
    final_state = service.get_status(session_id)
    assert final_state.completed_models == 1
    assert Path(final_state.summary_path or "").is_file()
    assert (session_path / "training" / "training_run.json").is_file()
    assert fake_executor.closed is True


def test_duplicate_session_start_is_rejected(
    test_config_loader: ConfigLoader,
) -> None:
    session_id = "session_duplicate"
    create_training_artifacts(test_config_loader, session_id)
    release = Event()
    service = TrainingService(
        config_loader=test_config_loader,
        session_manager=SessionManager(test_config_loader.paths.workspace_root),
        event_bus=TrainingEventBus(),
        orchestrator_factory=lambda model_root, bus: FakeOrchestrator(release=release),
        executor_factory=lambda model_root, target: FakeExecutor(),
    )
    request = TrainingStartRequest(session_id=session_id, execution_mode="ray")
    service.start(request)

    with pytest.raises(TrainingRunConflictError):
        service.start(request)
    release.set()


def test_missing_artifacts_are_reported_before_background_start(
    test_config_loader: ConfigLoader,
) -> None:
    session_id = "session_missing"
    (test_config_loader.paths.workspace_root / session_id).mkdir(parents=True)
    service = TrainingService(
        config_loader=test_config_loader,
        session_manager=SessionManager(test_config_loader.paths.workspace_root),
        event_bus=TrainingEventBus(),
    )

    with pytest.raises(TrainingArtifactError) as error:
        service.start(TrainingStartRequest(session_id=session_id))

    assert len(error.value.missing_paths) == 4


def test_cancel_stops_active_ray_executor_and_preserves_cancelled_state(
    test_config_loader: ConfigLoader,
) -> None:
    session_id = "session_cancel"
    create_training_artifacts(test_config_loader, session_id)
    started = Event()
    release = Event()
    fake_executor = FakeExecutor()
    event_bus = TrainingEventBus()
    service = TrainingService(
        config_loader=test_config_loader,
        session_manager=SessionManager(test_config_loader.paths.workspace_root),
        event_bus=event_bus,
        orchestrator_factory=lambda model_root, bus: FakeOrchestrator(
            started=started,
            release=release,
        ),
        executor_factory=lambda model_root, target: fake_executor,
    )
    service.start(
        TrainingStartRequest(session_id=session_id, execution_mode="ray")
    )
    assert started.wait(timeout=2)

    cancelled = service.cancel(session_id)
    release.set()
    time.sleep(0.05)

    assert cancelled.status == "cancelled"
    assert cancelled.cancelled_jobs == 2
    assert service.get_status(session_id).status == "cancelled"
    assert event_bus.history(session_id)[-1].status == "all_completed"
