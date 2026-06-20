from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config_loader import ConfigLoader
from backend.routers import training
from backend.services.training_service import TrainingService


class ApiFakeExecutor:
    def cancel_all(self, *, force: bool = True) -> int:
        del force
        return 1

    def close(self) -> None:
        return None


class ApiFakeOrchestrator:
    def prepare_and_execute_ray(self, **kwargs: object) -> SimpleNamespace:
        Path(str(kwargs["summary_path"])).write_text("{}", encoding="utf-8")
        Path(str(kwargs["manifest_path"])).write_text("{}", encoding="utf-8")
        return SimpleNamespace(status="partial_failure", completed=1, failed=1)

    def prepare_and_execute_local(self, **kwargs: object) -> SimpleNamespace:
        return self.prepare_and_execute_ray(**kwargs)


def prepare_session(config_loader: ConfigLoader, session_id: str) -> None:
    session_path = config_loader.paths.workspace_root / session_id
    (session_path / "reports").mkdir(parents=True, exist_ok=True)
    (session_path / "data").mkdir(parents=True, exist_ok=True)
    (session_path / "reports" / "metadata.json").write_text("{}", encoding="utf-8")
    (session_path / "model_config.json").write_text("[]", encoding="utf-8")
    (session_path / "data" / "train.csv").write_text("x,y\n1,a\n", encoding="utf-8")
    (session_path / "data" / "test.csv").write_text("x,y\n2,b\n", encoding="utf-8")


def build_client(config_loader: ConfigLoader) -> TestClient:
    from backend.session import SessionManager
    from backend.orchestration.events import TrainingEventBus

    app = FastAPI()
    event_bus = TrainingEventBus()
    app.state.training_service = TrainingService(
        config_loader=config_loader,
        session_manager=SessionManager(config_loader.paths.workspace_root),
        event_bus=event_bus,
        orchestrator_factory=lambda model_root, bus: ApiFakeOrchestrator(),
        executor_factory=lambda model_root, target: ApiFakeExecutor(),
    )
    app.include_router(training.router)
    return TestClient(app)


def test_training_start_status_and_duplicate_contract(
    test_config_loader: ConfigLoader,
) -> None:
    session_id = "api_session"
    prepare_session(test_config_loader, session_id)
    client = build_client(test_config_loader)

    response = client.post(
        "/api/training/start",
        json={
            "session_id": session_id,
            "target_column": "y",
            "execution_mode": "ray",
        },
    )

    assert response.status_code == 202
    assert response.json()["session_id"] == session_id
    assert response.json()["events_url"].endswith(session_id)

    deadline = time.monotonic() + 2
    status_payload = None
    while time.monotonic() < deadline:
        status_response = client.get(f"/api/training/status/{session_id}")
        status_payload = status_response.json()
        if status_payload["status"] == "partial_failure":
            break
        time.sleep(0.01)
    assert status_payload is not None
    assert status_payload["status"] == "partial_failure"
    assert status_payload["completed_models"] == 1
    assert status_payload["failed_models"] == 1

    duplicate = client.post(
        "/api/training/start",
        json={"session_id": session_id},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["error"] == "TRAINING_ALREADY_EXISTS"


def test_training_start_missing_artifacts_returns_422(
    test_config_loader: ConfigLoader,
) -> None:
    session_id = "api_missing"
    (test_config_loader.paths.workspace_root / session_id).mkdir(parents=True)
    client = build_client(test_config_loader)

    response = client.post(
        "/api/training/start",
        json={"session_id": session_id},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "TRAINING_ARTIFACTS_INVALID"
    assert len(detail["missing_paths"]) == 4


def test_training_status_and_cancel_missing_run_return_404(
    test_config_loader: ConfigLoader,
) -> None:
    client = build_client(test_config_loader)

    status_response = client.get("/api/training/status/not_started")
    cancel_response = client.post("/api/training/cancel/not_started")

    assert status_response.status_code == 404
    assert cancel_response.status_code == 404


def test_training_start_invalid_session_id_returns_404(
    test_config_loader: ConfigLoader,
) -> None:
    client = build_client(test_config_loader)

    response = client.post(
        "/api/training/start",
        json={"session_id": "../invalid-session"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "SESSION_NOT_FOUND"
