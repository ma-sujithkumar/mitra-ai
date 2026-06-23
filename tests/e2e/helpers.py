from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config_loader import ConfigLoader
from backend.routers import training
from backend.services.training_service import TrainingService
from backend.session import SessionManager
from backend.orchestration.events import TrainingEventBus


def build_training_client(
    config_loader: ConfigLoader,
    *,
    orchestrator_factory: Any | None = None,
    executor_factory: Any | None = None,
) -> tuple[TestClient, TrainingService, TrainingEventBus]:
    event_bus = TrainingEventBus()
    service = TrainingService(
        config_loader=config_loader,
        session_manager=SessionManager(config_loader.paths.workspace_root),
        event_bus=event_bus,
        orchestrator_factory=orchestrator_factory,
        executor_factory=executor_factory,
    )
    app = FastAPI()
    app.state.training_service = service
    app.include_router(training.router)
    return TestClient(app), service, event_bus


def write_rows(path: Path, header: list[str], rows: Iterable[Iterable[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def prepare_session(
    config_loader: ConfigLoader,
    *,
    session_id: str,
    metadata: dict[str, Any],
    model_config: list[dict[str, Any]],
    train_header: list[str],
    train_rows: Iterable[Iterable[Any]],
    test_rows: Iterable[Iterable[Any]],
) -> Path:
    session_path = config_loader.paths.workspace_root / session_id
    reports_dir = session_path / "reports"
    data_dir = session_path / "data"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (session_path / "model_config.json").write_text(
        json.dumps(model_config, indent=2),
        encoding="utf-8",
    )
    write_rows(data_dir / "train.csv", train_header, train_rows)
    write_rows(data_dir / "test.csv", train_header, test_rows)
    return session_path


def wait_for_terminal(
    client: TestClient,
    session_id: str,
    *,
    timeout_sec: float = 15.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    terminal = {"completed", "partial_failure", "failed", "cancelled"}
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/training/status/{session_id}")
        assert response.status_code == 200
        latest = response.json()
        if latest["status"] in terminal:
            return latest
        time.sleep(0.02)
    raise AssertionError(f"training did not finish: {latest}")
