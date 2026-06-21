import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.config_loader import ConfigLoader
from backend.main import create_app
from backend.services.session_progress import SessionProgress


def _make_session(config_loader: ConfigLoader, session_id: str) -> Path:
    session_dir = config_loader.paths.workspace_root / session_id
    (session_dir / "reports" / "feature_engineering").mkdir(parents=True, exist_ok=True)
    (session_dir / "data").mkdir(parents=True, exist_ok=True)
    return session_dir


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_progress_empty_session_all_pending(test_config_loader: ConfigLoader) -> None:
    session_dir = _make_session(test_config_loader, "s_empty")
    progress = SessionProgress(session_dir=session_dir, config_loader=test_config_loader)

    assert progress.phase_status() == {
        "validation": "pending",
        "metadata": "pending",
        "feature_engineering": "pending",
        "training": "pending",
        "evaluation": "pending",
    }
    assert progress.first_incomplete_phase() == "validation"


def test_progress_validation_failed_blocks_at_validation(
    test_config_loader: ConfigLoader,
) -> None:
    session_dir = _make_session(test_config_loader, "s_failed")
    _write(session_dir / "reports" / "validation_report.json", {"passed": False})
    progress = SessionProgress(session_dir=session_dir, config_loader=test_config_loader)

    assert progress.phase_status()["validation"] == "failed"
    assert progress.first_incomplete_phase() == "validation"


def test_progress_partial_completion_next_is_feature_engineering(
    test_config_loader: ConfigLoader,
) -> None:
    session_dir = _make_session(test_config_loader, "s_partial")
    _write(session_dir / "reports" / "validation_report.json", {"passed": True})
    _write(session_dir / "reports" / "metadata.json", {"problem_type": "classification"})
    progress = SessionProgress(session_dir=session_dir, config_loader=test_config_loader)

    statuses = progress.phase_status()
    assert statuses["validation"] == "passed"
    assert statuses["metadata"] == "complete"
    assert statuses["feature_engineering"] == "pending"
    assert progress.first_incomplete_phase() == "feature_engineering"


def test_progress_all_complete_returns_none(test_config_loader: ConfigLoader) -> None:
    session_dir = _make_session(test_config_loader, "s_done")
    _write(session_dir / "reports" / "validation_report.json", {"passed": True})
    _write(session_dir / "reports" / "metadata.json", {})
    _write(session_dir / "reports" / "feature_engineering" / "feature_artifact.json", {})
    _write(session_dir / "reports" / "model_config.json", {})
    _write(session_dir / "reports" / "training_summary.json", {})
    _write(session_dir / "reports" / "judge_decision.json", {})
    progress = SessionProgress(session_dir=session_dir, config_loader=test_config_loader)

    assert progress.first_incomplete_phase() is None
    assert all(
        status in {"complete", "passed"} for status in progress.phase_status().values()
    )


def test_progress_endpoint_returns_phases_and_next(
    test_config_loader: ConfigLoader,
) -> None:
    session_dir = _make_session(test_config_loader, "s_api")
    _write(session_dir / "reports" / "validation_report.json", {"passed": True})
    _write(session_dir / "reports" / "metadata.json", {})
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.get("/api/runs/s_api/progress")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "s_api"
    assert payload["phases"]["metadata"] == "complete"
    assert payload["next_phase"] == "feature_engineering"


def test_progress_endpoint_unknown_session_404(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    response = client.get("/api/runs/does_not_exist/progress")
    assert response.status_code == 404


def test_metadata_starter_skips_when_cached(test_config_loader: ConfigLoader) -> None:
    session_dir = _make_session(test_config_loader, "s_meta_skip")
    (session_dir / "data" / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    _write(session_dir / "reports" / "metadata.json", {"problem_type": "classification"})
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.post("/api/metadata", json={"session_id": "s_meta_skip"})

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert response.json()["artifact"] == "metadata.json"


def test_feature_engineering_starter_skips_when_cached(
    test_config_loader: ConfigLoader,
) -> None:
    session_dir = _make_session(test_config_loader, "s_fe_skip")
    (session_dir / "data" / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    _write(session_dir / "reports" / "metadata.json", {})
    _write(session_dir / "reports" / "feature_engineering" / "feature_artifact.json", {})
    _write(session_dir / "reports" / "model_config.json", {})
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.post(
        "/api/feature-engineering", json={"session_id": "s_fe_skip"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
