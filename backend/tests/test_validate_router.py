import json

from fastapi.testclient import TestClient

from backend.config_loader import ConfigLoader
from backend.main import create_app


def upload_valid_csv(client: TestClient) -> str:
    response = client.post(
        "/api/upload",
        files={
            "dataset_file": (
                "iris.csv",
                (
                    b"feature_one,feature_two,target\n"
                    b"1,2,a\n2,3,a\n3,4,b\n4,5,b\n5,6,c\n6,7,c\n"
                    b"7,8,a\n8,9,b\n9,10,c\n10,11,a\n"
                ),
                "text/csv",
            )
        },
    )
    assert response.status_code == 200
    return response.json()["session_id"]


def test_validate_writes_run_config_and_report(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    session_id = upload_valid_csv(client=client)

    response = client.post(
        "/api/validate",
        json={
            "session_id": session_id,
            "target_col": "target",
            "validation_split": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"

    reports_dir = test_config_loader.paths.workspace_root / session_id / "reports"
    run_config = json.loads((reports_dir / "run_config.json").read_text("utf-8"))
    validation_report = json.loads(
        (reports_dir / "validation_report.json").read_text("utf-8")
    )
    assert run_config["target_col"] == "target"
    assert run_config["validation_split"] == 0.2
    assert validation_report["session_id"] == session_id
    assert validation_report["passed"] is True


def test_validate_events_stream_sse(test_config_loader: ConfigLoader) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    session_id = upload_valid_csv(client=client)

    client.post(
        "/api/validate",
        json={"session_id": session_id, "target_col": "target"},
    )
    response = client.get(f"/api/validate/events?session_id={session_id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "check"' in response.text
    assert '"type": "done"' in response.text


def test_rerunning_validation_overwrites_report(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    session_id = upload_valid_csv(client=client)
    reports_dir = test_config_loader.paths.workspace_root / session_id / "reports"

    client.post(
        "/api/validate",
        json={"session_id": session_id, "target_col": "target"},
    )
    (reports_dir / "validation_report.json").write_text(
        "{\"stale\": true}",
        encoding="utf-8",
    )
    client.post(
        "/api/validate",
        json={"session_id": session_id, "target_col": "target"},
    )

    validation_report = json.loads(
        (reports_dir / "validation_report.json").read_text("utf-8")
    )
    assert validation_report["session_id"] == session_id
    assert "stale" not in validation_report


def test_validate_missing_session_returns_404(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.post(
        "/api/validate",
        json={"session_id": "missing_session", "target_col": "target"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "SESSION_NOT_FOUND"
