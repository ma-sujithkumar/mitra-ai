import json
from pathlib import Path

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


def test_health_returns_uptime_and_llm_status_without_secrets(
    tmp_path: Path,
    test_config_loader: ConfigLoader,
) -> None:
    (tmp_path / ".env").write_text(
        "LLM_TYPE=openai\nLLM_API_KEY=secret-key\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["uptime_seconds"] >= 0
    assert payload["llm"]["provider"] == "openai"
    assert payload["llm"]["env_configured"] is True
    assert "secret-key" not in response.text
    assert "api_key" not in response.text.lower()


def test_public_config_returns_defaults_without_secrets(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.get("/api/config/public")

    assert response.status_code == 200
    payload = response.json()
    assert payload["upload"]["allowed_extensions"] == [".csv", ".xls", ".xlsx"]
    assert payload["upload"]["max_file_size_mb"] == 200
    assert payload["upload"]["recent_upload_limit"] == 5
    assert payload["pipeline"]["train_test_split"] == 0.8
    assert payload["llm"]["providers"] == ["openai", "anthropic", "gemini"]
    assert payload["llm"]["base_models"]["openai"] == "openai/gpt-5.1"
    assert payload["llm"]["base_urls"]["anthropic"] == "https://api.anthropic.com"
    assert "api_key" not in response.text.lower()


def test_advanced_config_lists_tunable_params(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))

    response = client.get("/api/config/advanced")

    assert response.status_code == 200
    payload = response.json()
    keys = {param["key"] for param in payload["params"]}
    assert "pipeline.max_judge_turns" in keys
    assert "training_api.default_execution_mode" in keys
    # Secrets and absolute paths must never be surfaced here.
    assert "api_key" not in response.text.lower()


def test_advanced_config_persists_and_validates_overrides(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    session_id = upload_valid_csv(client=client)

    # Valid override is saved and reflected on the next GET.
    save_response = client.put(
        f"/api/config/advanced?session_id={session_id}",
        json={"overrides": {"pipeline.max_judge_turns": 5}},
    )
    assert save_response.status_code == 200
    assert save_response.json()["saved"]["pipeline.max_judge_turns"] == 5

    overrides_file = (
        test_config_loader.paths.workspace_root / session_id / "config_overrides.json"
    )
    assert overrides_file.is_file()

    get_response = client.get(f"/api/config/advanced?session_id={session_id}")
    effective = {param["key"]: param["value"] for param in get_response.json()["params"]}
    assert effective["pipeline.max_judge_turns"] == 5

    # Out-of-range / unknown overrides are rejected with 422.
    bad_response = client.put(
        f"/api/config/advanced?session_id={session_id}",
        json={"overrides": {"pipeline.max_judge_turns": 999, "nope.key": 1}},
    )
    assert bad_response.status_code == 422
    assert "rejected" in bad_response.json()["detail"]


def test_runs_reads_workspace_sessions(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    session_id = upload_valid_csv(client=client)
    client.post(
        "/api/validate",
        json={
            "session_id": session_id,
            "target_col": "target",
        },
    )

    response = client.get("/api/runs?limit=10")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runs"][0]["session_id"] == session_id
    assert payload["runs"][0]["validation_status"] == "passed"
    assert payload["runs"][0]["metadata_status"] == "pending"


def test_run_stats_returns_stable_aggregate_fields(
    test_config_loader: ConfigLoader,
) -> None:
    client = TestClient(create_app(config_loader=test_config_loader))
    session_id = upload_valid_csv(client=client)
    reports_dir = test_config_loader.paths.workspace_root / session_id / "reports"
    (reports_dir / "metadata.json").write_text(
        json.dumps({"session_id": session_id}),
        encoding="utf-8",
    )

    response = client.get("/api/runs/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_uploads"] == 1
    assert payload["validated_runs"] == 0
    assert payload["metadata_runs"] == 1
    assert payload["leaderboard_runs"] == 0
