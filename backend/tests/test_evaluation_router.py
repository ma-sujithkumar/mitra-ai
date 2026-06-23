import json
from pathlib import Path
from fastapi.testclient import TestClient

from backend.config_loader import ConfigLoader
from backend.main import create_app


def prepare_session_with_statuses(config_loader: ConfigLoader, session_id: str) -> None:
    session_path = config_loader.paths.workspace_root / session_id
    (session_path / "reports").mkdir(parents=True, exist_ok=True)
    (session_path / "evaluation").mkdir(parents=True, exist_ok=True)


def test_evaluation_status_routes_defaults(test_config_loader: ConfigLoader) -> None:
    session_id = "test_eval_status_defaults"
    prepare_session_with_statuses(test_config_loader, session_id)
    client = TestClient(create_app(config_loader=test_config_loader))

    # Test SHAP Status (Default/Pending)
    response = client.get(f"/api/runs/{session_id}/evaluation/shap/status")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["session_id"] == session_id
    assert res_data["status"] == "pending"
    assert res_data["progress"] == 0
    assert "Awaiting SHAP" in res_data["message"]

    # Test Overfitting Status (Default/Pending)
    response = client.get(f"/api/runs/{session_id}/evaluation/overfitting/status")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["session_id"] == session_id
    assert res_data["status"] == "pending"
    assert res_data["progress"] == 0
    assert "Awaiting Overfitting" in res_data["message"]

    # Test Judge Status (Default/Pending)
    response = client.get(f"/api/runs/{session_id}/evaluation/judge/status")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["session_id"] == session_id
    assert res_data["status"] == "pending"
    assert res_data["progress"] == 0
    assert "Awaiting Judge" in res_data["message"]


def test_evaluation_status_routes_custom(test_config_loader: ConfigLoader) -> None:
    session_id = "test_eval_status_custom"
    prepare_session_with_statuses(test_config_loader, session_id)
    session_path = test_config_loader.paths.workspace_root / session_id

    # Write custom status files
    shap_data = {"status": "running", "progress": 45, "message": "Computing SHAP values..."}
    overfitting_data = {"status": "complete", "progress": 100, "message": "Overfitting checked."}
    judge_data = {"status": "running", "progress": 60, "message": "Turn 1: Evaluating models...", "logs": ["Step 1", "Step 2"]}

    (session_path / "evaluation" / "shap_status.json").write_text(json.dumps(shap_data), encoding="utf-8")
    (session_path / "evaluation" / "overfitting_status.json").write_text(json.dumps(overfitting_data), encoding="utf-8")
    (session_path / "reports" / "judge_status.json").write_text(json.dumps(judge_data), encoding="utf-8")

    client = TestClient(create_app(config_loader=test_config_loader))

    # Test SHAP Status (Custom)
    response = client.get(f"/api/runs/{session_id}/evaluation/shap/status")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["session_id"] == session_id
    assert res_data["status"] == "running"
    assert res_data["progress"] == 45
    assert res_data["message"] == "Computing SHAP values..."

    # Test Overfitting Status (Custom)
    response = client.get(f"/api/runs/{session_id}/evaluation/overfitting/status")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["session_id"] == session_id
    assert res_data["status"] == "complete"
    assert res_data["progress"] == 100
    assert res_data["message"] == "Overfitting checked."

    # Test Judge Status (Custom)
    response = client.get(f"/api/runs/{session_id}/evaluation/judge/status")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["session_id"] == session_id
    assert res_data["status"] == "running"
    assert res_data["progress"] == 60
    assert res_data["message"] == "Turn 1: Evaluating models..."
    assert res_data["logs"] == ["Step 1", "Step 2"]


def test_default_shap_model_fallback(test_config_loader: ConfigLoader) -> None:
    session_id = "test_default_shap_model_fallback_session"
    prepare_session_with_statuses(test_config_loader, session_id)
    session_path = test_config_loader.paths.workspace_root / session_id

    # Create directories for models
    shap_dir = session_path / "evaluation" / "shap"
    (shap_dir / "ModelA" / "csv").mkdir(parents=True, exist_ok=True)
    (shap_dir / "ModelB" / "csv").mkdir(parents=True, exist_ok=True)

    # Write global_feature_importance.csv only for ModelB
    (shap_dir / "ModelB" / "csv" / "global_feature_importance.csv").write_text(
        "feature_name,mean_absolute_shap_value\nfeat1,0.5\n", encoding="utf-8"
    )

    # Write judge decision selecting ModelA as the winner (which has no SHAP CSV)
    judge_decision = {
        "selected_model": "ModelA",
        "selected_models": ["ModelA", "ModelB"],
        "ranked_models": [{"model_name": "ModelA", "rank": 1}, {"model_name": "ModelB", "rank": 2}],
    }
    (session_path / "reports" / "judge_decision.json").write_text(
        json.dumps(judge_decision), encoding="utf-8"
    )

    # Set SHAP status to completed
    shap_status = {"status": "completed", "progress": 100, "message": "SHAP complete"}
    (session_path / "evaluation" / "shap_status.json").write_text(
        json.dumps(shap_status), encoding="utf-8"
    )

    client = TestClient(create_app(config_loader=test_config_loader))

    # Requesting shap without model_name should fall back to ModelB
    # since ModelA (the winner) lacks SHAP CSV file.
    response = client.get(f"/api/runs/{session_id}/shap")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "complete"
    assert res_data["model_name"] == "ModelB"
    assert len(res_data["features"]) == 1
    assert res_data["features"][0]["feature"] == "feat1"
    assert res_data["features"][0]["importance"] == 0.5

