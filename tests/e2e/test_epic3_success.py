from __future__ import annotations

import json
from pathlib import Path

from backend.schemas.training import TrainingStartRequest
from backend.services.training_service import TrainingRunConflictError
from backend.services.training_service import TrainingService
from backend.session import SessionManager
from backend.orchestration.events import TrainingEventBus
from backend.agents.training.contracts import TrainingResult
from backend.agents.training_orchestrator import TrainingOrchestrator

from .helpers import build_training_client
from .helpers import prepare_session
from .helpers import wait_for_terminal


class DeterministicClassificationWorker:
    def run(self, job) -> TrainingResult:
        model_path = Path(job.output_dir) / "model.pkl"
        model_path.write_bytes(b"deterministic-classification-model")
        validation_score = 0.94 if job.model_id == "model_001" else 0.91
        return TrainingResult(
            model_id=job.model_id,
            model_name=job.model_name,
            status="completed",
            metrics={
                "task_type": "classification",
                "primary_metric": "accuracy",
                "train_score": validation_score + 0.02,
                "validation_score": validation_score,
                "train": {"accuracy": validation_score + 0.02},
                "validation": {"accuracy": validation_score},
            },
            model_path=str(model_path),
            training_time_sec=0.01,
            error=None,
        )


class DeterministicRegressionWorker:
    def run(self, job) -> TrainingResult:
        model_path = Path(job.output_dir) / "model.pkl"
        model_path.write_bytes(b"deterministic-regression-model")
        return TrainingResult(
            model_id=job.model_id,
            model_name=job.model_name,
            status="completed",
            metrics={
                "task_type": "regression",
                "primary_metric": "r2",
                "train_score": 0.85,
                "validation_score": 0.80,
                "train": {"r2": 0.85},
                "validation": {"r2": 0.80},
            },
            model_path=str(model_path),
            training_time_sec=0.01,
            error=None,
        )


class DeterministicTrainingOrchestrator(TrainingOrchestrator):
    def __init__(self, *args, worker, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.worker = worker

    def prepare_and_execute_local(self, **kwargs):
        return super().prepare_and_execute_local(
            **kwargs,
            worker=self.worker,
        )


def test_classification_api_to_summary_preserves_ids_metrics_and_artifacts(
    e2e_config_loader,
) -> None:
    session_id = "e2e_classification"
    session_path = prepare_session(
        e2e_config_loader,
        session_id=session_id,
        metadata={
            "problem_type": "classification",
            "data_format": "tabular",
            "output_cols": ["species"],
        },
        model_config=[
            {
                "name": "DecisionTreeClassifier",
                "model_name": "DecisionTreeClassifier",
                "task_type": "classification",
                "priority": 1,
                "rationale": "Fast tree baseline",
            },
            {
                "name": "GaussianNB",
                "model_name": "GaussianNB",
                "task_type": "classification",
                "priority": 2,
                "rationale": "Probabilistic baseline",
            },
        ],
        train_header=["feature_a", "feature_b", "species"],
        train_rows=[[1.0, 2.0, 0], [2.0, 1.0, 1], [3.0, 4.0, 1]],
        test_rows=[[1.5, 2.5, 0], [2.5, 3.5, 1]],
    )
    client, service, _ = build_training_client(
        e2e_config_loader,
        orchestrator_factory=lambda model_root, event_sink: DeterministicTrainingOrchestrator(
            model_root,
            event_sink=event_sink,
            worker=DeterministicClassificationWorker(),
        ),
    )

    response = client.post(
        "/api/training/start",
        json={
            "session_id": session_id,
            "target_column": "species",
            "execution_mode": "local",
        },
    )
    assert response.status_code == 202
    final_state = wait_for_terminal(client, session_id)

    assert final_state["status"] == "completed"
    assert final_state["total_models"] == 2
    assert final_state["completed_models"] == 2
    assert final_state["failed_models"] == 0
    assert final_state["job_status_counts"] == {"completed": 2}

    output_dir = session_path / "training"
    manifest = json.loads((output_dir / "training_jobs.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    persisted_run = json.loads((output_dir / "training_run.json").read_text(encoding="utf-8"))

    manifest_ids = [job["model_id"] for job in manifest["jobs"]]
    summary_ids = [model["model_id"] for model in summary["models"]]
    status_ids = [model["model_id"] for model in persisted_run["model_states"]]
    assert manifest_ids == summary_ids == status_ids == ["model_001", "model_002"]
    assert [job["status"] for job in manifest["jobs"]] == ["completed", "completed"]

    for model in summary["models"]:
        assert model["metrics"]["primary_metric"] == "accuracy"
        assert 0.0 <= model["validation_score"] <= 1.0
        assert Path(model["model_path"]).is_file()
        state = next(
            item
            for item in persisted_run["model_states"]
            if item["model_id"] == model["model_id"]
        )
        assert state["validation_score"] == model["validation_score"]
        assert state["model_path"] == model["model_path"]

    restarted_service = TrainingService(
        config_loader=e2e_config_loader,
        session_manager=SessionManager(e2e_config_loader.paths.workspace_root),
        event_bus=TrainingEventBus(),
    )
    restored = restarted_service.get_status(session_id)
    assert restored.status == "completed"
    assert [item.model_id for item in restored.model_states] == manifest_ids
    try:
        restarted_service.start(
            TrainingStartRequest(
                session_id=session_id,
                target_column="species",
                execution_mode="local",
            )
        )
    except TrainingRunConflictError:
        pass
    else:
        raise AssertionError("persisted runs must reject duplicate restart attempts")

    restarted_service.shutdown()
    client.close()
    service.shutdown()


def test_regression_local_mode_reaches_training_summary(e2e_config_loader) -> None:
    session_id = "e2e_regression"
    session_path = prepare_session(
        e2e_config_loader,
        session_id=session_id,
        metadata={
            "problem_type": "regression",
            "data_format": "tabular",
            "output_cols": ["target"],
        },
        model_config=[
            {
                "name": "DecisionTreeRegressor",
                "model_name": "DecisionTreeRegressor",
                "task_type": "regression",
                "priority": 1,
                "rationale": "Tree regression baseline",
            },
            {
                "name": "DummyRegressor",
                "model_name": "DummyRegressor",
                "task_type": "regression",
                "priority": 2,
                "rationale": "Constant regression baseline",
            },
        ],
        train_header=["feature_a", "feature_b", "target"],
        train_rows=[[1.0, 2.0, 10.0], [2.0, 1.0, 12.0], [3.0, 4.0, 20.0]],
        test_rows=[[1.5, 2.5, 11.0], [2.5, 3.5, 18.0]],
    )
    client, service, _ = build_training_client(
        e2e_config_loader,
        orchestrator_factory=lambda model_root, event_sink: DeterministicTrainingOrchestrator(
            model_root,
            event_sink=event_sink,
            worker=DeterministicRegressionWorker(),
        ),
    )

    response = client.post(
        "/api/training/start",
        json={
            "session_id": session_id,
            "target_column": "target",
            "execution_mode": "local",
        },
    )
    assert response.status_code == 202
    final_state = wait_for_terminal(client, session_id)
    assert final_state["status"] == "completed"
    assert final_state["job_status_counts"] == {"completed": 2}

    summary = json.loads(
        (session_path / "training" / "training_summary.json").read_text(encoding="utf-8")
    )
    assert summary["completed"] == 2
    assert all(model["metrics"]["primary_metric"] == "r2" for model in summary["models"])
    assert all(Path(model["model_path"]).is_file() for model in summary["models"])
    client.close()
    service.shutdown()
