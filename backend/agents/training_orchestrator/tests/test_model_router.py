from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.training_orchestrator.contracts import (
    OrchestratorMetadata,
    SelectedModelConfig,
)
from backend.agents.training_orchestrator.errors import (
    InvalidModelConfigError,
    ModelRoutingError,
)
from backend.agents.training_orchestrator.model_router import ModelRouter

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_LIBRARY = REPO_ROOT / "model_library"
FIXTURES = REPO_ROOT / "backend" / "agents" / "model_selection" / "fixtures"


def _selected() -> list[SelectedModelConfig]:
    payload = json.loads((FIXTURES / "expected_iris_model_config.json").read_text())
    return [SelectedModelConfig.model_validate(item) for item in payload[:2]]


def _metadata() -> OrchestratorMetadata:
    return OrchestratorMetadata.model_validate_json(
        (FIXTURES / "iris_metadata.json").read_text()
    )


def test_routes_registry_models_in_priority_order(tmp_path: Path) -> None:
    jobs = ModelRouter(MODEL_LIBRARY).route_all(
        selected_models=list(reversed(_selected())),
        metadata=_metadata(),
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        session_dir=tmp_path / "session",
    )
    assert [job.model_id for job in jobs] == ["model_001", "model_002"]
    assert [job.priority for job in jobs] == [1, 2]
    assert all(job.trainer_type == "tabular_classification" for job in jobs)
    assert jobs[0].parameters["n_estimators"] == 100
    assert jobs[0].source.endswith("MODEL_REGISTRY")


def test_rejects_unknown_model(tmp_path: Path) -> None:
    selected = _selected()[0].model_copy(update={"model_name": "MysteryNet"})
    with pytest.raises(ModelRoutingError, match="not present"):
        ModelRouter(MODEL_LIBRARY).route_all(
            selected_models=[selected],
            metadata=_metadata(),
            train_path=tmp_path / "train.csv",
            test_path=tmp_path / "test.csv",
            session_dir=tmp_path / "session",
        )


def test_rejects_duplicate_models(tmp_path: Path) -> None:
    first = _selected()[0]
    duplicate = first.model_copy(update={"priority": 2})
    with pytest.raises(InvalidModelConfigError, match="duplicate model"):
        ModelRouter(MODEL_LIBRARY).route_all(
            selected_models=[first, duplicate],
            metadata=_metadata(),
            train_path=tmp_path / "train.csv",
            test_path=tmp_path / "test.csv",
            session_dir=tmp_path / "session",
        )


def test_rejects_task_mismatch(tmp_path: Path) -> None:
    selected = _selected()[0].model_copy(update={"task_type": "regression"})
    with pytest.raises(ModelRoutingError, match="model library declares"):
        ModelRouter(MODEL_LIBRARY).route_all(
            selected_models=[selected],
            metadata=_metadata(),
            train_path=tmp_path / "train.csv",
            test_path=tmp_path / "test.csv",
            session_dir=tmp_path / "session",
        )


def test_image_routing_accepts_only_cnn(tmp_path: Path) -> None:
    config = SelectedModelConfig(
        name="PyTorchCNNClassifier",
        model_name="PyTorchCNNClassifier",
        task_type="classification",
        priority=1,
    )
    metadata = OrchestratorMetadata(
        problem_type="classification",
        data_format="image",
        output_cols=["label"],
    )
    jobs = ModelRouter(MODEL_LIBRARY).route_all(
        selected_models=[config],
        metadata=metadata,
        train_path=tmp_path / "train.zip",
        test_path=tmp_path / "test.zip",
        session_dir=tmp_path / "session",
    )
    assert jobs[0].trainer_type == "image_classification"
