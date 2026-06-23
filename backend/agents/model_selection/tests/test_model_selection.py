from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.model_selection.agents import ModelSelectionOrchestratorAgent
from backend.agents.model_selection.catalog import ModelLibraryCatalogAgent
from backend.agents.model_selection.errors import UnsupportedProblemTypeError


REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_LIBRARY = REPO_ROOT / "model_library"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        assert "ALLOWED_MODELS=" in prompt
        return self.response


def test_catalog_is_derived_from_mlkit_registry() -> None:
    catalog = ModelLibraryCatalogAgent(MODEL_LIBRARY).run()
    assert len(catalog) == 60
    assert sum(item.task_type == "classification" for item in catalog.values()) == 30
    assert sum(item.task_type == "regression" for item in catalog.values()) == 30
    assert catalog["XGBClassifier"].default_hyperparameters["n_estimators"] == 100
    assert catalog["XGBRegressor"].task_type == "regression"


def test_deterministic_iris_selection_uses_only_registry(tmp_path: Path) -> None:
    output = tmp_path / "model_config.json"
    report = tmp_path / "report.json"
    agent = ModelSelectionOrchestratorAgent(MODEL_LIBRARY)
    candidates = agent.run(
        metadata_path=FIXTURES / "iris_metadata.json",
        feature_selection_path=FIXTURES / "iris_feature_selection.json",
        mini_data_path=FIXTURES / "iris_mini_data.csv",
        output_path=output,
        report_path=report,
        max_models=5,
    )
    catalog = ModelLibraryCatalogAgent(MODEL_LIBRARY).run()
    assert len(candidates) == 5
    assert [item.priority for item in candidates] == [1, 2, 3, 4, 5]
    assert all(item.model_name in catalog for item in candidates)
    assert all(catalog[item.model_name].task_type == "classification" for item in candidates)
    assert any(item.model_name == "XGBClassifier" for item in candidates)
    assert output.exists()
    assert json.loads(report.read_text())["selection_mode"] == "deterministic"


def test_llm_cannot_escape_model_library(tmp_path: Path) -> None:
    llm = FakeLLM(
        json.dumps(
            [
                {"model_name": "MysteryNet", "rationale": "invented"},
                {"model_name": "XGBClassifier", "rationale": "good tabular model"},
                {"model_name": "XGBRegressor", "rationale": "wrong task"},
            ]
        )
    )
    output = tmp_path / "model_config.json"
    agent = ModelSelectionOrchestratorAgent(MODEL_LIBRARY, llm_client=llm)
    candidates = agent.run(
        metadata_path=FIXTURES / "iris_metadata.json",
        feature_selection_path=FIXTURES / "iris_feature_selection.json",
        mini_data_path=FIXTURES / "iris_mini_data.csv",
        output_path=output,
        max_models=3,
    )
    assert llm.calls == 1
    names = [item.model_name for item in candidates]
    assert "MysteryNet" not in names
    assert "XGBRegressor" not in names
    assert names[0] == "XGBClassifier"
    assert len(names) == 3


def test_image_selection_is_cnn_only(tmp_path: Path) -> None:
    metadata = json.loads((FIXTURES / "iris_metadata.json").read_text())
    metadata["data_format"] = "image"
    metadata["input_cols"] = ["image_path"]
    metadata["col_types"] = {"image_path": "image_path", "species": "categorical"}
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    feature_path = tmp_path / "features.json"
    feature_path.write_text(json.dumps({"keep": ["image_path"], "drop": [], "engineered": []}))

    candidates = ModelSelectionOrchestratorAgent(MODEL_LIBRARY).run(
        metadata_path=metadata_path,
        feature_selection_path=feature_path,
        mini_data_path=None,
        output_path=tmp_path / "model_config.json",
        max_models=5,
    )
    assert [item.model_name for item in candidates] == ["PyTorchCNNClassifier"]


def test_regression_selection_uses_regressors(tmp_path: Path) -> None:
    metadata = {
        "problem_type": "regression",
        "data_format": "tabular",
        "output_cols": ["price"],
        "input_cols": ["size", "rooms", "age"],
        "col_types": {"size": "numeric", "rooms": "numeric", "age": "numeric"},
        "row_count": 5000,
        "col_count": 4,
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    feature_path = tmp_path / "features.json"
    feature_path.write_text(json.dumps({"keep": ["size", "rooms", "age"]}))

    candidates = ModelSelectionOrchestratorAgent(MODEL_LIBRARY).run(
        metadata_path=metadata_path,
        feature_selection_path=feature_path,
        mini_data_path=None,
        output_path=tmp_path / "model_config.json",
        max_models=4,
    )
    catalog = ModelLibraryCatalogAgent(MODEL_LIBRARY).run()
    assert len(candidates) == 4
    assert all(catalog[item.model_name].task_type == "regression" for item in candidates)
    assert any(item.model_name == "XGBRegressor" for item in candidates)


def test_unsupervised_fails_instead_of_inventing_model(tmp_path: Path) -> None:
    metadata = {
        "problem_type": "unsupervised",
        "data_format": "tabular",
        "output_cols": [],
        "input_cols": ["x", "y"],
        "row_count": 100,
        "col_count": 2,
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    feature_path = tmp_path / "features.json"
    feature_path.write_text(json.dumps({"keep": ["x", "y"]}))

    with pytest.raises(UnsupportedProblemTypeError, match="no unsupervised model"):
        ModelSelectionOrchestratorAgent(MODEL_LIBRARY).run(
            metadata_path=metadata_path,
            feature_selection_path=feature_path,
            mini_data_path=None,
            output_path=tmp_path / "model_config.json",
        )
