"""Unit tests for backend.agents.evaluation.shap.explainers.explainer_factory."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.agents.evaluation.shap.errors import SHAPExecutionError
from backend.agents.evaluation.shap.explainers.explainer_factory import (
    BuiltExplainer,
    ExplainerFactory,
)
from backend.agents.evaluation.shap.tests.fixtures.fixture_factory import FixtureFactory


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _write_full_config(directory: Path) -> Path:
    """Write a model_type_detection.json with all required sections."""
    config_directory = directory / "config"
    config_directory.mkdir(parents=True, exist_ok=True)
    config_path = config_directory / "model_type_detection.json"
    config_data = {
        "class_name_to_model_family": {
            "XGBClassifier": "XGBoost",
            "RandomForestClassifier": "RandomForest",
            "LGBMClassifier": "LightGBM",
            "CatBoostClassifier": "CatBoost",
            "LogisticRegression": "LogisticRegression",
        },
        "supplied_name_to_family": {},
        "model_family_to_explainer": {
            "XGBoost": "TreeExplainer",
            "RandomForest": "TreeExplainer",
            "LightGBM": "TreeExplainer",
            "CatBoost": "TreeExplainer",
            "LogisticRegression": "LinearExplainer",
        },
        "class_name_to_prediction_category": {
            "XGBClassifier": "classification",
            "RandomForestClassifier": "classification",
            "LGBMClassifier": "classification",
            "CatBoostClassifier": "classification",
            "LogisticRegression": "classification",
        },
        "tree_explainer_kwargs_by_family": {
            "XGBoost": {},
            "RandomForest": {},
            "LightGBM": {},
            "CatBoost": {"check_additivity": False},
        },
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    return config_path


def _make_feature_dataframe(num_rows: int = 20, num_features: int = 3) -> pd.DataFrame:
    """Create a synthetic feature DataFrame for tests."""
    return pd.DataFrame(
        np.random.rand(num_rows, num_features),
        columns=[f"feature_{index}" for index in range(num_features)],
    )


def _make_factory(tmp_path: Path) -> ExplainerFactory:
    """Create an ExplainerFactory with a test config and logger."""
    config_path = _write_full_config(tmp_path)
    logger = FixtureFactory.make_execution_logger(tmp_path)
    return ExplainerFactory(
        execution_logger=logger,
        model_type_config_path=config_path,
        linear_background_samples=200,
    )


# ---------------------------------------------------------------------------
# BuiltExplainer dataclass
# ---------------------------------------------------------------------------

def test_built_explainer_is_frozen() -> None:
    """BuiltExplainer must be immutable (frozen dataclass)."""
    built = BuiltExplainer(
        explainer_object=object(),
        explainer_name="TreeExplainer",
        model_family="XGBoost",
    )
    with pytest.raises(Exception):
        built.explainer_name = "changed"  # type: ignore[misc]


def test_built_explainer_fields_are_set() -> None:
    sentinel_explainer = object()
    built = BuiltExplainer(
        explainer_object=sentinel_explainer,
        explainer_name="LinearExplainer",
        model_family="LogisticRegression",
    )
    assert built.explainer_object is sentinel_explainer
    assert built.explainer_name == "LinearExplainer"
    assert built.model_family == "LogisticRegression"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_missing_config_raises_shap_execution_error(tmp_path: Path) -> None:
    """ExplainerFactory init must raise SHAPExecutionError for missing config."""
    logger = FixtureFactory.make_execution_logger(tmp_path)
    with pytest.raises(SHAPExecutionError, match="not found"):
        ExplainerFactory(
            execution_logger=logger,
            model_type_config_path=tmp_path / "nonexistent.json",
        )


def test_config_missing_family_to_explainer_section_raises(tmp_path: Path) -> None:
    """Config without model_family_to_explainer section must raise at init."""
    config_directory = tmp_path / "config"
    config_directory.mkdir()
    config_path = config_directory / "model_type_detection.json"
    config_path.write_text(json.dumps({"class_name_to_model_family": {}}), encoding="utf-8")

    logger = FixtureFactory.make_execution_logger(tmp_path)
    with pytest.raises(SHAPExecutionError, match="missing or empty"):
        ExplainerFactory(execution_logger=logger, model_type_config_path=config_path)


# ---------------------------------------------------------------------------
# TreeExplainer selection -- all four tree families
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_family", ["XGBoost", "RandomForest", "LightGBM", "CatBoost"])
def test_tree_explainer_selected_for_tree_families(
    tmp_path: Path, model_family: str
) -> None:
    """All tree-based families must produce a BuiltExplainer with TreeExplainer name."""
    factory = _make_factory(tmp_path)
    session_context = FixtureFactory.make_session_context()
    feature_dataframe = _make_feature_dataframe()
    mock_model = MagicMock()

    with patch("shap.TreeExplainer") as mock_tree_explainer_class:
        mock_tree_explainer_class.return_value = MagicMock()
        result = factory.create(
            model_family=model_family,
            model_object=mock_model,
            feature_dataframe=feature_dataframe,
            session_context=session_context,
        )

    assert result.explainer_name == "TreeExplainer"
    assert result.model_family == model_family
    mock_tree_explainer_class.assert_called_once_with(mock_model)


# ---------------------------------------------------------------------------
# LinearExplainer selection -- LogisticRegression
# ---------------------------------------------------------------------------

def test_linear_explainer_selected_for_logistic_regression(tmp_path: Path) -> None:
    """LogisticRegression must produce a BuiltExplainer with LinearExplainer name."""
    factory = _make_factory(tmp_path)
    session_context = FixtureFactory.make_session_context()
    feature_dataframe = _make_feature_dataframe(num_rows=10)
    mock_model = MagicMock()

    with (
        patch("shap.maskers.Independent") as mock_masker_class,
        patch("shap.LinearExplainer") as mock_linear_explainer_class,
    ):
        mock_masker_class.return_value = MagicMock()
        mock_linear_explainer_class.return_value = MagicMock()
        result = factory.create(
            model_family="LogisticRegression",
            model_object=mock_model,
            feature_dataframe=feature_dataframe,
            session_context=session_context,
        )

    assert result.explainer_name == "LinearExplainer"
    assert result.model_family == "LogisticRegression"
    mock_linear_explainer_class.assert_called_once()


# ---------------------------------------------------------------------------
# Unsupported family
# ---------------------------------------------------------------------------

def test_unsupported_model_family_raises_shap_execution_error(tmp_path: Path) -> None:
    """Family not in config must raise SHAPExecutionError and mark context failed."""
    factory = _make_factory(tmp_path)
    session_context = FixtureFactory.make_session_context()
    feature_dataframe = _make_feature_dataframe()

    with pytest.raises(SHAPExecutionError):
        factory.create(
            model_family="SomeUnsupportedFamily",
            model_object=MagicMock(),
            feature_dataframe=feature_dataframe,
            session_context=session_context,
        )


def test_unsupported_family_marks_session_context_failed(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path)
    session_context = FixtureFactory.make_session_context()

    with pytest.raises(SHAPExecutionError):
        factory.create(
            model_family="SomeUnsupportedFamily",
            model_object=MagicMock(),
            feature_dataframe=_make_feature_dataframe(),
            session_context=session_context,
        )

    assert session_context.has_failed()


# ---------------------------------------------------------------------------
# SessionContext explainer_name written
# ---------------------------------------------------------------------------

def test_create_writes_explainer_name_to_session_context(tmp_path: Path) -> None:
    """Successful create() must write explainer_name to SessionContext."""
    factory = _make_factory(tmp_path)
    session_context = FixtureFactory.make_session_context()

    with patch("shap.TreeExplainer") as mock_tree:
        mock_tree.return_value = MagicMock()
        factory.create(
            model_family="XGBoost",
            model_object=MagicMock(),
            feature_dataframe=_make_feature_dataframe(),
            session_context=session_context,
        )

    assert session_context.explainer_name == "TreeExplainer"


# ---------------------------------------------------------------------------
# LinearExplainer background sampling
# ---------------------------------------------------------------------------

def test_linear_explainer_subsamples_large_dataframe(tmp_path: Path) -> None:
    """When num_rows > linear_background_samples, a subsample should be used."""
    config_path = _write_full_config(tmp_path)
    logger = FixtureFactory.make_execution_logger(tmp_path)
    factory = ExplainerFactory(
        execution_logger=logger,
        model_type_config_path=config_path,
        linear_background_samples=5,
    )
    # DataFrame with more rows than the cap
    large_feature_dataframe = _make_feature_dataframe(num_rows=50)
    session_context = FixtureFactory.make_session_context()

    captured_masker_input = []

    def capture_masker(background_data: Any) -> MagicMock:
        captured_masker_input.append(background_data)
        return MagicMock()

    with (
        patch("shap.maskers.Independent", side_effect=capture_masker),
        patch("shap.LinearExplainer") as mock_linear,
    ):
        mock_linear.return_value = MagicMock()
        factory.create(
            model_family="LogisticRegression",
            model_object=MagicMock(),
            feature_dataframe=large_feature_dataframe,
            session_context=session_context,
        )

    assert len(captured_masker_input) == 1
    background_passed = captured_masker_input[0]
    assert len(background_passed) == 5


def test_linear_explainer_uses_full_dataframe_when_below_cap(tmp_path: Path) -> None:
    """When num_rows <= linear_background_samples, full DataFrame is used."""
    config_path = _write_full_config(tmp_path)
    logger = FixtureFactory.make_execution_logger(tmp_path)
    factory = ExplainerFactory(
        execution_logger=logger,
        model_type_config_path=config_path,
        linear_background_samples=200,
    )
    small_feature_dataframe = _make_feature_dataframe(num_rows=10)
    session_context = FixtureFactory.make_session_context()

    captured_masker_input = []

    def capture_masker(background_data: Any) -> MagicMock:
        captured_masker_input.append(background_data)
        return MagicMock()

    with (
        patch("shap.maskers.Independent", side_effect=capture_masker),
        patch("shap.LinearExplainer") as mock_linear,
    ):
        mock_linear.return_value = MagicMock()
        factory.create(
            model_family="LogisticRegression",
            model_object=MagicMock(),
            feature_dataframe=small_feature_dataframe,
            session_context=session_context,
        )

    assert len(captured_masker_input[0]) == 10


# ---------------------------------------------------------------------------
# Explainer construction failure propagation
# ---------------------------------------------------------------------------

def test_shap_construction_exception_wrapped_in_shap_execution_error(
    tmp_path: Path,
) -> None:
    """If shap.TreeExplainer raises, SHAPExecutionError must be raised."""
    factory = _make_factory(tmp_path)
    session_context = FixtureFactory.make_session_context()

    with patch("shap.TreeExplainer", side_effect=ValueError("bad model")):
        with pytest.raises(SHAPExecutionError, match="Failed to construct"):
            factory.create(
                model_family="XGBoost",
                model_object=MagicMock(),
                feature_dataframe=_make_feature_dataframe(),
                session_context=session_context,
            )
