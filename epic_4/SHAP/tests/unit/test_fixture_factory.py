"""Unit tests for tests.fixtures.fixture_factory."""

from pathlib import Path

import pandas as pd

from shap_explainability.loaders.dataset_loader import LoadedDataset
from shap_explainability.loaders.model_loader import LoadedModel
from shap_explainability.session_context import ExecutionStatus, SessionContext
from shap_explainability.utils.logger import ExecutionLogger
from tests.fixtures.fixture_factory import FixtureFactory


def test_make_execution_logger_returns_valid_logger(tmp_path: Path) -> None:
    """make_execution_logger returns an ExecutionLogger that can write to disk."""
    logger = FixtureFactory.make_execution_logger(tmp_path)

    assert isinstance(logger, ExecutionLogger)
    # Confirm logging does not raise.
    logger.log_execution_start("test start")
    assert (tmp_path / "logs" / "execution.log").exists()


def test_make_session_context_returns_running_context(tmp_path: Path) -> None:
    """make_session_context returns a SessionContext in RUNNING status."""
    session_context = FixtureFactory.make_session_context(supplied_model_name="catboost")

    assert isinstance(session_context, SessionContext)
    assert session_context.execution_status == ExecutionStatus.RUNNING
    assert session_context.supplied_model_name == "catboost"
    assert not session_context.has_failed()


def test_make_loaded_model_returns_valid_loaded_model(tmp_path: Path) -> None:
    """make_loaded_model returns a frozen LoadedModel with expected field values."""
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier",
        model_family="XGBoost",
        feature_names_from_model=("feature_a", "feature_b"),
        num_features_from_model=2,
    )

    assert isinstance(loaded_model, LoadedModel)
    assert loaded_model.detected_class_name == "XGBClassifier"
    assert loaded_model.model_family == "XGBoost"
    assert loaded_model.feature_names_from_model == ("feature_a", "feature_b")
    assert loaded_model.num_features_from_model == 2


def test_make_loaded_dataset_with_target_column(tmp_path: Path) -> None:
    """make_loaded_dataset includes target column when include_target_column=True."""
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        num_rows=5,
        include_target_column=True,
        target_column_name="target",
    )

    assert isinstance(loaded_dataset, LoadedDataset)
    assert "target" in loaded_dataset.column_names
    assert loaded_dataset.num_rows == 5
    assert loaded_dataset.num_columns == 3  # 2 features + 1 target


def test_make_loaded_dataset_without_target_column(tmp_path: Path) -> None:
    """make_loaded_dataset excludes target column when include_target_column=False."""
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        num_rows=5,
        include_target_column=False,
    )

    assert isinstance(loaded_dataset.dataframe, pd.DataFrame)
    assert "target" not in loaded_dataset.column_names
    assert loaded_dataset.num_columns == 2
