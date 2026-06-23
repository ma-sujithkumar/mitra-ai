"""Unit tests for backend.agents.evaluation.shap.exporters.metadata_exporter."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backend.agents.evaluation.shap.errors import ExportError
from backend.agents.evaluation.shap.exporters.metadata_exporter import MetadataExporter
from backend.agents.evaluation.shap.models.shap_result import SHAPResult
from backend.agents.evaluation.shap.session_context import ExecutionStatus, ModelNameValidationStatus, SessionContext
from backend.agents.evaluation.shap.tests.fixtures.fixture_factory import FixtureFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exporter(tmp_path: Path) -> MetadataExporter:
    logger = FixtureFactory.make_execution_logger(tmp_path)
    return MetadataExporter(execution_logger=logger)


def _make_populated_session_context(
    session_id: str = "test-session-001",
    supplied_model_name: str = "xgboost",
    detected_model_type: str = "XGBClassifier",
    explainer_name: str = "TreeExplainer",
    num_samples: int = 100,
    num_features: int = 5,
) -> SessionContext:
    """Build a SessionContext that resembles a successful mid-pipeline state."""
    context = SessionContext(
        session_id=session_id,
        supplied_model_name=supplied_model_name,
        pickle_file_path="/test/model.pkl",
        engineered_dataset_path="/test/dataset.csv",
    )
    context.detected_model_type = detected_model_type
    context.model_name_validation_status = ModelNameValidationStatus.MATCH
    context.model_name_validation_message = "Supplied name matches detected type."
    context.explainer_name = explainer_name
    context.num_samples = num_samples
    context.num_features = num_features
    context.mark_success()
    return context


def _make_minimal_shap_result(prediction_type: str = "binary_classification") -> SHAPResult:
    """Build a minimal SHAPResult for testing MetadataExporter."""
    importance_dataframe = pd.DataFrame({
        "feature_name": ["feature_a"],
        "mean_absolute_shap_value": [0.5],
    })
    mapping_dataframe = pd.DataFrame({
        "record_id": [0],
        "feature_name": ["feature_a"],
        "feature_value": [1.0],
        "shap_value": [0.3],
    })
    return SHAPResult(
        prediction_type=prediction_type,
        shap_values_array=np.array([[0.3]]),
        feature_names=("feature_a",),
        class_names=None,
        global_importance_dataframe=importance_dataframe,
        mapping_dataframe=mapping_dataframe,
    )


def _load_metadata(output_path: Path) -> dict:
    with open(output_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

def test_export_creates_metadata_json(tmp_path: Path) -> None:
    """export() must create metadata.json at the given path."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata" / "metadata.json"
    context = _make_populated_session_context()

    exporter.export(context, _make_minimal_shap_result(), output_path)

    assert output_path.exists()
    assert output_path.is_file()


def test_export_returns_output_path(tmp_path: Path) -> None:
    """export() must return the path that was written."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = _make_populated_session_context()

    returned_path = exporter.export(context, _make_minimal_shap_result(), output_path)

    assert returned_path == output_path


def test_export_creates_parent_directory_if_missing(tmp_path: Path) -> None:
    """export() must mkdir -p the parent directory if it does not exist."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "deep" / "nested" / "metadata.json"

    assert not output_path.parent.exists()
    exporter.export(
        _make_populated_session_context(), _make_minimal_shap_result(), output_path
    )
    assert output_path.exists()


def test_export_produces_valid_json(tmp_path: Path) -> None:
    """The written file must be parseable as JSON."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    exporter.export(
        _make_populated_session_context(), _make_minimal_shap_result(), output_path
    )

    loaded = _load_metadata(output_path)
    assert isinstance(loaded, dict)


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

def test_export_session_id_matches(tmp_path: Path) -> None:
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = _make_populated_session_context(session_id="my-session-42")

    exporter.export(context, _make_minimal_shap_result(), output_path)

    loaded = _load_metadata(output_path)
    assert loaded["session_id"] == "my-session-42"


def test_export_provided_model_name_matches(tmp_path: Path) -> None:
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = _make_populated_session_context(supplied_model_name="random_forest")

    exporter.export(context, _make_minimal_shap_result(), output_path)

    loaded = _load_metadata(output_path)
    assert loaded["provided_model_name"] == "random_forest"


def test_export_detected_model_type_matches(tmp_path: Path) -> None:
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = _make_populated_session_context(detected_model_type="RandomForestClassifier")

    exporter.export(context, _make_minimal_shap_result(), output_path)

    loaded = _load_metadata(output_path)
    assert loaded["detected_model_type"] == "RandomForestClassifier"


def test_export_validation_status_is_success(tmp_path: Path) -> None:
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = _make_populated_session_context()

    exporter.export(context, _make_minimal_shap_result(), output_path)

    loaded = _load_metadata(output_path)
    assert loaded["validation_status"] == "success"


def test_export_explainer_name_matches(tmp_path: Path) -> None:
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = _make_populated_session_context(explainer_name="LinearExplainer")

    exporter.export(context, _make_minimal_shap_result(), output_path)

    loaded = _load_metadata(output_path)
    assert loaded["explainer"] == "LinearExplainer"


def test_export_num_samples_and_features_match(tmp_path: Path) -> None:
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = _make_populated_session_context(num_samples=250, num_features=12)

    exporter.export(context, _make_minimal_shap_result(), output_path)

    loaded = _load_metadata(output_path)
    assert loaded["num_samples"] == 250
    assert loaded["num_features"] == 12


def test_export_prediction_type_from_shap_result(tmp_path: Path) -> None:
    """prediction_type must be read from SHAPResult.prediction_type."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    shap_result = _make_minimal_shap_result(prediction_type="multiclass_classification")

    exporter.export(_make_populated_session_context(), shap_result, output_path)

    loaded = _load_metadata(output_path)
    assert loaded["prediction_type"] == "multiclass_classification"


def test_export_prediction_type_null_when_shap_result_is_none(tmp_path: Path) -> None:
    """prediction_type must be null when shap_result=None (early-failure path)."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"

    exporter.export(_make_populated_session_context(), None, output_path)

    loaded = _load_metadata(output_path)
    assert loaded["prediction_type"] is None


def test_export_execution_timestamp_is_iso_parseable(tmp_path: Path) -> None:
    """execution_timestamp must be a parseable ISO-8601 datetime string."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"

    exporter.export(
        _make_populated_session_context(), _make_minimal_shap_result(), output_path
    )

    loaded = _load_metadata(output_path)
    timestamp_string = loaded["execution_timestamp"]
    # Should not raise:
    parsed_timestamp = datetime.fromisoformat(timestamp_string)
    assert isinstance(parsed_timestamp, datetime)


# ---------------------------------------------------------------------------
# Warnings and failure paths
# ---------------------------------------------------------------------------

def test_export_includes_warnings_list(tmp_path: Path) -> None:
    """warnings list from SessionContext must appear in metadata.json."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = _make_populated_session_context()
    context.add_warning("Supplied name differs from detected type.")

    exporter.export(context, _make_minimal_shap_result(), output_path)

    loaded = _load_metadata(output_path)
    assert "Supplied name differs from detected type." in loaded["warnings"]


def test_export_warnings_is_empty_list_on_clean_run(tmp_path: Path) -> None:
    """warnings must be an empty list when no warnings were recorded."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"

    exporter.export(
        _make_populated_session_context(), _make_minimal_shap_result(), output_path
    )

    loaded = _load_metadata(output_path)
    assert loaded["warnings"] == []


def test_export_error_message_present_on_failed_context(tmp_path: Path) -> None:
    """error_message must be set in metadata when pipeline failed."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = FixtureFactory.make_session_context()
    context.mark_failed("Model file not found.")

    exporter.export(context, None, output_path)

    loaded = _load_metadata(output_path)
    assert loaded["validation_status"] == "failed"
    assert loaded["error_message"] == "Model file not found."


def test_export_validation_status_warning_on_mismatch(tmp_path: Path) -> None:
    """validation_status must be 'warning' when model-name mismatch recorded."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    context = _make_populated_session_context()
    # Manually simulate a warning status (SessionContext.add_warning sets it)
    context.warnings.append("Model name mismatch.")
    context.execution_status = ExecutionStatus.WARNING

    exporter.export(context, _make_minimal_shap_result(), output_path)

    loaded = _load_metadata(output_path)
    assert loaded["validation_status"] == "warning"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_export_raises_export_error_on_io_failure(tmp_path: Path) -> None:
    """OSError during JSON write must be wrapped in ExportError."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with patch("builtins.open", side_effect=OSError("permission denied")):
        with pytest.raises(ExportError, match="Failed to write metadata JSON"):
            exporter.export(
                _make_populated_session_context(), _make_minimal_shap_result(), output_path
            )
