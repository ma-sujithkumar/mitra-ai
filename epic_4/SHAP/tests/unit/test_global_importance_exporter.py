"""Unit tests for shap_explainability.exporters.global_importance_exporter."""

import csv
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from shap_explainability.errors import ExportError
from shap_explainability.exporters.global_importance_exporter import GlobalImportanceExporter
from tests.fixtures.fixture_factory import FixtureFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exporter(tmp_path: Path) -> GlobalImportanceExporter:
    logger = FixtureFactory.make_execution_logger(tmp_path)
    return GlobalImportanceExporter(execution_logger=logger)


def _make_importance_dataframe() -> pd.DataFrame:
    """Return a realistic 3-feature importance DataFrame, sorted descending."""
    return pd.DataFrame({
        "feature_name": ["feature_b", "feature_a", "feature_c"],
        "mean_absolute_shap_value": [0.432, 0.217, 0.154],
    })


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

def test_export_creates_csv_file(tmp_path: Path) -> None:
    """export() must create the CSV file at the given path."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "csv" / "global_feature_importance.csv"

    exporter.export(_make_importance_dataframe(), output_path)

    assert output_path.exists()
    assert output_path.is_file()


def test_export_returns_output_path(tmp_path: Path) -> None:
    """export() must return the path that was written."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "csv" / "global_feature_importance.csv"

    returned_path = exporter.export(_make_importance_dataframe(), output_path)

    assert returned_path == output_path


def test_export_creates_parent_directory_if_missing(tmp_path: Path) -> None:
    """export() must mkdir -p the parent directory if it does not exist."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "nested" / "deep" / "global_feature_importance.csv"

    assert not output_path.parent.exists()
    exporter.export(_make_importance_dataframe(), output_path)
    assert output_path.exists()


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

def test_export_csv_has_correct_columns(tmp_path: Path) -> None:
    """Exported CSV must have exactly feature_name and mean_absolute_shap_value columns."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "global_feature_importance.csv"
    exporter.export(_make_importance_dataframe(), output_path)

    loaded = pd.read_csv(output_path)
    assert list(loaded.columns) == ["feature_name", "mean_absolute_shap_value"]


def test_export_csv_row_count_matches_features(tmp_path: Path) -> None:
    """One row per feature must be written."""
    exporter = _make_exporter(tmp_path)
    importance_dataframe = _make_importance_dataframe()
    output_path = tmp_path / "global_feature_importance.csv"
    exporter.export(importance_dataframe, output_path)

    loaded = pd.read_csv(output_path)
    assert len(loaded) == len(importance_dataframe)


def test_export_csv_preserves_feature_names(tmp_path: Path) -> None:
    """Feature names in the CSV must match those in the input DataFrame exactly."""
    exporter = _make_exporter(tmp_path)
    importance_dataframe = _make_importance_dataframe()
    output_path = tmp_path / "global_feature_importance.csv"
    exporter.export(importance_dataframe, output_path)

    loaded = pd.read_csv(output_path)
    assert list(loaded["feature_name"]) == list(importance_dataframe["feature_name"])


def test_export_csv_preserves_mean_abs_values(tmp_path: Path) -> None:
    """SHAP importance values must survive a CSV round-trip with adequate precision."""
    exporter = _make_exporter(tmp_path)
    importance_dataframe = _make_importance_dataframe()
    output_path = tmp_path / "global_feature_importance.csv"
    exporter.export(importance_dataframe, output_path)

    loaded = pd.read_csv(output_path)
    for original_value, loaded_value in zip(
        importance_dataframe["mean_absolute_shap_value"],
        loaded["mean_absolute_shap_value"],
    ):
        assert abs(original_value - loaded_value) < 1e-12


def test_export_csv_preserves_sort_order(tmp_path: Path) -> None:
    """The exporter must preserve the descending sort order from SHAPService."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "global_feature_importance.csv"
    exporter.export(_make_importance_dataframe(), output_path)

    loaded = pd.read_csv(output_path)
    values = list(loaded["mean_absolute_shap_value"])
    assert values == sorted(values, reverse=True)


def test_export_csv_single_feature(tmp_path: Path) -> None:
    """Export must work correctly with a single-feature DataFrame."""
    exporter = _make_exporter(tmp_path)
    single_feature_dataframe = pd.DataFrame({
        "feature_name": ["only_feature"],
        "mean_absolute_shap_value": [0.999],
    })
    output_path = tmp_path / "global_feature_importance.csv"
    exporter.export(single_feature_dataframe, output_path)

    loaded = pd.read_csv(output_path)
    assert len(loaded) == 1
    assert loaded["feature_name"].iloc[0] == "only_feature"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_export_raises_export_error_on_io_failure(tmp_path: Path) -> None:
    """OSError during write must be wrapped in ExportError."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "global_feature_importance.csv"

    with patch("pandas.DataFrame.to_csv", side_effect=OSError("disk full")):
        with pytest.raises(ExportError, match="Failed to write global feature importance"):
            exporter.export(_make_importance_dataframe(), output_path)
