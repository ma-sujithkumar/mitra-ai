"""Unit tests for shap_explainability.exporters.feature_shap_mapping_exporter."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from shap_explainability.errors import ExportError
from shap_explainability.exporters.feature_shap_mapping_exporter import FeatureSHAPMappingExporter
from tests.fixtures.fixture_factory import FixtureFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exporter(tmp_path: Path) -> FeatureSHAPMappingExporter:
    logger = FixtureFactory.make_execution_logger(tmp_path)
    return FeatureSHAPMappingExporter(execution_logger=logger)


def _make_binary_mapping_dataframe(
    num_samples: int = 4,
    num_features: int = 3,
) -> pd.DataFrame:
    """Build a realistic binary/regression mapping DataFrame."""
    record_ids = np.repeat(np.arange(num_samples), num_features)
    feature_names_col = np.tile(
        [f"feature_{idx}" for idx in range(num_features)], num_samples
    )
    feature_values = np.random.rand(num_samples * num_features)
    shap_values = np.random.randn(num_samples * num_features)
    return pd.DataFrame({
        "record_id": record_ids,
        "feature_name": feature_names_col,
        "feature_value": feature_values,
        "shap_value": shap_values,
    })


def _make_multiclass_mapping_dataframe(
    num_samples: int = 3,
    num_features: int = 2,
    num_classes: int = 3,
) -> pd.DataFrame:
    """Build a realistic multiclass mapping DataFrame."""
    class_frames = []
    for class_index in range(num_classes):
        record_ids = np.repeat(np.arange(num_samples), num_features)
        class_labels = np.full(num_samples * num_features, f"class_{class_index}")
        feature_names_col = np.tile(
            [f"feature_{idx}" for idx in range(num_features)], num_samples
        )
        feature_values = np.random.rand(num_samples * num_features)
        shap_values = np.random.randn(num_samples * num_features)
        class_frames.append(pd.DataFrame({
            "record_id": record_ids,
            "class_name": class_labels,
            "feature_name": feature_names_col,
            "feature_value": feature_values,
            "shap_value": shap_values,
        }))
    return pd.concat(class_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

def test_export_creates_csv_file(tmp_path: Path) -> None:
    """export() must create the CSV file at the given path."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "csv" / "feature_shap_mapping.csv"

    exporter.export(_make_binary_mapping_dataframe(), output_path)

    assert output_path.exists()
    assert output_path.is_file()


def test_export_returns_output_path(tmp_path: Path) -> None:
    """export() must return the path that was written."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "feature_shap_mapping.csv"

    returned_path = exporter.export(_make_binary_mapping_dataframe(), output_path)

    assert returned_path == output_path


def test_export_creates_parent_directory_if_missing(tmp_path: Path) -> None:
    """export() must mkdir -p the parent directory if it does not exist."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "deep" / "nested" / "feature_shap_mapping.csv"

    assert not output_path.parent.exists()
    exporter.export(_make_binary_mapping_dataframe(), output_path)
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Binary / Regression schema
# ---------------------------------------------------------------------------

def test_binary_csv_has_four_columns(tmp_path: Path) -> None:
    """Binary/Regression export must produce exactly 4 columns."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "feature_shap_mapping.csv"
    exporter.export(_make_binary_mapping_dataframe(), output_path)

    loaded = pd.read_csv(output_path)
    assert list(loaded.columns) == ["record_id", "feature_name", "feature_value", "shap_value"]


def test_binary_csv_row_count(tmp_path: Path) -> None:
    """Binary CSV row count must equal n_samples * n_features."""
    exporter = _make_exporter(tmp_path)
    num_samples, num_features = 5, 4
    mapping_dataframe = _make_binary_mapping_dataframe(num_samples, num_features)
    output_path = tmp_path / "feature_shap_mapping.csv"
    exporter.export(mapping_dataframe, output_path)

    loaded = pd.read_csv(output_path)
    assert len(loaded) == num_samples * num_features


def test_binary_csv_record_ids_are_sequential(tmp_path: Path) -> None:
    """record_id values must follow sequential integer ordering (0..n_samples-1)."""
    exporter = _make_exporter(tmp_path)
    num_samples, num_features = 3, 2
    mapping_dataframe = _make_binary_mapping_dataframe(num_samples, num_features)
    output_path = tmp_path / "feature_shap_mapping.csv"
    exporter.export(mapping_dataframe, output_path)

    loaded = pd.read_csv(output_path)
    unique_record_ids = sorted(loaded["record_id"].unique().tolist())
    assert unique_record_ids == list(range(num_samples))


def test_binary_csv_shap_values_preserved(tmp_path: Path) -> None:
    """SHAP values must survive a CSV round-trip with adequate precision."""
    exporter = _make_exporter(tmp_path)
    mapping_dataframe = _make_binary_mapping_dataframe(num_samples=2, num_features=2)
    output_path = tmp_path / "feature_shap_mapping.csv"
    exporter.export(mapping_dataframe, output_path)

    loaded = pd.read_csv(output_path)
    original_shap = mapping_dataframe["shap_value"].values
    loaded_shap = loaded["shap_value"].values
    assert np.allclose(original_shap, loaded_shap, atol=1e-12)


# ---------------------------------------------------------------------------
# Multiclass schema
# ---------------------------------------------------------------------------

def test_multiclass_csv_has_five_columns(tmp_path: Path) -> None:
    """Multiclass export must produce exactly 5 columns including class_name."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "feature_shap_mapping.csv"
    exporter.export(_make_multiclass_mapping_dataframe(), output_path)

    loaded = pd.read_csv(output_path)
    assert list(loaded.columns) == [
        "record_id", "class_name", "feature_name", "feature_value", "shap_value"
    ]


def test_multiclass_csv_row_count(tmp_path: Path) -> None:
    """Multiclass CSV row count must equal n_samples * n_features * n_classes."""
    exporter = _make_exporter(tmp_path)
    num_samples, num_features, num_classes = 4, 3, 3
    mapping_dataframe = _make_multiclass_mapping_dataframe(num_samples, num_features, num_classes)
    output_path = tmp_path / "feature_shap_mapping.csv"
    exporter.export(mapping_dataframe, output_path)

    loaded = pd.read_csv(output_path)
    assert len(loaded) == num_samples * num_features * num_classes


def test_multiclass_csv_class_names_preserved(tmp_path: Path) -> None:
    """class_name column values must match the input DataFrame."""
    exporter = _make_exporter(tmp_path)
    mapping_dataframe = _make_multiclass_mapping_dataframe(
        num_samples=2, num_features=2, num_classes=2
    )
    output_path = tmp_path / "feature_shap_mapping.csv"
    exporter.export(mapping_dataframe, output_path)

    loaded = pd.read_csv(output_path)
    expected_class_names = set(mapping_dataframe["class_name"].unique())
    actual_class_names = set(loaded["class_name"].unique())
    assert actual_class_names == expected_class_names


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_export_raises_export_error_on_io_failure(tmp_path: Path) -> None:
    """OSError during write must be wrapped in ExportError."""
    exporter = _make_exporter(tmp_path)
    output_path = tmp_path / "feature_shap_mapping.csv"

    with patch("pandas.DataFrame.to_csv", side_effect=OSError("disk full")):
        with pytest.raises(ExportError, match="Failed to write feature SHAP mapping"):
            exporter.export(_make_binary_mapping_dataframe(), output_path)
