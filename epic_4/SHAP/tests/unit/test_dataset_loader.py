"""Unit tests for shap_explainability.loaders.dataset_loader."""

import uuid
from pathlib import Path

import pandas as pd
import pytest

from shap_explainability.errors import DatasetLoadError
from shap_explainability.loaders.dataset_loader import DatasetLoader, LoadedDataset
from shap_explainability.utils.logger import ExecutionLogger

# ---------------------------------------------------------------------------
# CSV content helpers
# ---------------------------------------------------------------------------

_SIMPLE_CSV_CONTENT: str = (
    "feature_alpha,feature_beta,target\n"
    "1.0,2.0,0\n"
    "3.0,4.0,1\n"
    "5.0,6.0,0\n"
    "7.0,8.0,1\n"
    "9.0,10.0,0\n"
)

_HEADER_ONLY_CSV_CONTENT: str = "feature_alpha,feature_beta,target\n"

# UTF-8 BOM prefix followed by valid CSV content
_BOM_ENCODED_CSV_CONTENT: bytes = (
    b"\xef\xbb\xbf"  # UTF-8 BOM
    b"col_a,col_b\n"
    b"1,2\n"
    b"3,4\n"
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_test_logger(tmp_path: Path) -> ExecutionLogger:
    """Creates an ExecutionLogger writing to a unique temp log file."""
    return ExecutionLogger(
        session_id=f"test-{uuid.uuid4().hex}",
        log_file_path=tmp_path / "logs" / "execution.log",
    )


def _write_csv(directory: Path, content: str, filename: str = "dataset.csv") -> Path:
    csv_path = directory / filename
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# Happy path: structure of the returned LoadedDataset
# ---------------------------------------------------------------------------

def test_load_valid_csv_returns_loaded_dataset_instance(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _SIMPLE_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    result = loader.load(csv_path)

    assert isinstance(result, LoadedDataset)


def test_column_names_preserve_original_ordering(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _SIMPLE_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    result = loader.load(csv_path)

    assert result.column_names == ("feature_alpha", "feature_beta", "target")


def test_num_rows_matches_data_row_count(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _SIMPLE_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    result = loader.load(csv_path)

    assert result.num_rows == 5


def test_num_columns_matches_header_column_count(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _SIMPLE_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    result = loader.load(csv_path)

    assert result.num_columns == 3


def test_column_names_is_a_tuple_not_a_list(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _SIMPLE_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    result = loader.load(csv_path)

    assert isinstance(result.column_names, tuple)


def test_dataframe_column_ordering_matches_column_names(tmp_path: Path) -> None:
    """The dataframe's columns must be in the same order as column_names."""
    csv_path = _write_csv(tmp_path, _SIMPLE_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    result = loader.load(csv_path)

    assert tuple(result.dataframe.columns) == result.column_names


def test_dataframe_row_count_matches_num_rows(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _SIMPLE_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    result = loader.load(csv_path)

    assert len(result.dataframe) == result.num_rows


def test_dataframe_values_are_preserved(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, _SIMPLE_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    result = loader.load(csv_path)

    assert result.dataframe["feature_alpha"].iloc[0] == 1.0
    assert result.dataframe["feature_beta"].iloc[0] == 2.0


# ---------------------------------------------------------------------------
# BOM encoding: Excel-exported CSV compatibility
# ---------------------------------------------------------------------------

def test_bom_encoded_csv_column_names_contain_no_bom_character(tmp_path: Path) -> None:
    """utf-8-sig encoding strips the BOM so column names do not start with the BOM byte."""
    bom_csv_path = tmp_path / "bom_dataset.csv"
    bom_csv_path.write_bytes(_BOM_ENCODED_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    result = loader.load(bom_csv_path)

    assert result.column_names[0] == "col_a"
    assert not result.column_names[0].startswith("﻿")


# ---------------------------------------------------------------------------
# Error cases: file not found
# ---------------------------------------------------------------------------

def test_missing_file_raises_dataset_load_error(tmp_path: Path) -> None:
    nonexistent_path = tmp_path / "does_not_exist.csv"
    loader = DatasetLoader(_make_test_logger(tmp_path))

    with pytest.raises(DatasetLoadError, match="does not exist"):
        loader.load(nonexistent_path)


# ---------------------------------------------------------------------------
# Error cases: structural validation failures
# ---------------------------------------------------------------------------

def test_csv_with_header_only_and_no_data_rows_raises_dataset_load_error(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path, _HEADER_ONLY_CSV_CONTENT)
    loader = DatasetLoader(_make_test_logger(tmp_path))

    with pytest.raises(DatasetLoadError, match="no data rows"):
        loader.load(csv_path)


def test_completely_empty_file_raises_dataset_load_error(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.csv"
    empty_path.write_bytes(b"")
    loader = DatasetLoader(_make_test_logger(tmp_path))

    with pytest.raises(DatasetLoadError):
        loader.load(empty_path)


def test_binary_file_content_raises_dataset_load_error(tmp_path: Path) -> None:
    """A file containing non-UTF-8 bytes cannot be parsed and raises DatasetLoadError."""
    binary_path = tmp_path / "binary.csv"
    # Bytes that are not valid UTF-8 sequences
    binary_path.write_bytes(b"\xff\xfe\x00\x01\x02\x03\x04\x05")
    loader = DatasetLoader(_make_test_logger(tmp_path))

    with pytest.raises(DatasetLoadError):
        loader.load(binary_path)
