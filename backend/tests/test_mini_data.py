from pathlib import Path

import pandas as pd
import pytest

from backend.mini_data import DatasetNormalizer, UnsupportedDatasetTypeError


def test_csv_upload_creates_canonical_data_and_mini_data(tmp_path: Path) -> None:
    source_file = tmp_path / "iris.csv"
    source_file.write_text(
        "a,b,target\n1,2,x\n3,4,y\n",
        encoding="utf-8",
    )
    session_data_dir = tmp_path / "session" / "data"
    session_data_dir.mkdir(parents=True)

    normalizer = DatasetNormalizer(mini_data_sample_rows=1000, chunk_size_rows=50000)
    summary = normalizer.normalize(source_file=source_file, data_dir=session_data_dir)

    assert (session_data_dir / "data.csv").is_file()
    assert (session_data_dir / "mini_data.csv").is_file()
    assert summary.row_count == 2
    assert summary.column_count == 3
    assert summary.data_type == "csv"
    assert summary.columns == ["a", "b", "target"]


def test_excel_upload_preserves_source_and_creates_csv(tmp_path: Path) -> None:
    source_file = tmp_path / "data.xlsx"
    pd.DataFrame({"a": [1, 2], "target": ["x", "y"]}).to_excel(
        source_file,
        index=False,
    )
    session_data_dir = tmp_path / "session" / "data"
    session_data_dir.mkdir(parents=True)

    normalizer = DatasetNormalizer(mini_data_sample_rows=1000, chunk_size_rows=50000)
    summary = normalizer.normalize(source_file=source_file, data_dir=session_data_dir)

    assert (session_data_dir / "source.xlsx").is_file()
    assert (session_data_dir / "data.csv").is_file()
    assert (session_data_dir / "mini_data.csv").is_file()
    assert summary.row_count == 2
    assert summary.column_count == 2
    assert summary.data_type == "excel"


def test_mini_data_contains_transposed_describe_output(tmp_path: Path) -> None:
    source_file = tmp_path / "data.csv"
    source_file.write_text(
        "feature,target\n1,a\n2,b\n3,c\n",
        encoding="utf-8",
    )
    session_data_dir = tmp_path / "session" / "data"
    session_data_dir.mkdir(parents=True)

    normalizer = DatasetNormalizer(mini_data_sample_rows=1000, chunk_size_rows=2)
    normalizer.normalize(source_file=source_file, data_dir=session_data_dir)

    mini_data = pd.read_csv(session_data_dir / "mini_data.csv", index_col=0)
    assert "feature" in mini_data.index
    assert "count" in mini_data.columns


def test_unsupported_upload_extension_fails(tmp_path: Path) -> None:
    source_file = tmp_path / "images.zip"
    source_file.write_bytes(b"not supported")
    session_data_dir = tmp_path / "session" / "data"
    session_data_dir.mkdir(parents=True)

    normalizer = DatasetNormalizer(mini_data_sample_rows=1000, chunk_size_rows=50000)

    with pytest.raises(UnsupportedDatasetTypeError):
        normalizer.normalize(source_file=source_file, data_dir=session_data_dir)
