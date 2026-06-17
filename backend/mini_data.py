from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class UnsupportedDatasetTypeError(ValueError):
    """Raised when an uploaded dataset extension is outside Epic 1 scope."""


@dataclass(frozen=True)
class DatasetSummary:
    row_count: int
    column_count: int
    columns: list[str]
    file_size_bytes: int
    data_type: str
    canonical_data_path: Path
    mini_data_path: Path
    source_path: Path | None = None


class DatasetNormalizer:
    csv_extensions = {".csv"}
    excel_extensions = {".xls", ".xlsx"}

    def __init__(self, mini_data_sample_rows: int, chunk_size_rows: int) -> None:
        self.mini_data_sample_rows = mini_data_sample_rows
        self.chunk_size_rows = chunk_size_rows

    def normalize(self, source_file: Path, data_dir: Path) -> DatasetSummary:
        data_dir.mkdir(parents=True, exist_ok=True)
        file_extension = source_file.suffix.lower()
        canonical_data_path = data_dir / "data.csv"

        if file_extension in self.csv_extensions:
            shutil.copyfile(source_file, canonical_data_path)
            source_path: Path | None = None
            data_type = "csv"
        elif file_extension in self.excel_extensions:
            source_path = data_dir / f"source{file_extension}"
            shutil.copyfile(source_file, source_path)
            self._convert_excel_to_csv(source_file=source_path, csv_path=canonical_data_path)
            data_type = "excel"
        else:
            raise UnsupportedDatasetTypeError(
                f"Unsupported dataset extension: {file_extension}"
            )

        row_count, column_names, sample_frame = self._read_csv_summary(
            csv_path=canonical_data_path
        )
        mini_data_path = data_dir / "mini_data.csv"
        self._write_mini_data(sample_frame=sample_frame, mini_data_path=mini_data_path)

        return DatasetSummary(
            row_count=row_count,
            column_count=len(column_names),
            columns=column_names,
            file_size_bytes=canonical_data_path.stat().st_size,
            data_type=data_type,
            canonical_data_path=canonical_data_path,
            mini_data_path=mini_data_path,
            source_path=source_path,
        )

    @staticmethod
    def _convert_excel_to_csv(source_file: Path, csv_path: Path) -> None:
        excel_frame = pd.read_excel(source_file, sheet_name=0)
        excel_frame.to_csv(csv_path, index=False)

    def _read_csv_summary(self, csv_path: Path) -> tuple[int, list[str], pd.DataFrame]:
        row_count = 0
        column_names: list[str] = []
        sample_chunks: list[pd.DataFrame] = []
        sampled_rows = 0

        for data_chunk in pd.read_csv(csv_path, chunksize=self.chunk_size_rows):
            if not column_names:
                column_names = list(data_chunk.columns)
            row_count += len(data_chunk)

            if sampled_rows < self.mini_data_sample_rows:
                remaining_sample_rows = self.mini_data_sample_rows - sampled_rows
                sample_chunk = data_chunk.head(remaining_sample_rows)
                sample_chunks.append(sample_chunk)
                sampled_rows += len(sample_chunk)

        sample_frame = (
            pd.concat(sample_chunks, ignore_index=True)
            if sample_chunks
            else pd.DataFrame(columns=column_names)
        )
        return row_count, column_names, sample_frame

    @staticmethod
    def _write_mini_data(sample_frame: pd.DataFrame, mini_data_path: Path) -> None:
        mini_data = sample_frame.describe(include="all").transpose()
        mini_data.to_csv(mini_data_path)
