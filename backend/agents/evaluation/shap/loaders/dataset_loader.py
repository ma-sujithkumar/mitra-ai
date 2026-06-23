"""Load the engineered dataset CSV produced by Epic 2.

Implements spec.md Section 9 (dataset validation) and Sec 10 (dataset as
authoritative source of feature names and ordering).
DatasetLoader performs only structural validation; target column identification
and schema compatibility belong to SchemaValidator.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.agents.evaluation.shap.errors import DatasetLoadError
from backend.agents.evaluation.shap.utils.logger import ExecutionLogger


@dataclass(frozen=True)
class LoadedDataset:
    """Container for a successfully loaded and structurally validated dataset.

    Attributes:
        dataframe: Full loaded DataFrame including any target column if present.
            Column ordering is preserved exactly as it appears in the CSV (Sec 10).
        column_names: Ordered tuple of all column names as they appear in the CSV
            header. Tuple (not list) so callers cannot accidentally mutate ordering.
        num_rows: Number of data rows (excluding the header).
        num_columns: Total number of columns including any target column.
    """

    dataframe: pd.DataFrame
    column_names: tuple[str, ...]
    num_rows: int
    num_columns: int


class DatasetLoader:
    """Loads and structurally validates the Epic 2 engineered dataset CSV.

    Responsibilities (spec.md Sec 9, architecture.md Section 3 and 4):
      - Validate that the dataset file exists at the given path.
      - Read the CSV with BOM-safe encoding (utf-8-sig) to handle Excel exports.
      - Validate the dataset is non-empty (at least one data row, Sec 9).
      - Validate that at least one column is present (feature count > 0, Sec 9).
      - Preserve original column ordering exactly as supplied by Epic 2 (Sec 10).

    DatasetLoader has no knowledge of target column identification or schema
    compatibility against the model; those responsibilities belong to SchemaValidator.
    """

    def __init__(self, execution_logger: ExecutionLogger) -> None:
        """Initializes the DatasetLoader.

        Args:
            execution_logger: Session-scoped logger for recording Sec 19 events.
        """
        self._execution_logger: ExecutionLogger = execution_logger

    def load(self, dataset_path: str | Path) -> LoadedDataset:
        """Load the engineered dataset CSV and validate its structure.

        Args:
            dataset_path: Path to the engineered dataset CSV (spec.md Sec 4.4).

        Returns:
            LoadedDataset with the full DataFrame and extracted structural metadata.

        Raises:
            DatasetLoadError: If the file does not exist, is empty or completely
                unparseable, contains no data rows, or contains no columns.
        """
        file_path = Path(dataset_path).resolve()

        self._execution_logger.log_dataset_validation(
            f"Validating dataset path: {file_path}"
        )
        self._validate_file_exists(file_path)

        self._execution_logger.log_dataset_validation(
            f"Reading dataset CSV: {file_path}"
        )
        dataframe = self._read_csv(file_path)
        self._validate_dataframe(dataframe, file_path)

        column_names = tuple(str(col) for col in dataframe.columns)

        self._execution_logger.log_dataset_validation(
            f"Dataset loaded successfully: {len(dataframe)} rows, "
            f"{len(column_names)} columns."
        )

        return LoadedDataset(
            dataframe=dataframe,
            column_names=column_names,
            num_rows=len(dataframe),
            num_columns=len(column_names),
        )

    def _validate_file_exists(self, file_path: Path) -> None:
        if not file_path.is_file():
            raise DatasetLoadError(
                f"Engineered dataset file does not exist: {file_path}"
            )

    def _read_csv(self, file_path: Path) -> pd.DataFrame:
        """Read the CSV file with BOM-safe UTF-8 encoding.

        utf-8-sig strips the BOM character automatically when present, which
        prevents column names from being prefixed with the BOM byte sequence
        in datasets exported from Excel (pattern reused from epic_3/training).
        """
        try:
            return pd.read_csv(file_path, encoding="utf-8-sig")
        except pd.errors.EmptyDataError as empty_exc:
            raise DatasetLoadError(
                f"Dataset file is empty or contains no parseable content: {file_path}"
            ) from empty_exc
        except pd.errors.ParserError as parse_exc:
            raise DatasetLoadError(
                f"Dataset file could not be parsed as CSV: {file_path}: {parse_exc}"
            ) from parse_exc
        except UnicodeDecodeError as unicode_exc:
            raise DatasetLoadError(
                f"Dataset file encoding is not valid UTF-8: {file_path}"
            ) from unicode_exc
        except OSError as os_exc:
            raise DatasetLoadError(
                f"Failed to read dataset file: {file_path}: {os_exc}"
            ) from os_exc

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame, file_path: Path) -> None:
        """Validate that the loaded DataFrame is usable for SHAP processing."""
        if len(dataframe.columns) == 0:
            raise DatasetLoadError(
                f"Dataset contains no columns: {file_path}"
            )
        if len(dataframe) == 0:
            raise DatasetLoadError(
                f"Dataset contains no data rows: {file_path}"
            )
