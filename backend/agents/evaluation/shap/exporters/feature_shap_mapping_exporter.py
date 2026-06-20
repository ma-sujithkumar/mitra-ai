"""Writes feature_shap_mapping.csv from the SHAPResult mapping DataFrame.

Implements spec.md Section 17.2. The schema differs by prediction type:
- Binary/Regression: record_id, feature_name, feature_value, shap_value
- Multiclass: record_id, class_name, feature_name, feature_value, shap_value

The mapping DataFrame is already built in the correct long-form schema by
SHAPService; this module is pure I/O with no re-computation.
"""

from pathlib import Path

import pandas as pd

from backend.agents.evaluation.shap.errors import ExportError
from backend.agents.evaluation.shap.utils.logger import ExecutionLogger

_FEATURE_SHAP_MAPPING_FILENAME: str = "feature_shap_mapping.csv"
_FLOAT_FORMAT: str = "%.17g"


class FeatureSHAPMappingExporter:
    """Exports the long-form feature-SHAP mapping table to CSV (spec.md Sec 17.2).

    Responsibilities:
      - Accept the pre-computed mapping_dataframe from SHAPResult.
      - Write it verbatim to the specified path.
      - Guarantee the parent directory exists before writing.
      - Log the CSV-export event via ExecutionLogger.

    The exporter is prediction-type-agnostic: it writes whatever DataFrame it
    receives. The correct schema (4 or 5 columns) is already set by SHAPService.
    """

    def __init__(self, execution_logger: ExecutionLogger) -> None:
        """Initializes the exporter with a session-scoped logger.

        Args:
            execution_logger: Session-scoped logger for recording the csv_export
                event (spec.md Sec 19).
        """
        self._execution_logger: ExecutionLogger = execution_logger

    def export(
        self,
        mapping_dataframe: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Write feature_shap_mapping.csv to the specified path.

        Args:
            mapping_dataframe: Pre-built long-form mapping table from
                SHAPResult.mapping_dataframe.
                Binary/Regression columns: record_id, feature_name,
                    feature_value, shap_value.
                Multiclass columns: record_id, class_name, feature_name,
                    feature_value, shap_value.
            output_path: Full path where the CSV file will be written.
                Typically obtained from OutputManager.csv_path(
                "feature_shap_mapping.csv").

        Returns:
            The resolved output_path that was written.

        Raises:
            ExportError: If the CSV file cannot be created or written.
        """
        num_rows: int = len(mapping_dataframe)
        num_columns: int = len(mapping_dataframe.columns)

        self._execution_logger.log_csv_export(
            f"Starting feature SHAP mapping export: "
            f"{num_rows} rows, {num_columns} columns => {output_path}"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            mapping_dataframe.to_csv(
                output_path,
                index=False,
                float_format=_FLOAT_FORMAT,
            )
        except OSError as io_error:
            raise ExportError(
                f"Failed to write feature SHAP mapping CSV to "
                f"'{output_path}': {io_error}"
            ) from io_error

        self._execution_logger.log_csv_export(
            f"Feature SHAP mapping CSV written: {output_path}"
        )
        return output_path
