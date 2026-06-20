"""Writes global_feature_importance.csv from the SHAPResult importance DataFrame.

Implements spec.md Section 17.1. The input DataFrame is already computed and
sorted by SHAPService; this module is pure I/O -- no aggregation logic here.
"""

from pathlib import Path

import pandas as pd

from backend.agents.evaluation.shap.errors import ExportError
from backend.agents.evaluation.shap.utils.logger import ExecutionLogger

_GLOBAL_IMPORTANCE_FILENAME: str = "global_feature_importance.csv"
_FLOAT_FORMAT: str = "%.17g"


class GlobalImportanceExporter:
    """Exports per-feature mean absolute SHAP importance to CSV (spec.md Sec 17.1).

    Responsibilities:
      - Accept the pre-computed global_importance_dataframe from SHAPResult.
      - Write it verbatim to the specified path.
      - Guarantee the parent directory exists before writing.
      - Log the CSV-export event via ExecutionLogger.
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
        global_importance_dataframe: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Write global_feature_importance.csv to the specified path.

        The DataFrame must already have columns feature_name and
        mean_absolute_shap_value in descending sort order (guaranteed by
        SHAPService._compute_global_importance).

        Args:
            global_importance_dataframe: Pre-computed importance table from
                SHAPResult.global_importance_dataframe. Columns:
                feature_name, mean_absolute_shap_value.
            output_path: Full path where the CSV file will be written.
                Typically obtained from OutputManager.csv_path(
                "global_feature_importance.csv").

        Returns:
            The resolved output_path that was written.

        Raises:
            ExportError: If the CSV file cannot be created or written.
        """
        self._execution_logger.log_csv_export(
            f"Starting global feature importance export: "
            f"{len(global_importance_dataframe)} features => {output_path}"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            global_importance_dataframe.to_csv(
                output_path,
                index=False,
                float_format=_FLOAT_FORMAT,
            )
        except OSError as io_error:
            raise ExportError(
                f"Failed to write global feature importance CSV to "
                f"'{output_path}': {io_error}"
            ) from io_error

        self._execution_logger.log_csv_export(
            f"Global feature importance CSV written: {output_path}"
        )
        return output_path
