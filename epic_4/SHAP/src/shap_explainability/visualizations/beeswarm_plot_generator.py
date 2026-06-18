"""Beeswarm plot generator for SHAP explainability (spec.md Sec 16.3).

Renders beeswarm_plot.png using shap.summary_plot() with plot_type="violin".
The violin representation shows the density distribution of SHAP values per feature,
complementing the dot-scatter summary plot by revealing value spread and concentration.

matplotlib Agg backend is activated by the parent package __init__.py before
any pyplot import. This module must be imported via the visualizations package
(not directly as a top-level import) to guarantee backend initialization order.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from shap_explainability.errors import VisualizationError
from shap_explainability.models.shap_result import SHAPResult
from shap_explainability.utils.logger import ExecutionLogger

_BEESWARM_PLOT_FILENAME: str = "beeswarm_plot.png"


class BeeswarmPlotGenerator:
    """Renders beeswarm_plot.png from a SHAPResult (spec.md Sec 16.3).

    Uses shap.summary_plot() with plot_type="violin" to show the SHAP value
    density distribution per feature. Supports binary, multiclass, and regression
    prediction types:
    - Binary/Regression: shap_values_array is a 2D ndarray passed directly.
    - Multiclass: shap_values_array is a list of K arrays passed directly;
      SHAP aggregates across classes automatically for the violin representation.
    """

    def __init__(
        self,
        execution_logger: ExecutionLogger,
        plot_format: str,
        max_display_features: int,
    ) -> None:
        """Initializes the generator with session-scoped config.

        Args:
            execution_logger: Session-scoped logger for plot_generation events (spec Sec 19).
            plot_format: Image format for savefig (e.g. "PNG"). From AppConfig.plot_format.
            max_display_features: Maximum number of features to display. From AppConfig.max_display_features.
        """
        self._execution_logger: ExecutionLogger = execution_logger
        self._plot_format: str = plot_format
        self._max_display_features: int = max_display_features

    def render(
        self,
        shap_result: SHAPResult,
        feature_dataframe: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Renders beeswarm_plot.png and writes it to output_path.

        Passes shap_result.shap_values_array directly to shap.summary_plot() with
        plot_type="violin". No type-specific branching is needed; the SHAP library
        handles both 2D ndarray (binary/regression) and list-of-arrays (multiclass).

        Args:
            shap_result: Completed SHAPResult produced by SHAPService.
            feature_dataframe: Cleaned feature DataFrame (target column excluded),
                with columns in the same order as shap_result.feature_names.
            output_path: Destination path, typically from OutputManager.plot_path().

        Returns:
            The resolved output_path that was written.

        Raises:
            VisualizationError: If shap.summary_plot() or plt.savefig() fails.
        """
        self._execution_logger.log_plot_generation(
            f"Starting beeswarm plot generation => {output_path}"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shap.summary_plot(
                shap_result.shap_values_array,
                features=feature_dataframe,
                feature_names=list(shap_result.feature_names),
                plot_type="violin",
                max_display=self._max_display_features,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(
                output_path,
                format=self._plot_format.lower(),
                bbox_inches="tight",
                dpi=150,
            )
        except Exception as plot_error:
            raise VisualizationError(
                f"Failed to generate beeswarm plot at '{output_path}': {plot_error}"
            ) from plot_error
        finally:
            plt.close("all")

        self._execution_logger.log_plot_generation(
            f"Beeswarm plot written => {output_path}"
        )
        return output_path
