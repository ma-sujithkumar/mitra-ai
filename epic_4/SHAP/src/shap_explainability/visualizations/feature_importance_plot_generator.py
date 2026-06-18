"""Feature importance bar chart generator for SHAP explainability (spec.md Sec 16.2).

Renders feature_importance_bar.png using shap.summary_plot() with plot_type="bar".
Each feature is shown as a horizontal bar sized by mean(|SHAP value|) across all
samples, ranked descending. For multiclass, SHAP renders stacked bars per class.

matplotlib Agg backend is activated by the parent package __init__.py before
any pyplot import. This module must be imported via the visualizations package
(not directly as a top-level import) to guarantee backend initialization order.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import shap

from shap_explainability.errors import VisualizationError
from shap_explainability.models.shap_result import SHAPResult
from shap_explainability.utils.logger import ExecutionLogger

_FEATURE_IMPORTANCE_BAR_FILENAME: str = "feature_importance_bar.png"


class FeatureImportancePlotGenerator:
    """Renders feature_importance_bar.png from a SHAPResult (spec.md Sec 16.2).

    Uses shap.summary_plot() with plot_type="bar" to show features ranked by
    mean absolute SHAP value. Supports binary, multiclass, and regression types:
    - Binary/Regression: shap_values_array is a 2D ndarray passed directly.
    - Multiclass: shap_values_array is a list of K arrays; SHAP renders stacked bars.
    feature_dataframe is not required for bar charts (mean absolute values only),
    but the render method accepts None safely for interface consistency.
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
        output_path: Path,
    ) -> Path:
        """Renders feature_importance_bar.png and writes it to output_path.

        Passes shap_result.shap_values_array directly to shap.summary_plot() with
        plot_type="bar". No feature_dataframe is needed; bar charts use only
        mean(|SHAP values|) and feature names.

        Args:
            shap_result: Completed SHAPResult produced by SHAPService.
            output_path: Destination path, typically from OutputManager.plot_path().

        Returns:
            The resolved output_path that was written.

        Raises:
            VisualizationError: If shap.summary_plot() or plt.savefig() fails.
        """
        self._execution_logger.log_plot_generation(
            f"Starting feature importance bar chart generation => {output_path}"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shap.summary_plot(
                shap_result.shap_values_array,
                features=None,
                feature_names=list(shap_result.feature_names),
                plot_type="bar",
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
                f"Failed to generate feature importance bar chart at '{output_path}': {plot_error}"
            ) from plot_error
        finally:
            plt.close("all")

        self._execution_logger.log_plot_generation(
            f"Feature importance bar chart written => {output_path}"
        )
        return output_path
