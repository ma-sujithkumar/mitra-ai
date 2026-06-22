"""Post-hoc, decoupled visualization generator for a completed MITRA pipeline run.

This module reads already-written session artifacts (no live pipeline state) and
dumps a comprehensive set of plots under ``session_dir/plots/<stage>/``. It is
designed to never abort the caller: every individual plot body is guarded so one
failing plot cannot stop the rest, and the public entry point never raises.

Artifacts consumed (only when present; missing inputs are skipped silently):
  data/engineered_dataset.csv (fallback data/train.csv)  => eda/
  reports/metadata.json                                   => task_type, target_column
  reports/training_summary.json                           => training/
  reports/judge_decision.json                             => judge/
  evaluation/overfitting/<model>/overfitting_analysis.json => overfitting/
  evaluation/hpt/hpt_results.json                          => hpt/
  evaluation/shap/<model>/csv/global_feature_importance.csv => shap/
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib

# Force the non-interactive Agg backend immediately after importing matplotlib,
# before pyplot is imported, so plotting works headless with no display.
matplotlib.use("Agg")

import matplotlib.pyplot as pyplot
import numpy
import pandas

logger = logging.getLogger(__name__)


class PipelinePlotGenerator:
    """Generates comprehensive post-hoc plots for one completed pipeline session.

    The generator is purely artifact-driven: it inspects the on-disk session
    directory and produces whatever plots the available artifacts support.
    """

    # Config-driven render defaults (no scattered magic numbers).
    DEFAULT_DPI: int = 110
    DEFAULT_IMAGE_FORMAT: str = "png"
    DEFAULT_MAX_FEATURES: int = 20

    # Number of histogram bins used across distribution plots.
    HISTOGRAM_BINS: int = 30
    # Maximum SHAP features shown per model (keeps charts readable).
    SHAP_TOP_FEATURES: int = 10
    # Gap threshold above which overfitting is flagged on the chart.
    OVERFITTING_GAP_WARN_THRESHOLD: float = 0.10

    # Color mapping for judge verdicts (avoids if-else ladder per CLAUDE.md #4/#23).
    VERDICT_COLOR_MAP: dict[str, str] = {
        "select": "#2ca02c",
        "reject": "#d62728",
    }
    VERDICT_DEFAULT_COLOR: str = "#7f7f7f"

    # Candidate keys for the per-model primary metric, in priority order.
    PRIMARY_METRIC_KEYS: tuple[str, ...] = ("accuracy", "r2", "f1", "rmse")

    def __init__(
        self,
        session_dir: Path,
        dpi: int = DEFAULT_DPI,
        image_format: str = DEFAULT_IMAGE_FORMAT,
        max_features: int = DEFAULT_MAX_FEATURES,
    ) -> None:
        self.session_dir: Path = Path(session_dir)
        self.dpi: int = dpi
        self.image_format: str = image_format
        self.max_features: int = max_features

        # Resolved artifact locations (existence is checked lazily by each stage).
        self.data_dir: Path = self.session_dir / "data"
        self.reports_dir: Path = self.session_dir / "reports"
        self.evaluation_dir: Path = self.session_dir / "evaluation"
        self.plots_root: Path = self.session_dir / "plots"

        self.engineered_csv: Path = self.data_dir / "engineered_dataset.csv"
        self.train_csv: Path = self.data_dir / "train.csv"
        self.metadata_path: Path = self.reports_dir / "metadata.json"
        self.training_summary_path: Path = self.reports_dir / "training_summary.json"
        self.judge_decision_path: Path = self.reports_dir / "judge_decision.json"
        self.overfitting_dir: Path = self.evaluation_dir / "overfitting"
        self.shap_dir: Path = self.evaluation_dir / "shap"
        self.hpt_results_path: Path = self.evaluation_dir / "hpt" / "hpt_results.json"

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _load_json(json_path: Path) -> Optional[dict[str, Any]]:
        """Load a JSON artifact, returning None if it is absent or unreadable."""
        if not json_path.exists():
            return None
        with json_path.open(encoding="utf-8") as json_file:
            return json.load(json_file)

    def _resolve_dataset_path(self) -> Optional[Path]:
        """Return the engineered dataset path, falling back to train.csv."""
        if self.engineered_csv.exists():
            return self.engineered_csv
        if self.train_csv.exists():
            return self.train_csv
        return None

    def _resolve_target_column(self, metadata: Optional[dict[str, Any]]) -> Optional[str]:
        """Read the target column name from metadata under either accepted key."""
        if metadata is None:
            return None
        # metadata may store the target under 'target_column' or 'target'.
        return metadata.get("target_column") or metadata.get("target")

    @staticmethod
    def _resolve_task_type(metadata: Optional[dict[str, Any]]) -> str:
        """Read task_type from metadata, defaulting to classification."""
        if metadata is None:
            return "classification"
        return metadata.get("task_type", "classification")

    def _ensure_stage_dir(self, stage_name: str) -> Path:
        """Create and return the output directory for a plotting stage."""
        stage_dir = self.plots_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def _output_path(self, stage_dir: Path, plot_name: str) -> Path:
        """Build a stage-relative output file path with the configured format."""
        return stage_dir / f"{plot_name}.{self.image_format}"

    def _save_figure(self, figure: "pyplot.Figure", output_path: Path) -> str:
        """Save a figure with configured dpi/format, close it, return path str."""
        figure.savefig(output_path, dpi=self.dpi, format=self.image_format, bbox_inches="tight")
        pyplot.close(figure)
        return str(output_path)

    def _guarded(
        self,
        plot_callable: Callable[[], Optional[str]],
        plot_label: str,
    ) -> Optional[str]:
        """Run one plot body, logging and swallowing any failure.

        Each plot is individually guarded so a single failing plot never aborts
        the remaining plots or the public method.
        """
        # The try/except wraps only the plot BODY, never module imports.
        try:
            return plot_callable()
        except Exception as plot_error:  # noqa: BLE001 - intentional broad guard
            logger.warning("=> [plots] skipped %s: %s", plot_label, plot_error)
            return None

    @staticmethod
    def _primary_metric_value(model_entry: dict[str, Any]) -> Optional[float]:
        """Pick the best available primary metric for a completed model.

        Preference order: accuracy => r2 => f1 => rmse => validation_score.
        """
        metrics = model_entry.get("metrics") or {}
        for metric_key in PipelinePlotGenerator.PRIMARY_METRIC_KEYS:
            metric_value = metrics.get(metric_key)
            if metric_value is not None:
                return float(metric_value)
        validation_score = model_entry.get("validation_score")
        if validation_score is not None:
            return float(validation_score)
        return None

    @staticmethod
    def _primary_metric_label(model_entry: dict[str, Any]) -> str:
        """Return the name of the metric chosen by _primary_metric_value."""
        metrics = model_entry.get("metrics") or {}
        for metric_key in PipelinePlotGenerator.PRIMARY_METRIC_KEYS:
            if metrics.get(metric_key) is not None:
                return metric_key
        if model_entry.get("validation_score") is not None:
            return "validation_score"
        return "score"

    # ------------------------------------------------------------- public API

    def generate_all(self) -> dict[str, list[str]]:
        """Generate every supported plot and return stage_name -> [file paths].

        Stages whose source artifacts are missing yield an empty list. This
        method never raises.
        """
        self.plots_root.mkdir(parents=True, exist_ok=True)

        metadata = self._load_json(self.metadata_path)
        target_column = self._resolve_target_column(metadata)
        task_type = self._resolve_task_type(metadata)

        # Stage dispatch map (avoids an if-else ladder; each entry is a thunk).
        stage_builders: dict[str, Callable[[], list[str]]] = {
            "training": self._build_training,
            "overfitting": self._build_overfitting,
            "hpt": self._build_hpt,
            "judge": self._build_judge,
            "shap": self._build_shap,
        }

        results: dict[str, list[str]] = {}
        for stage_name, stage_builder in stage_builders.items():
            # Guard at the stage level too, so a stage-level error is isolated.
            try:
                results[stage_name] = stage_builder()
            except Exception as stage_error:  # noqa: BLE001 - intentional broad guard
                logger.warning("=> [plots] stage %s failed: %s", stage_name, stage_error)
                results[stage_name] = []
        return results

    # ----------------------------------------------------------------- eda

    def _build_eda(self, target_column: Optional[str], task_type: str) -> list[str]:
        """Generate exploratory data analysis plots from the feature matrix."""
        dataset_path = self._resolve_dataset_path()
        if dataset_path is None:
            return []

        dataframe = pandas.read_csv(dataset_path)
        stage_dir = self._ensure_stage_dir("eda")
        written: list[str] = []

        plot_jobs: list[tuple[str, Callable[[], Optional[str]]]] = [
            ("target_distribution",
             lambda: self._plot_target_distribution(
                 dataframe, target_column, task_type, stage_dir)),
            ("missingness",
             lambda: self._plot_missingness(dataframe, stage_dir)),
        ]
        for plot_label, plot_callable in plot_jobs:
            result_path = self._guarded(plot_callable, f"eda/{plot_label}")
            if result_path is not None:
                written.append(result_path)
        return written

    def _plot_target_distribution(
        self,
        dataframe: "pandas.DataFrame",
        target_column: Optional[str],
        task_type: str,
        stage_dir: Path,
    ) -> Optional[str]:
        """Class-balance bar for classification or histogram for regression."""
        if target_column is None or target_column not in dataframe.columns:
            return None
        target_series = dataframe[target_column].dropna()
        figure, axis = pyplot.subplots(figsize=(8, 5))

        if task_type == "regression":
            axis.hist(target_series.values, bins=self.HISTOGRAM_BINS, color="#1f77b4")
            axis.set_xlabel(target_column)
            axis.set_ylabel("count")
            axis.set_title("Target Value Distribution (regression)")
        else:
            value_counts = target_series.value_counts().sort_index()
            bars = axis.bar(
                [str(label) for label in value_counts.index],
                value_counts.values,
                color="#1f77b4",
            )
            total_count = int(value_counts.sum())
            for bar_rect, sample_count in zip(bars, value_counts.values):
                percentage = sample_count / total_count
                axis.text(
                    bar_rect.get_x() + bar_rect.get_width() / 2,
                    bar_rect.get_height() + total_count * 0.005,
                    f"{percentage:.1%}",
                    ha="center", va="bottom", fontsize=9,
                )
            axis.set_xlabel(target_column)
            axis.set_ylabel("count")
            axis.set_title("Target Class Balance (classification)")
            axis.tick_params(axis="x", rotation=45)
        return self._save_figure(figure, self._output_path(stage_dir, "target_distribution"))

    def _plot_missingness(
        self, dataframe: "pandas.DataFrame", stage_dir: Path
    ) -> Optional[str]:
        """Bar of per-feature missing-value fraction (only features with gaps)."""
        missing_fraction = dataframe.isna().mean()
        missing_fraction = missing_fraction[missing_fraction > 0.0].sort_values(ascending=False)
        if missing_fraction.empty:
            return None
        missing_fraction = missing_fraction.head(self.max_features)

        figure, axis = pyplot.subplots(figsize=(8, max(3, len(missing_fraction) * 0.35)))
        axis.barh(missing_fraction.index[::-1], missing_fraction.values[::-1], color="#ff7f0e")
        axis.set_xlabel("fraction missing")
        axis.set_title("Per-Feature Missingness")
        return self._save_figure(figure, self._output_path(stage_dir, "missingness"))

    # ------------------------------------------------------------- training

    def _build_training(self) -> list[str]:
        """Generate training plots from training_summary.json."""
        summary = self._load_json(self.training_summary_path)
        if summary is None:
            return []
        stage_dir = self._ensure_stage_dir("training")
        written: list[str] = []

        result_path = self._guarded(
            lambda: self._plot_multi_metric_comparison(summary, stage_dir),
            "training/multi_metric_comparison",
        )
        if result_path is not None:
            written.append(result_path)
        return written

    def _plot_metric_leaderboard(
        self, summary: dict[str, Any], stage_dir: Path
    ) -> Optional[str]:
        """Horizontal bar of the primary metric per completed model."""
        models = summary.get("models") or []
        completed_models = [
            model_entry for model_entry in models
            if model_entry.get("status") == "completed"
        ]
        scored: list[tuple[str, float]] = []
        metric_label = "score"
        for model_entry in completed_models:
            metric_value = self._primary_metric_value(model_entry)
            if metric_value is None:
                continue
            metric_label = self._primary_metric_label(model_entry)
            scored.append((model_entry.get("model_name", "unknown"), metric_value))
        if not scored:
            return None

        scored.sort(key=lambda pair: pair[1])  # ascending: winner is last
        model_names = [name for name, _ in scored]
        metric_values = [value for _, value in scored]

        # Highlight the top-scoring model in green; others in steel blue.
        bar_colors = ["#4c78a8"] * len(scored)
        bar_colors[-1] = "#2ca02c"

        figure, axis = pyplot.subplots(figsize=(8, max(3, len(scored) * 0.45)))
        bars = axis.barh(model_names, metric_values, color=bar_colors)

        # Value labels on each bar.
        value_range = max(metric_values) - min(metric_values) if len(metric_values) > 1 else metric_values[0]
        label_offset = value_range * 0.01 if value_range > 0 else 0.001
        for bar_rect, metric_value in zip(bars, metric_values):
            axis.text(
                bar_rect.get_width() + label_offset,
                bar_rect.get_y() + bar_rect.get_height() / 2,
                f"{metric_value:.4f}",
                va="center", ha="left", fontsize=9,
            )

        axis.set_xlabel(metric_label)
        axis.set_title(f"Model Performance ({metric_label})  |  green = top scorer")
        axis.set_xlim(right=max(metric_values) * 1.15)
        return self._save_figure(figure, self._output_path(stage_dir, "metric_leaderboard"))

    def _plot_multi_metric_comparison(
        self, summary: dict[str, Any], stage_dir: Path
    ) -> Optional[str]:
        """One subplot per validation metric showing all models side by side.

        Unlike the leaderboard (single primary metric), this shows every persisted
        validation metric so users can see trade-offs across MAE, RMSE, R2, etc.
        Best model per metric is highlighted green.
        """
        models = summary.get("models") or []
        completed_models = [
            model_entry for model_entry in models
            if model_entry.get("status") == "completed"
        ]
        if not completed_models:
            return None

        # Gather metric names present in validation dicts across all models.
        metric_names_set: set[str] = set()
        for model_entry in completed_models:
            val_metrics = (model_entry.get("metrics") or {}).get("validation") or {}
            for metric_key, metric_val in val_metrics.items():
                if isinstance(metric_val, (int, float)) and metric_key != "task_type":
                    metric_names_set.add(metric_key)

        metric_names = sorted(metric_names_set)
        if not metric_names:
            return None

        model_names = [
            model_entry.get("model_name", f"model_{idx}")
            for idx, model_entry in enumerate(completed_models)
        ]

        # Metrics where lower is better (errors).
        error_metric_names = {"mse", "rmse", "mae", "loss", "error"}

        n_metrics = len(metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        figure, axes_grid = pyplot.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
        )
        # Normalise axes_grid to a 2-D list regardless of shape.
        if n_metrics == 1:
            axes_grid = [[axes_grid]]
        elif n_rows == 1:
            axes_grid = [list(axes_grid)]
        else:
            axes_grid = [list(row) for row in axes_grid]

        for metric_idx, metric_name in enumerate(metric_names):
            row_idx = metric_idx // n_cols
            col_idx = metric_idx % n_cols
            axis = axes_grid[row_idx][col_idx]

            values: list[float] = []
            for model_entry in completed_models:
                val_metrics = (model_entry.get("metrics") or {}).get("validation") or {}
                raw_val = val_metrics.get(metric_name)
                values.append(float(raw_val) if raw_val is not None else 0.0)

            lower_is_better = metric_name.lower() in error_metric_names
            best_idx = values.index(min(values) if lower_is_better else max(values))
            bar_colors = [
                "#2ca02c" if idx == best_idx else "#4c78a8"
                for idx in range(len(values))
            ]

            axis.barh(model_names, values, color=bar_colors)
            axis.set_title(
                f"{metric_name}  ({'lower' if lower_is_better else 'higher'} = better)",
                fontsize=10,
            )
            axis.tick_params(axis="y", labelsize=8)

        # Hide any unused subplot cells in the last row.
        for extra_idx in range(n_metrics, n_rows * n_cols):
            axes_grid[extra_idx // n_cols][extra_idx % n_cols].set_visible(False)

        figure.suptitle(
            "Multi-Metric Comparison (green = best model per metric)", fontsize=12
        )
        figure.tight_layout()
        return self._save_figure(
            figure, self._output_path(stage_dir, "multi_metric_comparison")
        )

    # ---------------------------------------------------------- overfitting

    def _build_overfitting(self) -> list[str]:
        """Generate overfitting and CV stability plots from per-model analyses."""
        if not self.overfitting_dir.exists():
            return []
        analysis_paths = sorted(self.overfitting_dir.glob("*/overfitting_analysis.json"))
        if not analysis_paths:
            return []
        stage_dir = self._ensure_stage_dir("overfitting")
        written: list[str] = []

        result_path = self._guarded(
            lambda: self._plot_overfitting_gaps(analysis_paths, stage_dir),
            "overfitting/train_vs_validation_gap",
        )
        if result_path is not None:
            written.append(result_path)
        return written

    def _plot_overfitting_gaps(
        self, analysis_paths: list[Path], stage_dir: Path
    ) -> Optional[str]:
        """Grouped bar of train vs test primary-metric score per model.

        Reads keys: model_name, primary_metric, train_metrics, test_metrics, gaps.
        """
        model_names: list[str] = []
        train_scores: list[float] = []
        test_scores: list[float] = []
        for analysis_path in analysis_paths:
            analysis = self._load_json(analysis_path)
            if analysis is None:
                continue
            primary_metric = analysis.get("primary_metric")
            train_metrics = analysis.get("train_metrics") or {}
            test_metrics = analysis.get("test_metrics") or {}
            train_value = train_metrics.get(primary_metric)
            test_value = test_metrics.get(primary_metric)
            if train_value is None or test_value is None:
                continue
            model_names.append(analysis.get("model_name", analysis_path.parent.name))
            train_scores.append(float(train_value))
            test_scores.append(float(test_value))
        if not model_names:
            return None

        bar_positions = numpy.arange(len(model_names))
        bar_width = 0.38
        figure, axis = pyplot.subplots(figsize=(max(6, len(model_names) * 1.2), 5))
        axis.bar(bar_positions - bar_width / 2, train_scores, bar_width,
                 label="train", color="#1f77b4")
        axis.bar(bar_positions + bar_width / 2, test_scores, bar_width,
                 label="validation/test", color="#ff7f0e")

        # Annotate pairs with a large gap to flag potential overfitting.
        for position, (train_score, test_score) in enumerate(zip(train_scores, test_scores)):
            gap = train_score - test_score
            if abs(gap) >= self.OVERFITTING_GAP_WARN_THRESHOLD:
                annotation_y = max(train_score, test_score) + 0.01
                axis.text(
                    position, annotation_y,
                    f"gap {gap:+.2f}",
                    ha="center", va="bottom", fontsize=8, color="#d62728",
                    fontweight="bold",
                )

        axis.set_xticks(bar_positions)
        axis.set_xticklabels(model_names, rotation=45, ha="right")
        axis.set_ylabel("primary metric")
        axis.set_title("Overfitting Analysis: Train vs Validation Score")
        axis.legend()
        return self._save_figure(figure, self._output_path(stage_dir, "train_vs_validation_gap"))

    def _plot_cv_fold_stability(
        self, analysis_paths: list[Path], stage_dir: Path
    ) -> Optional[str]:
        """Per-fold CV scores per model shown as scatter + mean line.

        Dots represent individual fold scores; the horizontal bar marks the mean.
        A tight cluster means the model is stable across data splits; wide spread
        indicates variance / sensitivity to the particular fold composition.
        """
        fold_data: dict[str, dict[str, Any]] = {}
        for analysis_path in analysis_paths:
            analysis = self._load_json(analysis_path)
            if analysis is None:
                continue
            cv_results = analysis.get("k_fold_cross_validation_results") or {}
            per_fold_scores = cv_results.get("per_fold_scores")
            if not per_fold_scores:
                continue
            model_name = analysis.get("model_name", analysis_path.parent.name)
            fold_data[model_name] = {
                "scores": per_fold_scores,
                "mean": float(cv_results.get("mean", 0.0)),
                "std": float(cv_results.get("std", 0.0)),
            }

        if not fold_data:
            return None

        model_names = list(fold_data.keys())
        n_models = len(model_names)
        positions = numpy.arange(n_models)

        figure, axis = pyplot.subplots(figsize=(max(8, n_models * 1.5), 5))

        for model_idx, model_name in enumerate(model_names):
            model_info = fold_data[model_name]
            fold_scores = model_info["scores"]
            n_folds = len(fold_scores)
            fold_x_positions = numpy.linspace(
                model_idx - 0.3, model_idx + 0.3, n_folds
            )
            axis.scatter(
                fold_x_positions, fold_scores,
                zorder=3, s=55, alpha=0.85, color="#1f77b4",
            )
            # Mean line spanning the model column.
            axis.hlines(
                model_info["mean"],
                model_idx - 0.35, model_idx + 0.35,
                linewidth=2.5, colors="#333333", zorder=4,
            )
            # Annotate with std deviation below the model label area.
            axis.text(
                model_idx, axis.get_ylim()[0] if axis.get_ylim()[0] != 0 else min(fold_scores) - 0.02,
                f"std={model_info['std']:.3f}",
                ha="center", va="top", fontsize=8, color="#555555",
            )

        axis.set_xticks(positions)
        axis.set_xticklabels(model_names, rotation=45, ha="right")
        axis.set_ylabel("CV fold score")
        axis.set_title(
            "Cross-Validation Fold Stability  (dots = per-fold score, line = mean)"
        )
        return self._save_figure(
            figure, self._output_path(stage_dir, "cv_fold_stability")
        )

    # ----------------------------------------------------------------- hpt

    def _build_hpt(self) -> list[str]:
        """Generate HPT optimization history and sensitivity charts."""
        hpt_data = self._load_json(self.hpt_results_path)
        if hpt_data is None:
            return []
        stage_dir = self._ensure_stage_dir("hpt")
        written: list[str] = []

        plot_jobs: list[tuple[str, Callable[[], Optional[str]]]] = [
            ("hpt/optimization_history",
             lambda: self._plot_hpt_optimization_history(hpt_data, stage_dir)),
            ("hpt/sensitivity_analysis",
             lambda: self._plot_hpt_sensitivity(hpt_data, stage_dir)),
        ]
        for plot_label, plot_callable in plot_jobs:
            result_path = self._guarded(plot_callable, plot_label)
            if result_path is not None:
                written.append(result_path)
        return written

    def _plot_hpt_optimization_history(
        self, hpt_data: dict[str, Any], stage_dir: Path
    ) -> Optional[str]:
        """Plot per-trial val scores per model when a trial history is exposed.

        hpt_results.json shape: {hpt_results:[{model_name, ...}], metadata:{...}}.
        Per-trial values are only plotted if a model entry exposes a trial list
        under one of the recognized keys; otherwise this is skipped.
        """
        model_results = hpt_data.get("hpt_results") or []
        if not model_results:
            return None

        trial_list_keys = ("all_trials", "trials", "trial_history")
        score_keys = ("val_score", "score", "value")

        figure, axis = pyplot.subplots(figsize=(8, 5))
        plotted_any = False
        for model_result in model_results:
            trial_list = None
            for trial_key in trial_list_keys:
                if isinstance(model_result.get(trial_key), list):
                    trial_list = model_result[trial_key]
                    break
            if not trial_list:
                continue
            trial_scores: list[float] = []
            for trial_entry in trial_list:
                if not isinstance(trial_entry, dict):
                    continue
                for score_key in score_keys:
                    if trial_entry.get(score_key) is not None:
                        trial_scores.append(float(trial_entry[score_key]))
                        break
            if not trial_scores:
                continue
            axis.plot(range(1, len(trial_scores) + 1), trial_scores,
                      marker="o", label=model_result.get("model_name", "model"))
            plotted_any = True

        if not plotted_any:
            pyplot.close(figure)
            return None
        axis.set_xlabel("trial number")
        axis.set_ylabel("validation score")
        axis.set_title("HPT Optimization History")
        axis.legend(fontsize=8)
        return self._save_figure(figure, self._output_path(stage_dir, "optimization_history"))

    def _plot_hpt_sensitivity(
        self, hpt_data: dict[str, Any], stage_dir: Path
    ) -> Optional[str]:
        """Bar chart of hyperparameter impact (score range) per model.

        score_range = max_val_score - min_val_score observed across trials when
        that parameter varied. A large range means the hyperparameter strongly
        influences model quality; near-zero means the model is insensitive to it.
        """
        model_results = hpt_data.get("hpt_results") or []
        models_with_sensitivity = [
            model_result for model_result in model_results
            if isinstance(model_result.get("hyperparam_sensitivity"), dict)
        ]
        if not models_with_sensitivity:
            return None

        n_models = len(models_with_sensitivity)
        figure, axes_list = pyplot.subplots(
            1, n_models,
            figsize=(max(5, n_models * 4), 5),
        )
        # Normalise to list regardless of whether subplots returns an Axes or array.
        if n_models == 1:
            axes_list = [axes_list]

        for axis, model_result in zip(axes_list, models_with_sensitivity):
            sensitivity_dict = model_result.get("hyperparam_sensitivity") or {}
            # Only entries with a nested dict containing score_range are params.
            param_ranges: dict[str, float] = {
                param_name: float(param_info["score_range"])
                for param_name, param_info in sensitivity_dict.items()
                if isinstance(param_info, dict) and "score_range" in param_info
            }
            if not param_ranges:
                axis.set_visible(False)
                continue

            sorted_params = sorted(
                param_ranges.keys(), key=lambda param: param_ranges[param]
            )
            score_range_values = [param_ranges[param] for param in sorted_params]
            bar_colors = pyplot.cm.Blues(
                numpy.linspace(0.35, 0.85, len(sorted_params))
            )

            axis.barh(sorted_params, score_range_values, color=bar_colors)
            axis.set_xlabel("score range")
            axis.set_title(
                model_result.get("model_name", "model"), fontsize=10
            )

        figure.suptitle(
            "HPT Sensitivity: Hyperparameter Impact  (larger bar = stronger influence on score)",
            fontsize=11,
        )
        figure.tight_layout()
        return self._save_figure(
            figure, self._output_path(stage_dir, "sensitivity_analysis")
        )

    # --------------------------------------------------------------- judge

    def _build_judge(self) -> list[str]:
        """Generate judge verdict and dimension scorecard from judge_decision.json."""
        judge_decision = self._load_json(self.judge_decision_path)
        if judge_decision is None:
            return []
        stage_dir = self._ensure_stage_dir("judge")
        written: list[str] = []

        plot_jobs: list[tuple[str, Callable[[], Optional[str]]]] = [
            ("judge/ranked_models",
             lambda: self._plot_judge_ranking(judge_decision, stage_dir)),
            ("judge/dimension_scorecard",
             lambda: self._plot_judge_dimension_scorecard(judge_decision, stage_dir)),
        ]
        for plot_label, plot_callable in plot_jobs:
            result_path = self._guarded(plot_callable, plot_label)
            if result_path is not None:
                written.append(result_path)
        return written

    def _plot_judge_ranking(
        self, judge_decision: dict[str, Any], stage_dir: Path
    ) -> Optional[str]:
        """Bar of judge score per model, colored by verdict (select/reject).

        Reads keys: ranked_models[{model_name, rank, score, verdict}].
        """
        ranked_models = judge_decision.get("ranked_models") or []
        if not ranked_models:
            return None
        # Order bars by rank ascending (best on top after horizontal flip).
        ranked_sorted = sorted(ranked_models, key=lambda entry: entry.get("rank", 0))
        model_names = [entry.get("model_name", "unknown") for entry in ranked_sorted]
        scores = [float(entry.get("score", 0.0)) for entry in ranked_sorted]
        bar_colors = [
            self.VERDICT_COLOR_MAP.get(entry.get("verdict", ""), self.VERDICT_DEFAULT_COLOR)
            for entry in ranked_sorted
        ]

        figure, axis = pyplot.subplots(figsize=(8, max(3, len(model_names) * 0.5)))
        # Reverse so rank 1 (best) appears at the top of the horizontal chart.
        axis.barh(model_names[::-1], scores[::-1], color=bar_colors[::-1])
        axis.set_xlabel("judge score")
        axis.set_title("Judge Ranked Models (green=select, red=reject)")
        return self._save_figure(figure, self._output_path(stage_dir, "ranked_models"))

    def _plot_judge_dimension_scorecard(
        self, judge_decision: dict[str, Any], stage_dir: Path
    ) -> Optional[str]:
        """Heatmap of per-dimension judge evaluation across all ranked models.

        Each cell shows PASS (green), FAIL (red), or INFO (yellow) for one of
        the 6 evaluation dimensions. This reveals which specific dimension caused
        a model to be penalised, which is not visible from the aggregate score.
        """
        # Numeric encoding for imshow: pass=1, info=0, fail=-1
        status_numeric_map: dict[str, int] = {"pass": 1, "info": 0, "fail": -1}
        status_label_map: dict[int, str] = {1: "PASS", 0: "INFO", -1: "FAIL"}

        ranked_models = judge_decision.get("ranked_models") or []
        models_with_findings = [
            model_entry for model_entry in ranked_models
            if model_entry.get("findings")
        ]
        if not models_with_findings:
            return None

        sorted_models = sorted(
            models_with_findings, key=lambda entry: entry.get("rank", 999)
        )
        model_names: list[str] = []
        dimension_labels: list[str] = []
        status_rows: list[list[int]] = []

        for model_entry in sorted_models:
            findings = model_entry.get("findings") or []
            model_names.append(model_entry.get("model_name", "unknown"))
            if not dimension_labels:
                dimension_labels = [
                    finding.get("label") or finding.get("dimension", "")
                    for finding in findings
                ]
            status_row = [
                status_numeric_map.get(finding.get("status", "info"), 0)
                for finding in findings
            ]
            status_rows.append(status_row)

        if not model_names or not dimension_labels:
            return None

        # Map -1/0/1 to 0/0.5/1 for the RdYlGn colormap (red/yellow/green).
        status_matrix = numpy.array(status_rows, dtype=float)
        normalized_matrix = (status_matrix + 1.0) / 2.0

        n_models = len(model_names)
        n_dimensions = len(dimension_labels)
        figure, axis = pyplot.subplots(
            figsize=(max(8, n_dimensions * 1.6), max(3, n_models * 0.65))
        )
        axis.imshow(normalized_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

        # Annotate every cell with its text label.
        for row_idx in range(n_models):
            for col_idx in range(n_dimensions):
                numeric_val = int(status_matrix[row_idx, col_idx])
                cell_label = status_label_map.get(numeric_val, "?")
                text_color = "white" if numeric_val != 0 else "#333333"
                axis.text(
                    col_idx, row_idx, cell_label,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color,
                )

        axis.set_xticks(range(n_dimensions))
        axis.set_xticklabels(dimension_labels, rotation=30, ha="right", fontsize=9)
        axis.set_yticks(range(n_models))
        axis.set_yticklabels(model_names, fontsize=9)
        axis.set_title(
            "Judge Dimension Scorecard  (green=PASS, red=FAIL, yellow=INFO)"
        )
        return self._save_figure(
            figure, self._output_path(stage_dir, "dimension_scorecard")
        )

    # ---------------------------------------------------------------- shap

    def _build_shap(self) -> list[str]:
        """Generate a global SHAP importance bar per model from the CSVs."""
        if not self.shap_dir.exists():
            return []
        importance_csvs = sorted(
            self.shap_dir.glob("*/csv/global_feature_importance.csv")
        )
        if not importance_csvs:
            return []
        stage_dir = self._ensure_stage_dir("shap")
        written: list[str] = []

        for importance_csv in importance_csvs:
            # Model name is the directory two levels up: <model>/csv/<file>.
            model_name = importance_csv.parent.parent.name
            result_path = self._guarded(
                lambda csv_path=importance_csv, name=model_name:
                    self._plot_shap_importance(csv_path, name, stage_dir),
                f"shap/{model_name}",
            )
            if result_path is not None:
                written.append(result_path)
        return written

    def _plot_shap_importance(
        self, importance_csv: Path, model_name: str, stage_dir: Path
    ) -> Optional[str]:
        """Top-N global SHAP importance bar for one model.

        CSV columns: feature_name, mean_absolute_shap_value (sorted descending).
        """
        importance_dataframe = pandas.read_csv(importance_csv)
        required_columns = {"feature_name", "mean_absolute_shap_value"}
        if not required_columns.issubset(importance_dataframe.columns):
            return None
        top_features = importance_dataframe.sort_values(
            "mean_absolute_shap_value", ascending=False
        ).head(self.SHAP_TOP_FEATURES)
        if top_features.empty:
            return None

        # Reversed so highest-importance feature appears at the top.
        reversed_features = top_features.iloc[::-1].reset_index(drop=True)
        feature_count = len(reversed_features)
        # Gradient from light (low importance) to dark (high importance) purple.
        bar_colors = pyplot.cm.Purples(numpy.linspace(0.35, 0.9, feature_count))

        figure, axis = pyplot.subplots(figsize=(8, max(3, feature_count * 0.38)))
        axis.barh(
            reversed_features["feature_name"],
            reversed_features["mean_absolute_shap_value"],
            color=bar_colors,
        )
        axis.set_xlabel("mean |SHAP value|")
        axis.set_title(f"Key Predictors: {model_name} (top {feature_count})")
        return self._save_figure(
            figure, self._output_path(stage_dir, f"shap_global_importance_{model_name}")
        )
