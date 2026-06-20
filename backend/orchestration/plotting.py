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

    # Cap on the number of standalone feature histograms drawn in the EDA stage.
    FEATURE_HISTOGRAM_LIMIT: int = 8
    # Maximum numeric columns included in the correlation heatmap to keep it legible.
    CORRELATION_COLUMN_LIMIT: int = 30
    # Number of histogram bins used across distribution plots.
    HISTOGRAM_BINS: int = 30

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
            "eda": lambda: self._build_eda(target_column, task_type),
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

        numeric_dataframe = dataframe.select_dtypes(include=[numpy.number])

        plot_jobs: list[tuple[str, Callable[[], Optional[str]]]] = [
            ("correlation_heatmap",
             lambda: self._plot_correlation_heatmap(numeric_dataframe, stage_dir)),
            ("target_distribution",
             lambda: self._plot_target_distribution(
                 dataframe, target_column, task_type, stage_dir)),
            ("missingness",
             lambda: self._plot_missingness(dataframe, stage_dir)),
            ("feature_histograms",
             lambda: self._plot_feature_histograms(
                 numeric_dataframe, target_column, stage_dir)),
            ("feature_count_note",
             lambda: self._plot_feature_count_note(dataframe, target_column, stage_dir)),
        ]
        for plot_label, plot_callable in plot_jobs:
            result_path = self._guarded(plot_callable, f"eda/{plot_label}")
            if result_path is not None:
                written.append(result_path)
        return written

    def _plot_correlation_heatmap(
        self, numeric_dataframe: "pandas.DataFrame", stage_dir: Path
    ) -> Optional[str]:
        """Draw a correlation heatmap over numeric columns (capped count)."""
        if numeric_dataframe.shape[1] < 2:
            return None
        capped = numeric_dataframe.iloc[:, : self.CORRELATION_COLUMN_LIMIT]
        correlation_matrix = capped.corr()

        figure, axis = pyplot.subplots(
            figsize=(max(6, correlation_matrix.shape[1] * 0.5),
                     max(5, correlation_matrix.shape[1] * 0.5))
        )
        heatmap_image = axis.imshow(
            correlation_matrix.values, cmap="coolwarm", vmin=-1.0, vmax=1.0
        )
        axis.set_xticks(range(correlation_matrix.shape[1]))
        axis.set_yticks(range(correlation_matrix.shape[0]))
        axis.set_xticklabels(correlation_matrix.columns, rotation=90, fontsize=7)
        axis.set_yticklabels(correlation_matrix.index, fontsize=7)
        axis.set_title("Feature Correlation Heatmap")
        figure.colorbar(heatmap_image, ax=axis, fraction=0.046, pad=0.04)
        return self._save_figure(figure, self._output_path(stage_dir, "correlation_heatmap"))

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
            axis.set_title("Target Distribution (regression)")
        else:
            value_counts = target_series.value_counts().sort_index()
            axis.bar([str(label) for label in value_counts.index],
                     value_counts.values, color="#1f77b4")
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

    def _plot_feature_histograms(
        self,
        numeric_dataframe: "pandas.DataFrame",
        target_column: Optional[str],
        stage_dir: Path,
    ) -> Optional[str]:
        """Grid of histograms for a capped number of numeric features."""
        feature_columns = [
            column for column in numeric_dataframe.columns if column != target_column
        ]
        feature_columns = feature_columns[: self.FEATURE_HISTOGRAM_LIMIT]
        if not feature_columns:
            return None

        column_count = min(3, len(feature_columns))
        row_count = int(numpy.ceil(len(feature_columns) / column_count))
        figure, axes = pyplot.subplots(
            row_count, column_count, figsize=(column_count * 4, row_count * 3)
        )
        axes_flat = numpy.atleast_1d(axes).ravel()
        for axis_index, feature_name in enumerate(feature_columns):
            axis = axes_flat[axis_index]
            axis.hist(
                numeric_dataframe[feature_name].dropna().values,
                bins=self.HISTOGRAM_BINS, color="#1f77b4",
            )
            axis.set_title(feature_name, fontsize=9)
        # Hide any unused subplot axes in the grid.
        for unused_index in range(len(feature_columns), len(axes_flat)):
            axes_flat[unused_index].axis("off")
        figure.suptitle("Feature Histograms")
        return self._save_figure(figure, self._output_path(stage_dir, "feature_histograms"))

    def _plot_feature_count_note(
        self,
        dataframe: "pandas.DataFrame",
        target_column: Optional[str],
        stage_dir: Path,
    ) -> Optional[str]:
        """Small text-card plot summarizing dataset shape and feature counts."""
        total_columns = dataframe.shape[1]
        feature_count = total_columns - (1 if target_column in dataframe.columns else 0)
        numeric_count = dataframe.select_dtypes(include=[numpy.number]).shape[1]
        non_numeric_count = total_columns - numeric_count

        note_lines = [
            f"rows: {dataframe.shape[0]}",
            f"total columns: {total_columns}",
            f"features (excl. target): {feature_count}",
            f"numeric columns: {numeric_count}",
            f"non-numeric columns: {non_numeric_count}",
        ]
        figure, axis = pyplot.subplots(figsize=(6, 4))
        axis.axis("off")
        axis.set_title("Dataset Summary")
        axis.text(
            0.05, 0.95, "\n".join(note_lines),
            transform=axis.transAxes, fontsize=12, va="top", family="monospace",
        )
        return self._save_figure(figure, self._output_path(stage_dir, "feature_count_note"))

    # ------------------------------------------------------------- training

    def _build_training(self) -> list[str]:
        """Generate the model metric leaderboard from training_summary.json."""
        summary = self._load_json(self.training_summary_path)
        if summary is None:
            return []
        stage_dir = self._ensure_stage_dir("training")
        written: list[str] = []

        result_path = self._guarded(
            lambda: self._plot_metric_leaderboard(summary, stage_dir),
            "training/metric_leaderboard",
        )
        if result_path is not None:
            written.append(result_path)
        # Confusion-matrix plots require per-row predictions which are not
        # persisted by the pipeline; intentionally skipped (do not fabricate).
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

        scored.sort(key=lambda pair: pair[1])
        model_names = [name for name, _ in scored]
        metric_values = [value for _, value in scored]

        figure, axis = pyplot.subplots(figsize=(8, max(3, len(scored) * 0.45)))
        axis.barh(model_names, metric_values, color="#2ca02c")
        axis.set_xlabel(metric_label)
        axis.set_title(f"Model Leaderboard ({metric_label})")
        return self._save_figure(figure, self._output_path(stage_dir, "metric_leaderboard"))

    # ---------------------------------------------------------- overfitting

    def _build_overfitting(self) -> list[str]:
        """Generate train-vs-validation gap bars from overfitting analyses."""
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
        axis.set_xticks(bar_positions)
        axis.set_xticklabels(model_names, rotation=45, ha="right")
        axis.set_ylabel("primary metric")
        axis.set_title("Train vs Validation Gap per Model")
        axis.legend()
        return self._save_figure(figure, self._output_path(stage_dir, "train_vs_validation_gap"))

    # ----------------------------------------------------------------- hpt

    def _build_hpt(self) -> list[str]:
        """Generate optimization-history line if per-trial scores are present."""
        hpt_data = self._load_json(self.hpt_results_path)
        if hpt_data is None:
            return []
        stage_dir = self._ensure_stage_dir("hpt")
        written: list[str] = []

        result_path = self._guarded(
            lambda: self._plot_hpt_optimization_history(hpt_data, stage_dir),
            "hpt/optimization_history",
        )
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

    # --------------------------------------------------------------- judge

    def _build_judge(self) -> list[str]:
        """Generate the ranked-models verdict bar from judge_decision.json."""
        judge_decision = self._load_json(self.judge_decision_path)
        if judge_decision is None:
            return []
        stage_dir = self._ensure_stage_dir("judge")
        written: list[str] = []

        result_path = self._guarded(
            lambda: self._plot_judge_ranking(judge_decision, stage_dir),
            "judge/ranked_models",
        )
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
        ).head(self.max_features)
        if top_features.empty:
            return None

        figure, axis = pyplot.subplots(figsize=(8, max(3, len(top_features) * 0.35)))
        axis.barh(
            top_features["feature_name"][::-1],
            top_features["mean_absolute_shap_value"][::-1],
            color="#9467bd",
        )
        axis.set_xlabel("mean |SHAP value|")
        axis.set_title(f"SHAP Global Importance: {model_name}")
        return self._save_figure(
            figure, self._output_path(stage_dir, f"shap_global_importance_{model_name}")
        )
