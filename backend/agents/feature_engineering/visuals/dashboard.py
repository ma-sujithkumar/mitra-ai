"""Orchestrates all visualizers and generates a self-contained dashboard HTML."""
from __future__ import annotations

from pathlib import Path

from backend.agents.feature_engineering.visuals.artifact_reader import ArtifactReader
from backend.agents.feature_engineering.visuals.correlation import CorrelationClusterVisualizer
from backend.agents.feature_engineering.visuals.creation import FeatureCreationVisualizer
from backend.agents.feature_engineering.visuals.decisions import DecisionTableVisualizer
from backend.agents.feature_engineering.visuals.importance import FeatureImportanceVisualizer
from backend.agents.feature_engineering.visuals.pca import PCAVarianceVisualizer
from backend.agents.feature_engineering.visuals.selection import SelectionRationaleVisualizer
from backend.agents.feature_engineering.visuals.timeline import PipelineTimelineVisualizer

_DASHBOARD_CSS = """
body { font-family: Arial, sans-serif; margin: 0; background: #f4f6f8; color: #222; }
header { background: #343a40; color: #fff; padding: 18px 30px; }
header h1 { margin: 0 0 6px 0; font-size: 1.4em; }
header .meta { font-size: 0.85em; opacity: 0.8; }
nav { background: #fff; border-bottom: 1px solid #ddd; padding: 10px 30px; display: flex; flex-wrap: wrap; gap: 8px; }
nav a { color: #1f77b4; text-decoration: none; padding: 4px 10px; border: 1px solid #1f77b4; border-radius: 4px; font-size: 0.85em; }
nav a:hover { background: #1f77b4; color: #fff; }
.warnings { background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; margin: 16px 24px; padding: 12px 18px; font-size: 0.88em; }
.warnings h3 { margin: 0 0 6px 0; color: #856404; }
.chart-section { margin: 20px 24px; background: #fff; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow: hidden; }
.chart-section h2 { margin: 0; padding: 12px 18px; background: #495057; color: #fff; font-size: 1em; }
iframe { display: block; width: 100%; border: none; }
"""

_CHART_META = [
    ("01_feature_importance.html", "Feature Importance (MI / RF / mRMR)", 620),
    ("02_correlation_clusters.html", "Correlation Clusters", 680),
    ("03_imputation_decisions.html", "Imputation Decisions", 500),
    ("04_outlier_decisions.html", "Outlier Handling Decisions", 440),
    ("05_scaling_decisions.html", "Scaling Decisions", 500),
    ("06_created_features.html", "Created Features", 540),
    ("07_selection_rationale.html", "Feature Selection Rationale", 620),
    ("08_pca_variance.html", "PCA Explained Variance", 500),
    ("09_pipeline_timeline.html", "Pipeline Step Timeline", 460),
]


class VisualDashboard:
    """Generates all pipeline visualizations and a unified dashboard HTML page."""

    def __init__(
        self,
        output_dir: Path,
        stats_dir: Path | None = None,
        verbose: bool = False,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._stats_dir = Path(stats_dir) if stats_dir else None
        self._verbose = verbose

    def _log(self, message: str) -> None:
        if self._verbose:
            print(f"=> [visuals] {message}")

    def run(self) -> Path:
        """Build all charts, write HTML files, generate dashboard. Returns dashboard path."""
        plots_dir = self._output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        reader = ArtifactReader(self._output_dir)
        self._log(f"run_id={reader.run_id}, task={reader.task}, target={reader.target_column}")

        generated_paths: dict[str, Path | None] = {}

        self._log("building feature importance chart")
        generated_paths["01_feature_importance.html"] = FeatureImportanceVisualizer(
            reader, plots_dir
        ).save("01_feature_importance.html")

        self._log("building correlation cluster heatmap")
        generated_paths["02_correlation_clusters.html"] = CorrelationClusterVisualizer(
            reader, plots_dir
        ).save("02_correlation_clusters.html")

        self._log("building imputation decisions table")
        generated_paths["03_imputation_decisions.html"] = DecisionTableVisualizer(
            reader, plots_dir, "imputation"
        ).save("03_imputation_decisions.html")

        self._log("building outlier decisions table")
        generated_paths["04_outlier_decisions.html"] = DecisionTableVisualizer(
            reader, plots_dir, "outlier"
        ).save("04_outlier_decisions.html")

        self._log("building scaling decisions table")
        generated_paths["05_scaling_decisions.html"] = DecisionTableVisualizer(
            reader, plots_dir, "scaling"
        ).save("05_scaling_decisions.html")

        self._log("building feature creation chart")
        generated_paths["06_created_features.html"] = FeatureCreationVisualizer(
            reader, plots_dir
        ).save("06_created_features.html")

        self._log("building selection rationale chart")
        generated_paths["07_selection_rationale.html"] = SelectionRationaleVisualizer(
            reader, plots_dir
        ).save("07_selection_rationale.html")

        self._log("building PCA scree plot")
        generated_paths["08_pca_variance.html"] = PCAVarianceVisualizer(
            reader, plots_dir
        ).save("08_pca_variance.html")

        self._log("building pipeline timeline")
        generated_paths["09_pipeline_timeline.html"] = PipelineTimelineVisualizer(
            reader, plots_dir
        ).save("09_pipeline_timeline.html")

        dashboard_path = self._build_dashboard_html(reader, generated_paths, plots_dir)
        self._log(f"dashboard written => {dashboard_path}")
        return dashboard_path

    def _build_dashboard_html(
        self,
        reader: ArtifactReader,
        generated_paths: dict[str, Path | None],
        plots_dir: Path,
    ) -> Path:
        num_selected = len(reader.selected_columns)
        num_total = len(reader.mi_scores) or (num_selected + len(reader.dropped_columns))

        meta_lines = [
            f"Run ID: {reader.run_id}",
            f"Task: {reader.task}",
            f"Target: {reader.target_column}",
            f"Features: {num_total} in => {num_selected} selected",
            f"Selection method: {reader.selection_method}",
        ]

        warnings = reader.warnings
        warnings_html = ""
        if warnings:
            items_html = "".join(f"<li>{w}</li>" for w in warnings)
            warnings_html = f'<div class="warnings"><h3>Warnings ({len(warnings)})</h3><ul>{items_html}</ul></div>'

        nav_links: list[str] = []
        chart_sections: list[str] = []
        for filename, title, iframe_height in _CHART_META:
            if generated_paths.get(filename) is None:
                continue
            anchor_id = filename.replace(".", "_")
            nav_links.append(f'<a href="#{anchor_id}">{title}</a>')
            chart_sections.append(
                f'<div class="chart-section" id="{anchor_id}">'
                f"<h2>{title}</h2>"
                f'<iframe src="{filename}" height="{iframe_height}"></iframe>'
                f"</div>"
            )

        meta_html = " &nbsp;|&nbsp; ".join(meta_lines)
        nav_html = "\n".join(nav_links)
        sections_html = "\n".join(chart_sections)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Feature Engineering Dashboard - {reader.run_id}</title>
<style>{_DASHBOARD_CSS}</style>
</head>
<body>
<header>
  <h1>Feature Engineering Pipeline - Dashboard</h1>
  <div class="meta">{meta_html}</div>
</header>
<nav>
{nav_html}
</nav>
{warnings_html}
{sections_html}
</body>
</html>"""

        dashboard_path = plots_dir / "dashboard.html"
        dashboard_path.write_text(html_content, encoding="utf-8")
        return dashboard_path
