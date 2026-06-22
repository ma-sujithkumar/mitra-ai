"""Decision tables for imputation, outlier handling, and scaling (all deterministic)."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from backend.agents.feature_engineering.visuals.artifact_reader import ArtifactReader
from backend.agents.feature_engineering.visuals.base import BaseVisualizer

# Amber fill for deterministic (rule-based) rows
COLOR_DETERMINISTIC_FILL = "#fff3cd"
COLOR_HEADER_FILL = "#343a40"
COLOR_HEADER_FONT = "#ffffff"


def _imputation_rule_text(strategy: str, null_rate: float, drop_threshold: float = 0.5) -> str:
    if strategy == "drop":
        return f"null_rate ({null_rate:.2f}) > threshold ({drop_threshold:.2f}) => drop"
    if strategy in ("median",):
        return "numeric type => median"
    if strategy in ("mode", "mode_fallback"):
        return "non-numeric type => mode"
    if strategy == "knn":
        return "LLM selected => knn imputation"
    if strategy == "iterative":
        return "LLM selected => iterative (MICE)"
    return f"strategy={strategy}"


def _scaling_rule_text(scaler: str, outlier_rate: float, skewness: float) -> str:
    if scaler == "robust":
        return f"outlier_rate ({outlier_rate:.3f}) > 0.10 => robust"
    if scaler == "power":
        return f"|skewness| ({abs(skewness):.3f}) > 1.0 => power (Yeo-Johnson)"
    return "no extremes (outlier_rate <= 0.10, |skewness| <= 1.0) => standard"


class DecisionTableVisualizer(BaseVisualizer):
    """Plotly table showing per-column decisions and the rule that drove each decision."""

    VALID_TYPES = ("imputation", "outlier", "scaling")

    def __init__(self, reader: ArtifactReader, output_dir: Path, decision_type: str) -> None:
        super().__init__(reader, output_dir)
        if decision_type not in self.VALID_TYPES:
            raise ValueError(f"decision_type must be one of {self.VALID_TYPES}")
        self.decision_type = decision_type

    def build(self) -> go.Figure | None:
        if self.decision_type == "imputation":
            return self._build_imputation_table()
        if self.decision_type == "outlier":
            return self._build_outlier_table()
        return self._build_scaling_table()

    def _build_imputation_table(self) -> go.Figure | None:
        imputation_rows = [
            transformer for transformer in self.reader.transformers
            if transformer.get("step") == "imputation"
        ]
        if not imputation_rows:
            return None

        profile = self.reader.profile
        columns_list: list[str] = []
        strategy_list: list[str] = []
        fill_value_list: list[str] = []
        null_rate_list: list[str] = []
        semantic_type_list: list[str] = []
        rule_list: list[str] = []

        for transformer in imputation_rows:
            col = transformer.get("column", "")
            strategy = transformer.get("strategy", "")
            fill_val = transformer.get("fill_value")
            col_profile = profile.get(col, {})
            null_rate = float(col_profile.get("null_rate", 0.0))

            semantic_type = "numeric" if strategy == "median" else "non-numeric"
            fill_str = f"{fill_val:.4f}" if isinstance(fill_val, float) else str(fill_val or "N/A")

            columns_list.append(col)
            strategy_list.append(strategy)
            fill_value_list.append(fill_str)
            null_rate_list.append(f"{null_rate:.4f}")
            semantic_type_list.append(semantic_type)
            rule_list.append(_imputation_rule_text(strategy, null_rate))

        row_count = len(columns_list)
        cell_fills = [[COLOR_DETERMINISTIC_FILL] * row_count for _ in range(6)]

        figure = go.Figure(data=go.Table(
            header=dict(
                values=["Column", "Strategy", "Fill Value", "Null Rate", "Semantic Type", "Rule Applied"],
                fill_color=COLOR_HEADER_FILL,
                font=dict(color=COLOR_HEADER_FONT, size=12),
                align="left",
            ),
            cells=dict(
                values=[
                    columns_list, strategy_list, fill_value_list,
                    null_rate_list, semantic_type_list, rule_list,
                ],
                fill_color=cell_fills,
                align="left",
                font=dict(size=11),
            ),
        ))
        figure.update_layout(
            title=f"Imputation Decisions - Rule-Based ({row_count} columns)",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return figure

    def _build_outlier_table(self) -> go.Figure | None:
        outlier_rows = [
            transformer for transformer in self.reader.transformers
            if transformer.get("step") == "outlier_scale"
        ]
        if not outlier_rows:
            return None

        columns_list: list[str] = []
        action_list: list[str] = []
        center_list: list[str] = []
        scale_list: list[str] = []
        rule_list: list[str] = []

        for transformer in outlier_rows:
            col = transformer.get("column", "")
            center = transformer.get("center", "N/A")
            scale = transformer.get("scale", "N/A")
            columns_list.append(col)
            action_list.append("scale (RobustScaler)")
            center_list.append(f"{center:.4f}" if isinstance(center, float) else str(center))
            scale_list.append(f"{scale:.4f}" if isinstance(scale, float) else str(scale))
            rule_list.append("deterministic: all numeric cols => iqr + scale")

        row_count = len(columns_list)
        cell_fills = [[COLOR_DETERMINISTIC_FILL] * row_count for _ in range(5)]

        figure = go.Figure(data=go.Table(
            header=dict(
                values=["Column", "Action", "Center (median)", "Scale (IQR)", "Rule Applied"],
                fill_color=COLOR_HEADER_FILL,
                font=dict(color=COLOR_HEADER_FONT, size=12),
                align="left",
            ),
            cells=dict(
                values=[columns_list, action_list, center_list, scale_list, rule_list],
                fill_color=cell_fills,
                align="left",
                font=dict(size=11),
            ),
        ))
        figure.update_layout(
            title=f"Outlier Handling Decisions - Rule-Based ({row_count} columns)",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return figure

    def _build_scaling_table(self) -> go.Figure | None:
        scaling_rows = [
            transformer for transformer in self.reader.transformers
            if transformer.get("step") == "scaling"
        ]
        if not scaling_rows:
            return None

        profile = self.reader.profile
        columns_list: list[str] = []
        scaler_list: list[str] = []
        outlier_rate_list: list[str] = []
        skewness_list: list[str] = []
        rule_list: list[str] = []

        for transformer in scaling_rows:
            col = transformer.get("column", "")
            scaler = transformer.get("strategy", "")
            col_profile = profile.get(col, {})
            outlier_rate = float(col_profile.get("outlier_rate", 0.0))
            skewness = float(col_profile.get("skewness", 0.0))

            columns_list.append(col)
            scaler_list.append(scaler)
            outlier_rate_list.append(f"{outlier_rate:.4f}")
            skewness_list.append(f"{abs(skewness):.4f}")
            rule_list.append(_scaling_rule_text(scaler, outlier_rate, skewness))

        row_count = len(columns_list)
        cell_fills = [[COLOR_DETERMINISTIC_FILL] * row_count for _ in range(5)]

        figure = go.Figure(data=go.Table(
            header=dict(
                values=["Column", "Scaler", "Outlier Rate", "|Skewness|", "Rule Applied"],
                fill_color=COLOR_HEADER_FILL,
                font=dict(color=COLOR_HEADER_FONT, size=12),
                align="left",
            ),
            cells=dict(
                values=[columns_list, scaler_list, outlier_rate_list, skewness_list, rule_list],
                fill_color=cell_fills,
                align="left",
                font=dict(size=11),
            ),
        ))
        figure.update_layout(
            title=f"Scaling Decisions - Rule-Based ({row_count} columns)",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return figure
