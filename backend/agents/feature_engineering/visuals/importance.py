"""Feature importance visualization: MI, RF importance, mRMR rank as grouped bar chart."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from backend.agents.feature_engineering.visuals.artifact_reader import ArtifactReader
from backend.agents.feature_engineering.visuals.base import BaseVisualizer

COLOR_SELECTED = "#2ca02c"
COLOR_DROPPED = "#d62728"


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to [0, 1] range for cross-metric comparability."""
    if not scores:
        return {}
    max_value = max(scores.values())
    if max_value == 0:
        return {col: 0.0 for col in scores}
    return {col: val / max_value for col, val in scores.items()}


class FeatureImportanceVisualizer(BaseVisualizer):
    """Grouped horizontal bar chart comparing MI, RF importance, and mRMR rank per feature."""

    def build(self) -> go.Figure | None:
        mi_scores = self.reader.mi_scores
        if not mi_scores:
            return None

        rf_scores = self.reader.rf_scores
        mrmr_ranked = self.reader.mrmr_ranked
        selected_set = set(self.reader.selected_columns)
        baseline_score = self.reader.linear_baseline.get("score", None)

        # Build mRMR score from rank position (rank 0 = best = score 1.0)
        total_mrmr = len(mrmr_ranked)
        mrmr_scores: dict[str, float] = {}
        if total_mrmr > 0:
            mrmr_scores = {
                col: (total_mrmr - rank_index) / total_mrmr
                for rank_index, col in enumerate(mrmr_ranked)
            }

        # Normalize each metric independently so bars are on the same [0,1] axis
        mi_norm = _normalize_scores(mi_scores)
        rf_norm = _normalize_scores(rf_scores)
        mrmr_norm = _normalize_scores(mrmr_scores)

        # Sort features by raw MI score descending
        sorted_features = sorted(mi_scores.keys(), key=lambda col: mi_scores[col], reverse=True)
        bar_colors = [COLOR_SELECTED if col in selected_set else COLOR_DROPPED for col in sorted_features]

        hover_mi = [
            f"<b>{col}</b><br>MI: {mi_scores.get(col, 0):.4f}<br>"
            f"RF: {rf_scores.get(col, 0):.4f}<br>"
            f"mRMR rank: {mrmr_ranked.index(col) + 1 if col in mrmr_ranked else 'N/A'}<br>"
            f"Status: {'SELECTED' if col in selected_set else 'DROPPED'}"
            for col in sorted_features
        ]

        figure = go.Figure()
        figure.add_trace(go.Bar(
            name="MI Score (norm)",
            x=[mi_norm.get(col, 0) for col in sorted_features],
            y=sorted_features,
            orientation="h",
            marker_color=bar_colors,
            customdata=hover_mi,
            hovertemplate="%{customdata}<extra></extra>",
            opacity=0.9,
        ))
        figure.add_trace(go.Bar(
            name="RF Importance (norm)",
            x=[rf_norm.get(col, 0) for col in sorted_features],
            y=sorted_features,
            orientation="h",
            marker_color="#1f77b4",
            opacity=0.6,
            hovertemplate="RF (norm): %{x:.4f}<extra></extra>",
        ))
        figure.add_trace(go.Bar(
            name="mRMR Rank Score (norm)",
            x=[mrmr_norm.get(col, 0) for col in sorted_features],
            y=sorted_features,
            orientation="h",
            marker_color="#9467bd",
            opacity=0.5,
            hovertemplate="mRMR score (norm): %{x:.4f}<extra></extra>",
        ))

        baseline_annotation_text = (
            f"Linear baseline: {baseline_score:.4f}" if baseline_score is not None else ""
        )

        figure.update_layout(
            title="Feature Importance - MI / RF / mRMR (green=selected, red=dropped)",
            xaxis_title="Normalized Score [0-1]",
            yaxis_title="Feature",
            barmode="overlay",
            height=max(400, len(sorted_features) * 22 + 100),
            margin=dict(l=200, r=40, t=60, b=60),
            legend=dict(orientation="h", y=1.05),
            annotations=[
                dict(
                    text=baseline_annotation_text,
                    xref="paper", yref="paper",
                    x=1.0, y=-0.05,
                    showarrow=False,
                    font=dict(size=11, color="#555"),
                )
            ] if baseline_annotation_text else [],
        )
        return figure
