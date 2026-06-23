"""Feature importance visualization: MI, Information Gain, mRMR rank, and Laplacian Score."""
from __future__ import annotations

import math
from pathlib import Path

import plotly.graph_objects as go

from backend.agents.feature_engineering.visuals.artifact_reader import ArtifactReader
from backend.agents.feature_engineering.visuals.base import BaseVisualizer, MAX_VISUAL_ROWS

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


def _invert_and_normalize_laplacian(scores: dict[str, float]) -> dict[str, float]:
    """Invert Laplacian scores so higher bar = more important (lower raw score = more important).

    Inf values (zero-variance features) are treated as worst (score 0 after inversion).
    """
    if not scores:
        return {}
    finite_values = [v for v in scores.values() if not math.isinf(v) and not math.isnan(v)]
    if not finite_values:
        return {col: 0.0 for col in scores}
    max_finite = max(finite_values)
    # Inverted: best feature (lowest laplacian) gets max_finite, worst (inf) gets 0.
    inverted = {
        col: (max_finite - val) if (not math.isinf(val) and not math.isnan(val)) else 0.0
        for col, val in scores.items()
    }
    max_inverted = max(inverted.values()) if inverted else 1.0
    if max_inverted == 0:
        return {col: 0.0 for col in inverted}
    return {col: val / max_inverted for col, val in inverted.items()}


class FeatureImportanceVisualizer(BaseVisualizer):
    """Grouped horizontal bar chart: MI, Information Gain, mRMR rank, Laplacian Score per feature."""

    def build(self) -> go.Figure | None:
        mi_scores = self.reader.mi_scores
        if not mi_scores:
            return None

        ig_scores = self.reader.information_gain_scores
        laplacian_scores = self.reader.laplacian_scores
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
        ig_norm = _normalize_scores(ig_scores)
        mrmr_norm = _normalize_scores(mrmr_scores)
        # Laplacian: lower raw score = more important, so invert before normalizing
        laplacian_norm = _invert_and_normalize_laplacian(laplacian_scores)

        # Sort features by raw MI score descending, cap at top MAX_VISUAL_ROWS
        sorted_features = sorted(mi_scores.keys(), key=lambda col: mi_scores[col], reverse=True)[:MAX_VISUAL_ROWS]
        bar_colors = [COLOR_SELECTED if col in selected_set else COLOR_DROPPED for col in sorted_features]

        hover_mi = [
            f"<b>{col}</b><br>MI: {mi_scores.get(col, 0):.4f}<br>"
            f"IG: {ig_scores.get(col, 0):.4f}<br>"
            f"Laplacian: {laplacian_scores.get(col, 0):.4f} (lower=better)<br>"
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

        if ig_scores:
            figure.add_trace(go.Bar(
                name="Info Gain (norm)",
                x=[ig_norm.get(col, 0) for col in sorted_features],
                y=sorted_features,
                orientation="h",
                marker_color="#ff7f0e",
                opacity=0.6,
                hovertemplate="IG (norm): %{x:.4f}<extra></extra>",
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

        if laplacian_scores:
            figure.add_trace(go.Bar(
                name="Laplacian Score (inv norm, higher=better)",
                x=[laplacian_norm.get(col, 0) for col in sorted_features],
                y=sorted_features,
                orientation="h",
                marker_color="#17becf",
                opacity=0.4,
                hovertemplate="Laplacian (inv norm): %{x:.4f}<extra></extra>",
            ))

        baseline_annotation_text = (
            f"Linear baseline: {baseline_score:.4f}" if baseline_score is not None else ""
        )

        figure.update_layout(
            title="Feature Importance - MI / Info Gain / mRMR / Laplacian (green=selected, red=dropped)",
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
