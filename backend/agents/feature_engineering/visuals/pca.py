"""PCA scree plot: per-component explained variance + cumulative line + threshold marker."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from backend.agents.feature_engineering.visuals.artifact_reader import ArtifactReader
from backend.agents.feature_engineering.visuals.base import BaseVisualizer, MAX_VISUAL_ROWS


class PCAVarianceVisualizer(BaseVisualizer):
    """Scree plot showing per-component and cumulative explained variance ratio."""

    def build(self) -> go.Figure | None:
        pca_data = self.reader.pca_data
        explained_variance_ratios = pca_data.get("explained_variance_ratio", [])
        if not explained_variance_ratios:
            return None

        variance_threshold = float(pca_data.get("variance_retained", 0.95))
        n_components_at_threshold = int(pca_data.get("n_components_for_threshold", 0))

        # Cap at MAX_VISUAL_ROWS components; first N already explain the most variance
        explained_variance_ratios = explained_variance_ratios[:MAX_VISUAL_ROWS]
        component_indices = list(range(1, len(explained_variance_ratios) + 1))
        cumulative_variance = []
        running_sum = 0.0
        for ratio in explained_variance_ratios:
            running_sum += float(ratio)
            cumulative_variance.append(min(running_sum, 1.0))

        figure = go.Figure()

        figure.add_trace(go.Bar(
            name="Per-component variance",
            x=component_indices,
            y=[float(ratio) for ratio in explained_variance_ratios],
            marker_color="#1f77b4",
            opacity=0.7,
            hovertemplate="Component %{x}<br>Variance: %{y:.4f}<extra></extra>",
        ))

        figure.add_trace(go.Scatter(
            name="Cumulative variance",
            x=component_indices,
            y=cumulative_variance,
            mode="lines+markers",
            line=dict(color="#d62728", width=2),
            marker=dict(size=5),
            hovertemplate="Component %{x}<br>Cumulative: %{y:.4f}<extra></extra>",
        ))

        shapes = []
        annotations = []

        if n_components_at_threshold > 0:
            shapes.append(dict(
                type="line",
                x0=n_components_at_threshold + 0.5,
                x1=n_components_at_threshold + 0.5,
                y0=0, y1=1,
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            ))
            annotations.append(dict(
                text=f"{n_components_at_threshold} components retain {variance_threshold * 100:.1f}% variance",
                x=n_components_at_threshold + 0.5,
                y=variance_threshold,
                showarrow=True,
                arrowhead=2,
                font=dict(size=11, color="#ff7f0e"),
            ))

        shapes.append(dict(
            type="line",
            x0=0, x1=len(component_indices),
            y0=variance_threshold, y1=variance_threshold,
            line=dict(color="#9467bd", width=1, dash="dot"),
        ))

        figure.update_layout(
            title="PCA Explained Variance (Scree Plot)",
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance Ratio",
            yaxis=dict(range=[0, 1.05]),
            shapes=shapes,
            annotations=annotations,
            height=450,
            margin=dict(l=60, r=40, t=60, b=60),
            legend=dict(orientation="h", y=1.05),
        )
        return figure
