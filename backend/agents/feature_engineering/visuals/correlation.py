"""Correlation heatmap reordered by cluster membership with cluster boundary overlays.

Provides a toggle button to switch between Pearson and Spearman heatmaps. Both share
the same feature ordering (union of high-correlation pair features, ordered by cluster
membership) so the cluster boundaries remain valid in both views.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from backend.agents.feature_engineering.visuals.artifact_reader import ArtifactReader
from backend.agents.feature_engineering.visuals.base import BaseVisualizer, MAX_VISUAL_ROWS

MAX_FEATURES_IN_HEATMAP = MAX_VISUAL_ROWS


def _build_correlation_matrix(
    all_features: list[str],
    pairs: list[list],
) -> np.ndarray:
    """Build a square NxN correlation matrix from a list of (col_a, col_b, corr) triples."""
    feature_index = {feature: idx for idx, feature in enumerate(all_features)}
    num_features = len(all_features)
    corr_matrix = np.zeros((num_features, num_features))
    np.fill_diagonal(corr_matrix, 1.0)
    for triple in pairs:
        col_a, col_b, corr_value = triple[0], triple[1], float(triple[2])
        idx_a = feature_index.get(col_a)
        idx_b = feature_index.get(col_b)
        if idx_a is not None and idx_b is not None:
            corr_matrix[idx_a][idx_b] = corr_value
            corr_matrix[idx_b][idx_a] = corr_value
    return corr_matrix


class CorrelationClusterVisualizer(BaseVisualizer):
    """Square Pearson/Spearman correlation heatmap with cluster-ordered axes and boundary rectangles."""

    def build(self) -> go.Figure | None:
        pearson_pairs = self.reader.pearson_pairs
        spearman_pairs = self.reader.spearman_pairs

        # Need at least one correlation method to have data
        if not pearson_pairs and not spearman_pairs:
            return None

        clusters = self.reader.clusters
        selected_set = set(self.reader.selected_columns)
        mi_scores = self.reader.mi_scores

        # Collect features from BOTH methods so both heatmaps share the same feature set
        pair_features: set[str] = set()
        for triple in pearson_pairs:
            pair_features.add(triple[0])
            pair_features.add(triple[1])
        for triple in spearman_pairs:
            pair_features.add(triple[0])
            pair_features.add(triple[1])

        # Limit to top N features by MI score to keep heatmap readable
        if len(pair_features) > MAX_FEATURES_IN_HEATMAP:
            top_features = sorted(pair_features, key=lambda col: mi_scores.get(col, 0), reverse=True)
            pair_features = set(top_features[:MAX_FEATURES_IN_HEATMAP])
            pearson_pairs = [
                triple for triple in pearson_pairs
                if triple[0] in pair_features and triple[1] in pair_features
            ]
            spearman_pairs = [
                triple for triple in spearman_pairs
                if triple[0] in pair_features and triple[1] in pair_features
            ]

        # Order features by cluster membership (same ordering for both views)
        cluster_ordered: list[str] = []
        cluster_boundary_positions: list[int] = []
        for cluster_id in sorted(clusters.keys(), key=lambda cid: int(cid) if cid.isdigit() else 0):
            cluster_members = [col for col in clusters[cluster_id] if col in pair_features]
            if cluster_members:
                cluster_boundary_positions.append(len(cluster_ordered))
                cluster_ordered.extend(cluster_members)

        remaining_features = [col for col in pair_features if col not in set(cluster_ordered)]
        cluster_ordered.extend(sorted(remaining_features))

        pearson_matrix = _build_correlation_matrix(cluster_ordered, pearson_pairs)
        spearman_matrix = _build_correlation_matrix(cluster_ordered, spearman_pairs)

        def _hover_texts(corr_matrix: np.ndarray, method_label: str) -> list[list[str]]:
            hover = []
            for row_idx, row_col in enumerate(cluster_ordered):
                hover_row = []
                for col_idx, col_col in enumerate(cluster_ordered):
                    corr_val = corr_matrix[row_idx][col_idx]
                    row_status = "selected" if row_col in selected_set else "dropped"
                    col_status = "selected" if col_col in selected_set else "dropped"
                    hover_row.append(
                        f"{row_col} ({row_status})<br>{col_col} ({col_status})<br>"
                        f"{method_label} r = {corr_val:.3f}"
                    )
                hover.append(hover_row)
            return hover

        pearson_hover = _hover_texts(pearson_matrix, "Pearson")
        spearman_hover = _hover_texts(spearman_matrix, "Spearman")

        figure = go.Figure()

        # Pearson heatmap (default visible)
        figure.add_trace(go.Heatmap(
            name="Pearson",
            z=pearson_matrix,
            x=cluster_ordered,
            y=cluster_ordered,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=pearson_hover,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="r"),
            visible=True,
        ))

        # Spearman heatmap (hidden by default, toggled via button)
        figure.add_trace(go.Heatmap(
            name="Spearman",
            z=spearman_matrix,
            x=cluster_ordered,
            y=cluster_ordered,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=spearman_hover,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="r"),
            visible=False,
        ))

        # Draw cluster boundary lines (shared across both views)
        num_features = len(cluster_ordered)
        shapes = []
        for boundary_start in cluster_boundary_positions:
            if boundary_start == 0:
                continue
            shapes.append(dict(
                type="line",
                x0=boundary_start - 0.5,
                x1=boundary_start - 0.5,
                y0=-0.5,
                y1=num_features - 0.5,
                line=dict(color="#333", width=2),
            ))
            shapes.append(dict(
                type="line",
                x0=-0.5,
                x1=num_features - 0.5,
                y0=boundary_start - 0.5,
                y1=boundary_start - 0.5,
                line=dict(color="#333", width=2),
            ))

        figure.update_layout(
            title="Feature Correlation by Cluster - Pearson (RdBu: red=positive, blue=negative)",
            height=max(500, num_features * 18 + 120),
            width=max(600, num_features * 18 + 200),
            shapes=shapes,
            margin=dict(l=150, r=40, t=80, b=150),
            xaxis=dict(tickangle=45),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.0,
                    y=1.08,
                    xanchor="left",
                    showactive=True,
                    buttons=[
                        dict(
                            label="Pearson",
                            method="update",
                            args=[
                                {"visible": [True, False]},
                                {"title": "Feature Correlation by Cluster - Pearson (RdBu: red=positive, blue=negative)"},
                            ],
                        ),
                        dict(
                            label="Spearman",
                            method="update",
                            args=[
                                {"visible": [False, True]},
                                {"title": "Feature Correlation by Cluster - Spearman (RdBu: red=positive, blue=negative)"},
                            ],
                        ),
                    ],
                )
            ],
        )
        return figure
