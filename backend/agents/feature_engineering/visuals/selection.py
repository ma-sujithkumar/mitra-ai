"""Feature selection visualization: MI bar chart colored by keep/drop with LLM rationale."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from backend.agents.feature_engineering.visuals.artifact_reader import ArtifactReader
from backend.agents.feature_engineering.visuals.base import BaseVisualizer

COLOR_SELECTED = "#2ca02c"
COLOR_DROPPED = "#d62728"
RATIONALE_MAX_DISPLAY_CHARS = 300


class SelectionRationaleVisualizer(BaseVisualizer):
    """Horizontal bar chart of MI scores colored green=kept / red=dropped, with LLM rationale."""

    def build(self) -> go.Figure | None:
        mi_scores = self.reader.mi_scores
        if not mi_scores:
            return None

        selected_set = set(self.reader.selected_columns)
        all_feature_cols = list(mi_scores.keys())
        sorted_features = sorted(all_feature_cols, key=lambda col: mi_scores[col], reverse=True)

        bar_colors = [COLOR_SELECTED if col in selected_set else COLOR_DROPPED for col in sorted_features]
        hover_texts = [
            f"<b>{col}</b><br>MI: {mi_scores[col]:.4f}<br>Status: {'KEPT' if col in selected_set else 'DROPPED'}"
            for col in sorted_features
        ]

        figure = go.Figure()
        figure.add_trace(go.Bar(
            name="MI Score",
            x=[mi_scores[col] for col in sorted_features],
            y=sorted_features,
            orientation="h",
            marker_color=bar_colors,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))

        selection_method = self.reader.selection_method
        num_selected = len(selected_set)
        num_total = len(sorted_features)

        # Show LLM rationale as an annotation if available
        rationale = self.reader.selection_rationale
        annotations = []
        if rationale:
            display_rationale = rationale[:RATIONALE_MAX_DISPLAY_CHARS]
            if len(rationale) > RATIONALE_MAX_DISPLAY_CHARS:
                display_rationale += "..."
            annotations.append(dict(
                text=f"<b>LLM Rationale:</b> {display_rationale}",
                xref="paper", yref="paper",
                x=0, y=-0.12,
                showarrow=False,
                font=dict(size=10, color="#444"),
                align="left",
                xanchor="left",
            ))

        figure.update_layout(
            title=(
                f"Feature Selection - Method: {selection_method} "
                f"({num_selected} kept / {num_total - num_selected} dropped, green=kept, red=dropped)"
            ),
            xaxis_title="Mutual Information Score",
            yaxis_title="Feature",
            height=max(400, len(sorted_features) * 20 + 160),
            margin=dict(l=200, r=40, t=60, b=120 if rationale else 40),
            annotations=annotations,
        )
        return figure
