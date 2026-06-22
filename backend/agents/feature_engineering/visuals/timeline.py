"""Pipeline step timeline: horizontal bar chart of step durations colored by status."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from backend.agents.feature_engineering.visuals.artifact_reader import ArtifactReader
from backend.agents.feature_engineering.visuals.base import BaseVisualizer

COLOR_OK = "#2ca02c"
COLOR_ERROR = "#d62728"

# Canonical pipeline step order for consistent y-axis ordering
PIPELINE_STEP_ORDER = [
    "profile_data",
    "infer_types",
    "handle_missing",
    "handle_outliers",
    "encode_features",
    "create_features",
    "scale_features",
    "compute_feature_stats",
    "select_features",
    "validate_features",
    "write_report",
]


class PipelineTimelineVisualizer(BaseVisualizer):
    """Horizontal Gantt-style bar chart of each pipeline step's duration."""

    def build(self) -> go.Figure | None:
        timeline_events = self.reader.timeline_events
        if not timeline_events:
            return None

        # Sort events by canonical pipeline order, then any unrecognized steps at the end
        def step_sort_key(event: dict) -> int:
            step_name = event.get("step", "")
            if step_name in PIPELINE_STEP_ORDER:
                return PIPELINE_STEP_ORDER.index(step_name)
            return len(PIPELINE_STEP_ORDER)

        sorted_events = sorted(timeline_events, key=step_sort_key)

        step_labels: list[str] = []
        elapsed_values: list[float] = []
        bar_colors: list[str] = []
        hover_texts: list[str] = []

        for event in sorted_events:
            step_name = event.get("step", "")
            elapsed_seconds = float(event.get("elapsed_s", 0.0))
            status = event.get("status", "ok")
            llm_source = event.get("llm_source", "")

            llm_label = f" | llm={llm_source}" if llm_source else ""
            hover_text = (
                f"<b>{step_name}</b><br>"
                f"Duration: {elapsed_seconds:.2f}s<br>"
                f"Status: {status}{llm_label}"
            )

            step_labels.append(step_name)
            elapsed_values.append(elapsed_seconds)
            bar_colors.append(COLOR_OK if status == "ok" else COLOR_ERROR)
            hover_texts.append(hover_text)

        figure = go.Figure(go.Bar(
            name="Duration",
            x=elapsed_values,
            y=step_labels,
            orientation="h",
            marker_color=bar_colors,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            textposition="none",
        ))

        total_duration = sum(elapsed_values)
        figure.update_layout(
            title=f"Pipeline Step Duration (total: {total_duration:.2f}s, green=ok, red=error)",
            xaxis_title="Time (seconds)",
            yaxis_title="Step",
            height=max(350, len(step_labels) * 35 + 100),
            margin=dict(l=180, r=40, t=60, b=60),
        )
        return figure
