"""Abstract base class for all feature engineering pipeline visualizers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import plotly.graph_objects as go

from backend.agents.feature_engineering.visuals.artifact_reader import ArtifactReader


# Shared cap applied across all visualizers so charts stay readable.
MAX_VISUAL_ROWS = 30


class BaseVisualizer(ABC):
    """Every concrete visualizer extends this class."""

    def __init__(self, reader: ArtifactReader, output_dir: Path) -> None:
        self.reader = reader
        self.output_dir = Path(output_dir)

    @abstractmethod
    def build(self) -> go.Figure | None:
        """Build and return the Plotly figure, or None if there is no data to show."""

    def save(self, filename: str) -> Path | None:
        """Build the figure and write a self-contained HTML file. Returns the path or None."""
        figure = self.build()
        if figure is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename
        figure.write_html(str(output_path), include_plotlyjs=True)
        return output_path
