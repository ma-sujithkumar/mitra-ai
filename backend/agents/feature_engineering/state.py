from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backend.agents.feature_engineering.config import ConfigSchema


@dataclass
class PipelineState:
    df: pd.DataFrame
    task: str
    run_id: str
    config: ConfigSchema

    # Both are None when task == "clustering" (no supervision signal)
    target: pd.Series | None = None
    target_column: str | None = None

    profile: dict[str, Any] | None = None
    column_types: dict[str, str] | None = None

    transformers: list[dict] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    created_columns: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    selected_columns: list[str] | None = None
    selection_method: str | None = None

    output_dir: Path | None = None
    stats_dir: Path | None = None

    pre_encoding_done: bool = False
    row_count_after_outlier: int | None = None

    last_llm_source: str | None = None
