from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import ConfigSchema


@dataclass
class PipelineState:
    df: pd.DataFrame
    target: pd.Series
    task: str
    target_column: str
    run_id: str
    config: ConfigSchema

    profile: dict[str, Any] | None = None
    column_types: dict[str, str] | None = None

    transformers: list[dict] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    created_columns: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    selected_columns: list[str] | None = None
    selection_method: str | None = None

    output_dir: Path | None = None

    pre_encoding_done: bool = False
    row_count_after_outlier: int | None = None

    last_llm_source: str | None = None
