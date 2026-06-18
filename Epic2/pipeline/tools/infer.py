"""SemanticTypeInfer — heuristic type assignment, no LLM call.

Per-column algorithm:
  1. nunique == 2                        → binary
  2. pd.to_numeric succeeds (≥90% rows)  → numeric
  3. pd.to_datetime succeeds (≥90% rows) → datetime  (decomposed to year/month/day/quarter)
  4. fallthrough                          → categorical
"""
from __future__ import annotations

import pandas as pd

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.parallel import compute_mini_profile
from pipeline.state import PipelineState

_CONVERT_THRESHOLD = 0.90  # fraction of non-null values that must convert


def _decompose_datetime(df: pd.DataFrame, col: str) -> list[str]:
    parsed = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"] = parsed.dt.year
    df[f"{col}_month"] = parsed.dt.month
    df[f"{col}_day"] = parsed.dt.day
    df[f"{col}_quarter"] = parsed.dt.quarter
    new_cols = [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_quarter"]
    df.drop(columns=[col], inplace=True)
    return new_cols


def _infer_type(series: pd.Series) -> str:
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return "categorical"

    if series.nunique(dropna=True) == 2:
        return "binary"

    # Try numeric
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().sum() / n >= _CONVERT_THRESHOLD:
        return "numeric"

    # Try datetime
    try:
        dt = pd.to_datetime(non_null, errors="coerce")
        if dt.notna().sum() / n >= _CONVERT_THRESHOLD:
            return "datetime"
    except Exception:
        pass

    return "categorical"


class SemanticTypeInfer(BaseTool):
    def __init__(self, model_call=None):
        # model_call kept in signature for API compatibility; not used
        pass

    def precondition(self, state: PipelineState) -> None:
        if state.profile is None:
            raise PreconditionError("SemanticTypeInfer: state.profile is None")

    def run(self, state: PipelineState) -> None:
        df = state.df

        column_types: dict[str, str] = {}
        for col in df.columns:
            column_types[col] = _infer_type(df[col])

        column_types[state.target_column] = "target"

        new_datetime_cols: list[str] = []
        for col in list(column_types.keys()):
            if column_types[col] == "datetime" and col != state.target_column:
                new_cols = _decompose_datetime(df, col)
                new_datetime_cols.extend(new_cols)
                column_types.pop(col)
                for nc in new_cols:
                    column_types[nc] = "numeric"

        if new_datetime_cols:
            mini = compute_mini_profile(df, new_datetime_cols)
            for nc, mp in mini.items():
                mp["mi_with_target"] = 0.0
                mp["null_mask_corr"] = {}
                state.profile[nc] = mp

        state.column_types = column_types
        state.last_llm_source = "heuristic"

    def postcondition(self, state: PipelineState) -> None:
        if state.column_types is None:
            raise PostconditionError("SemanticTypeInfer: state.column_types is None")

    def __call__(self, state: PipelineState) -> PipelineState:
        self.precondition(state)
        self.run(state)
        self.postcondition(state)
        return state
