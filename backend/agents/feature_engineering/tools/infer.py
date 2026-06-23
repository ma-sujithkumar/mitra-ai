"""SemanticTypeInfer — heuristic type assignment, no LLM call.

Per-column algorithm:
  1. nunique / n_rows >= id_uniqueness_threshold  → identifier (dropped immediately)
  2. sorted unique values are a contiguous int range with step 1 → identifier (dropped)
  3. nunique == 2                                 → binary
  4. pd.to_numeric succeeds (>=90% rows)          → numeric
  5. pd.to_datetime succeeds (>=90% rows)         → datetime (decomposed to year/month/day/quarter)
  6. fallthrough                                  → categorical
"""
from __future__ import annotations

import pandas as pd

from backend.agents.feature_engineering.base import BaseTool, PostconditionError, PreconditionError
from backend.agents.feature_engineering.parallel import compute_mini_profile
from backend.agents.feature_engineering.state import PipelineState

_CONVERT_THRESHOLD = 0.90  # fraction of non-null values that must convert


def _is_identifier_column(series: pd.Series, uniqueness_threshold: float) -> bool:
    """Return True if the column looks like an identifier that carries no predictive signal.

    Two order-independent signals:
    1. Uniqueness ratio >= threshold: nearly every value is unique (ID, UUID, hash, email).
    2. Contiguous integer sequence: sorted unique values step by exactly 1 (auto-increment PK / row index).
    """
    n_total = len(series)
    if n_total == 0:
        return False

    uniqueness_ratio = series.nunique(dropna=False) / n_total
    if uniqueness_ratio >= uniqueness_threshold:
        return True

    # Contiguous integer sequence check — sort first so shuffled data is handled correctly.
    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().all() and n_total > 1:
        all_integer = (numeric_values % 1 == 0).all()
        if all_integer:
            sorted_values = numeric_values.sort_values().reset_index(drop=True)
            diffs = sorted_values.diff().dropna()
            if len(diffs) > 0 and (diffs == 1).all():
                return True

    return False


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
        uniqueness_threshold = state.config.feature_selection.id_uniqueness_threshold

        # Drop identifier columns before type assignment so they never reach
        # encoding, feature creation, stats computation, or the LLM selector.
        identifier_columns = [
            col for col in df.columns
            if col != state.target_column and _is_identifier_column(df[col], uniqueness_threshold)
        ]
        if identifier_columns:
            df.drop(columns=identifier_columns, inplace=True)
            state.dropped_columns.extend(identifier_columns)
            for col in identifier_columns:
                state.warnings.append(
                    f"Dropped identifier column '{col}' "
                    f"(uniqueness_ratio >= {uniqueness_threshold} or contiguous integer sequence)"
                )

        column_types: dict[str, str] = {}
        for col in df.columns:
            column_types[col] = _infer_type(df[col])

        # Unsupervised runs have no target column to tag.
        if state.target_column is not None:
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
