import json
import re
from typing import Callable

import pandas as pd

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.parallel import compute_mini_profile
from pipeline.state import PipelineState

VALID_TYPES = {"numeric", "categorical", "datetime", "id", "text", "binary", "target"}

INFER_PROMPT = """You are a data type inference engine. For each column below, assign exactly one type from:
numeric, categorical, datetime, id, text, binary, target.

Rules:
- `id`: primary keys, UUIDs, row indices, unique identifiers with no signal.
- `binary`: exactly two unique values.
- `datetime`: date/time strings or datetime dtype.
- `numeric`: continuous or count numeric.
- `categorical`: low-cardinality discrete (non-numeric or numeric codes).
- `text`: free-form long text.
- `target`: reserved — DO NOT use; the target column is set by the caller.

Columns:
{column_summary}

Respond with ONLY a JSON array, no prose:
[{{"column": "<name>", "type": "<type>"}}, ...]
"""


def _extract_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array in model response: {text[:200]}")
    return json.loads(m.group(0))


def _decompose_datetime(df: pd.DataFrame, col: str) -> list[str]:
    parsed = pd.to_datetime(df[col], errors="coerce")
    new_cols = []
    df[f"{col}_year"] = parsed.dt.year
    df[f"{col}_month"] = parsed.dt.month
    df[f"{col}_day"] = parsed.dt.day
    df[f"{col}_quarter"] = parsed.dt.quarter
    new_cols = [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_quarter"]
    df.drop(columns=[col], inplace=True)
    return new_cols


class SemanticTypeInfer(BaseTool):
    def __init__(self, model_call: Callable[[str], str]):
        self.model_call = model_call

    def precondition(self, state: PipelineState) -> None:
        if state.profile is None:
            raise PreconditionError("SemanticTypeInfer: state.profile is None")

    def run(self, state: PipelineState) -> None:
        df = state.df
        summary_lines = []
        for col in df.columns:
            p = state.profile.get(col, {})
            summary_lines.append(
                f"- {col}: dtype={p.get('dtype')}, null_rate={p.get('null_rate'):.3f}, "
                f"nunique={p.get('nunique')}, top_values={p.get('top_values')}"
            )
        prompt = INFER_PROMPT.format(column_summary="\n".join(summary_lines))

        response = self.model_call(prompt)
        try:
            assignments = _extract_json_array(response)
        except Exception as e:
            state.warnings.append(f"SemanticTypeInfer parse failed: {e}; falling back to dtype inference")
            assignments = [{"column": c, "type": self._dtype_fallback(df[c])} for c in df.columns]

        column_types: dict[str, str] = {}
        for item in assignments:
            col, typ = item.get("column"), item.get("type")
            if col in df.columns and typ in VALID_TYPES:
                column_types[col] = typ
        for col in df.columns:
            if col not in column_types:
                column_types[col] = self._dtype_fallback(df[col])

        column_types[state.target_column] = "target"

        cols_to_drop = [c for c, t in column_types.items() if t in {"id", "text"} and c != state.target_column]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            state.dropped_columns.extend(cols_to_drop)
            for c in cols_to_drop:
                column_types.pop(c, None)

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
            for nc, p in mini.items():
                p["mi_with_target"] = 0.0
                p["null_mask_corr"] = {}
                state.profile[nc] = p

        state.column_types = column_types

    @staticmethod
    def _dtype_fallback(series: pd.Series) -> str:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        nunique = series.nunique(dropna=True)
        if nunique == 2:
            return "binary"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        return "categorical"

    def postcondition(self, state: PipelineState) -> None:
        if state.column_types is None:
            raise PostconditionError("SemanticTypeInfer: state.column_types is None")
