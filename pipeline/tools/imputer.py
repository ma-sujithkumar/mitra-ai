import json
import re
from typing import Callable

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.state import PipelineState

IMPUTE_PROMPT = """You are an imputation strategist. For each column with nulls, pick exactly one strategy from:
median, mode, knn, iterative, drop.

Guidance:
- median: numeric, low null rate, MCAR (no strong null-mask correlations).
- mode: categorical/binary, low null rate.
- knn: nulls correlate with other columns (surrounding structure carries signal).
- iterative: numeric, mid null rate, multivariate relationships present.
- drop: null rate above {drop_threshold}.

Columns with nulls:
{column_summary}

Respond with ONLY a JSON array, no prose:
[{{"column": "<name>", "strategy": "<strategy>"}}, ...]
"""


def _extract_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array in response: {text[:200]}")
    return json.loads(m.group(0))


class MissingValueHandler(BaseTool):
    def __init__(self, model_call: Callable[[str], str]):
        self.model_call = model_call

    def precondition(self, state: PipelineState) -> None:
        if state.column_types is None:
            raise PreconditionError("MissingValueHandler: state.column_types is None")

    def run(self, state: PipelineState) -> None:
        df = state.df
        cfg = state.config
        drop_threshold = cfg.imputation.null_drop_threshold

        # Handle target separately
        if state.target.isna().any():
            if state.task == "classification":
                fill = state.target.mode().iloc[0]
            else:
                fill = state.target.median()
            state.target = state.target.fillna(fill)
            state.transformers.append(
                {"step": "imputation", "column": state.target_column, "strategy": "target_fill", "fill_value": _safe(fill)}
            )

        cols_with_nulls = [c for c in df.columns if df[c].isna().any()]
        if not cols_with_nulls:
            return

        summary_lines = []
        for col in cols_with_nulls:
            p = state.profile.get(col, {})
            summary_lines.append(
                f"- {col}: type={state.column_types.get(col)}, null_rate={p.get('null_rate'):.3f}, "
                f"null_mask_corr={p.get('null_mask_corr', {})}"
            )
        prompt = IMPUTE_PROMPT.format(
            drop_threshold=drop_threshold, column_summary="\n".join(summary_lines)
        )
        try:
            response = self.model_call(prompt)
            decisions = _extract_json_array(response)
        except Exception as e:
            state.warnings.append(f"MissingValueHandler model parse failed: {e}; using median/mode fallback")
            decisions = []
            for col in cols_with_nulls:
                t = state.column_types.get(col, "numeric")
                decisions.append({"column": col, "strategy": "median" if t == "numeric" else "mode"})

        decision_map = {d["column"]: d["strategy"] for d in decisions if "column" in d and "strategy" in d}

        # Drop high-null columns regardless of model output
        to_drop_high_null = [
            c for c in cols_with_nulls if state.profile[c]["null_rate"] > drop_threshold
        ]
        for col in to_drop_high_null:
            decision_map[col] = "drop"

        # Apply strategies
        knn_cols: list[str] = []
        iterative_cols: list[str] = []
        for col in cols_with_nulls:
            strategy = decision_map.get(col, "median")
            if strategy == "drop":
                df.drop(columns=[col], inplace=True)
                state.dropped_columns.append(col)
                state.column_types.pop(col, None)
                state.warnings.append(f"{col} dropped (null_rate={state.profile[col]['null_rate']:.2f})")
                continue
            if strategy == "median":
                fill = pd.to_numeric(df[col], errors="coerce").median()
                df[col] = df[col].fillna(fill)
                state.transformers.append(
                    {"step": "imputation", "column": col, "strategy": "median", "fill_value": _safe(fill)}
                )
            elif strategy == "mode":
                mode_vals = df[col].mode(dropna=True)
                fill = mode_vals.iloc[0] if not mode_vals.empty else 0
                df[col] = df[col].fillna(fill)
                state.transformers.append(
                    {"step": "imputation", "column": col, "strategy": "mode", "fill_value": _safe(fill)}
                )
            elif strategy == "knn":
                knn_cols.append(col)
            elif strategy == "iterative":
                iterative_cols.append(col)
            else:
                fill = pd.to_numeric(df[col], errors="coerce").median()
                df[col] = df[col].fillna(fill)
                state.transformers.append(
                    {"step": "imputation", "column": col, "strategy": "median", "fill_value": _safe(fill)}
                )

        if knn_cols:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            target_cols = [c for c in knn_cols if c in numeric_cols]
            if target_cols:
                imputer = KNNImputer(n_neighbors=cfg.imputation.knn_neighbors)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                for col in target_cols:
                    state.transformers.append(
                        {"step": "imputation", "column": col, "strategy": "knn", "n_neighbors": cfg.imputation.knn_neighbors}
                    )

        if iterative_cols:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            target_cols = [c for c in iterative_cols if c in numeric_cols]
            if target_cols:
                imputer = IterativeImputer(
                    max_iter=cfg.imputation.iterative_max_iter, random_state=cfg.pipeline.random_state
                )
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                for col in target_cols:
                    state.transformers.append(
                        {"step": "imputation", "column": col, "strategy": "iterative", "random_state": cfg.pipeline.random_state}
                    )

        # Final pass: any leftover nulls (non-numeric KNN/iterative targets) get mode
        for col in df.columns:
            if df[col].isna().any():
                mode_vals = df[col].mode(dropna=True)
                fill = mode_vals.iloc[0] if not mode_vals.empty else 0
                df[col] = df[col].fillna(fill)
                state.transformers.append(
                    {"step": "imputation", "column": col, "strategy": "mode_fallback", "fill_value": _safe(fill)}
                )

    def postcondition(self, state: PipelineState) -> None:
        nulls = state.df.isna().sum().sum()
        if nulls > 0:
            raise PostconditionError(f"MissingValueHandler: {nulls} nulls remain")


def _safe(value) -> object:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
