import json
import re
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.state import PipelineState

OUTLIER_PROMPT = """You are an outlier strategist. For each numeric column, pick exactly one detector and one action.

Detectors: iqr, zscore, isolation_forest
Actions:
- scale: apply RobustScaler (default for numeric).
- flag: add binary column `<col>_is_outlier`.
- drop_row: drop the offending rows + corresponding target rows (opt-in only).

Guidance:
- Data-entry errors (extreme values, low frequency) -> drop_row.
- Outliers correlated with target -> flag.
- Default numeric column -> scale.

Columns:
{column_summary}

Respond with ONLY a JSON array, no prose:
[{{"column": "<name>", "detector": "<detector>", "action": "<action>"}}, ...]
"""


def _extract_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array: {text[:200]}")
    return json.loads(m.group(0))


class OutlierHandler(BaseTool):
    def __init__(self, model_call: Callable[[str], str]):
        self.model_call = model_call

    def precondition(self, state: PipelineState) -> None:
        if state.df.isna().sum().sum() > 0:
            raise PreconditionError("OutlierHandler: nulls remain (imputer must run first)")

    def run(self, state: PipelineState) -> None:
        df = state.df
        cfg = state.config
        numeric_cols = [
            c for c in df.columns
            if state.column_types.get(c) == "numeric" and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            state.row_count_after_outlier = len(df)
            return

        summary_lines = []
        for col in numeric_cols:
            p = state.profile.get(col, {})
            summary_lines.append(
                f"- {col}: outlier_rate={p.get('outlier_rate')}, skewness={p.get('skewness')}, "
                f"kurtosis={p.get('kurtosis')}, mi_with_target={p.get('mi_with_target')}"
            )
        prompt = OUTLIER_PROMPT.format(column_summary="\n".join(summary_lines))
        try:
            response = self.model_call(prompt)
            decisions = _extract_json_array(response)
        except Exception as e:
            state.warnings.append(f"OutlierHandler model parse failed: {e}; defaulting to scale")
            decisions = [{"column": c, "detector": "iqr", "action": "scale"} for c in numeric_cols]

        decision_map = {
            d["column"]: (d.get("detector", "iqr"), d.get("action", cfg.outlier.default_action))
            for d in decisions if "column" in d
        }

        drop_indices: set[int] = set()
        for col in numeric_cols:
            detector, action = decision_map.get(col, ("iqr", cfg.outlier.default_action))
            mask = self._detect(df[col], detector, cfg)
            if action == "scale":
                scaler = RobustScaler()
                arr = df[[col]].to_numpy()
                df[col] = scaler.fit_transform(arr).flatten()
                state.transformers.append({
                    "step": "outlier_scale",
                    "column": col,
                    "strategy": "robust",
                    "center": float(scaler.center_[0]),
                    "scale": float(scaler.scale_[0]),
                })
            elif action == "flag":
                df[f"{col}_is_outlier"] = mask.astype(int)
                state.created_columns.append({
                    "name": f"{col}_is_outlier",
                    "operation": "outlier_flag",
                    "sources": [col],
                })
                state.column_types[f"{col}_is_outlier"] = "binary"
            elif action == "drop_row":
                drop_indices.update(df.index[mask].tolist())

        if drop_indices:
            keep_mask = ~df.index.isin(drop_indices)
            state.df = df[keep_mask].reset_index(drop=True)
            state.target = state.target[keep_mask].reset_index(drop=True)

        state.row_count_after_outlier = len(state.df)

    @staticmethod
    def _detect(series: pd.Series, detector: str, cfg) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if detector == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - cfg.outlier.iqr_multiplier * iqr, q3 + cfg.outlier.iqr_multiplier * iqr
            return (s < lo) | (s > hi)
        if detector == "zscore":
            mean, std = s.mean(), s.std()
            if std == 0 or pd.isna(std):
                return pd.Series([False] * len(s), index=s.index)
            z = ((s - mean) / std).abs()
            return z > cfg.outlier.zscore_threshold
        if detector == "isolation_forest":
            try:
                clf = IsolationForest(
                    contamination=cfg.outlier.isolation_contamination,
                    random_state=cfg.pipeline.random_state,
                )
                preds = clf.fit_predict(s.fillna(s.median()).to_numpy().reshape(-1, 1))
                return pd.Series(preds == -1, index=s.index)
            except Exception:
                return pd.Series([False] * len(s), index=s.index)
        return pd.Series([False] * len(s), index=s.index)

    def postcondition(self, state: PipelineState) -> None:
        if state.row_count_after_outlier is None:
            raise PostconditionError("OutlierHandler: row_count_after_outlier not set")
