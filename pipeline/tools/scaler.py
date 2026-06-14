import json
import re
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.state import PipelineState

SCALER_PROMPT = """You are a scaling strategist. For each numeric feature column, pick exactly one scaler from:
standard, robust, minmax, power.

Guidance:
- standard: near-normal (low skew/kurtosis).
- robust: skewed or has outliers.
- minmax: bounded ranges or tree-of-bounds features.
- power: heavily skewed (|skew|>1).

Columns:
{column_summary}

Respond with ONLY a JSON array, no prose:
[{{"column": "<name>", "scaler": "<scaler>"}}, ...]
"""


def _extract_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array: {text[:200]}")
    return json.loads(m.group(0))


class Scaler(BaseTool):
    def __init__(self, model_call: Callable[[str], str]):
        self.model_call = model_call

    def precondition(self, state: PipelineState) -> None:
        non_numeric = [c for c in state.df.columns if not pd.api.types.is_numeric_dtype(state.df[c])]
        if non_numeric:
            raise PreconditionError(f"Scaler: non-numeric columns present: {non_numeric}")

    def run(self, state: PipelineState) -> None:
        df = state.df
        cfg = state.config
        feature_cols = [c for c in df.columns if c != state.target_column]
        scale_targets = [
            c for c in feature_cols
            if state.column_types.get(c) == "numeric" and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not scale_targets:
            return

        summary_lines = []
        for col in scale_targets:
            p = state.profile.get(col, {})
            summary_lines.append(
                f"- {col}: skewness={p.get('skewness')}, kurtosis={p.get('kurtosis')}, outlier_rate={p.get('outlier_rate')}"
            )
        prompt = SCALER_PROMPT.format(column_summary="\n".join(summary_lines))
        try:
            response = self.model_call(prompt)
            decisions = _extract_json_array(response)
        except Exception as e:
            state.warnings.append(f"Scaler parse failed: {e}; defaulting to standard")
            decisions = [{"column": c, "scaler": "standard"} for c in scale_targets]

        decision_map = {d["column"]: d.get("scaler", "standard") for d in decisions if "column" in d}

        for col in scale_targets:
            scaler_name = decision_map.get(col, "standard")
            arr = df[[col]].to_numpy()
            scaler, params = self._fit(scaler_name, arr, cfg)
            df[col] = scaler.transform(arr).flatten()
            state.transformers.append({
                "step": "scaling",
                "column": col,
                "strategy": scaler_name,
                **params,
            })

    @staticmethod
    def _fit(name: str, arr: np.ndarray, cfg) -> tuple:
        if name == "standard":
            s = StandardScaler().fit(arr)
            return s, {"mean": float(s.mean_[0]), "std": float(s.scale_[0])}
        if name == "robust":
            s = RobustScaler().fit(arr)
            return s, {"center": float(s.center_[0]), "scale": float(s.scale_[0])}
        if name == "minmax":
            s = MinMaxScaler().fit(arr)
            return s, {"data_min": float(s.data_min_[0]), "data_max": float(s.data_max_[0])}
        if name == "power":
            s = PowerTransformer(method=cfg.scaling.power_transformer_method).fit(arr)
            return s, {"lambdas": [float(x) for x in s.lambdas_.tolist()]}
        s = StandardScaler().fit(arr)
        return s, {"mean": float(s.mean_[0]), "std": float(s.scale_[0])}

    def postcondition(self, state: PipelineState) -> None:
        pass
