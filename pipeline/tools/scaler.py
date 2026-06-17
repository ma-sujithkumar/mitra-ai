"""Scaler — pick a scaler per numeric feature column.

Typed ScalerEvidence + Pydantic ScalerResponse with the four content checks,
one revision, fall-through to standard.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.evidence import ScalerColumnEvidence, ScalerEvidence, render
from pipeline.responses import ScalerResponse, call_with_revision
from pipeline.state import PipelineState

STRATEGY_DEFINITIONS: dict[str, str] = {
    "standard": "Centres by mean, scales by standard deviation. Output ~ N(0, 1) for normal inputs.",
    "robust": "Centres by median, scales by interquartile range. Insensitive to extreme values.",
    "minmax": "Linearly rescales to [0, 1] using observed minimum and maximum.",
    "power": "Applies Yeo-Johnson power transform to symmetrise heavy-tailed distributions.",
}

SCALER_PROMPT = """For each numeric feature column pick exactly one scaler.

## STRATEGY DEFINITIONS (mechanical descriptions only)
{strategy_definitions}

## RESPONSE SHAPE
Return ONLY a JSON object of this shape:
{{
  "decisions": [
    {{
      "column": "<name>",
      "scaler": "<standard|robust|minmax|power>",
      "rationale": "<at least {min_rationale_chars} characters citing fields from EVIDENCE>",
      "evidence_cited": ["<field paths from EVIDENCE you used>"],
      "alternatives_considered": ["<other scalers you weighed>"]
    }}
  ]
}}

{evidence_block}
"""


def _strategy_definitions_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in STRATEGY_DEFINITIONS.items())


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
        outlier_scaled = {t["column"] for t in state.transformers if t.get("step") == "outlier_scale"}
        scale_targets = [
            c for c in feature_cols
            if state.column_types.get(c) == "numeric"
            and pd.api.types.is_numeric_dtype(df[c])
            and c not in outlier_scaled
        ]
        if not scale_targets:
            return

        per_col: list[ScalerColumnEvidence] = []
        for col in scale_targets:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                hist = [0] * 20
                bounded = False
                bounds = None
            else:
                counts, _ = np.histogram(series.to_numpy(), bins=20)
                hist = [int(c) for c in counts.tolist()]
                lo, hi = float(series.min()), float(series.max())
                bounded = bool(np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > 0)
                bounds = (lo, hi) if bounded else None
            p = state.profile.get(col, {})
            try:
                mono = float(series.corr(pd.to_numeric(state.target, errors="coerce"), method="spearman"))
                if np.isnan(mono):
                    mono = 0.0
            except Exception:
                mono = 0.0
            per_col.append(
                ScalerColumnEvidence(
                    name=col,
                    histogram_20bin=hist,
                    skewness=float(p.get("skewness") or 0.0),
                    kurtosis=float(p.get("kurtosis") or 0.0),
                    outlier_rate=float(p.get("outlier_rate") or 0.0),
                    bounded=bounded,
                    bounds=bounds,
                    monotonic_with_target=mono,
                )
            )

        packet = ScalerEvidence(columns=per_col)
        evidence_block, sent_fields = render(packet, truncate_after_chars=int(cfg.llm.max_tokens * 0.7 * 4))

        prompt = SCALER_PROMPT.format(
            strategy_definitions=_strategy_definitions_block(),
            min_rationale_chars=cfg.validation.min_rationale_chars,
            evidence_block=evidence_block,
        )

        parsed, source, failures = call_with_revision(
            self.model_call, prompt, ScalerResponse, sent_fields, cfg,
            caller="Scaler",
        )

        if parsed is None or source == "fallback":
            state.warnings.append(
                f"Scaler fell through to standard (failures={failures})"
            )
            decision_map = {c: "standard" for c in scale_targets}
        else:
            decision_map = {d.column: d.scaler for d in parsed.decisions}  # type: ignore[attr-defined]
            for c in scale_targets:
                decision_map.setdefault(c, "standard")

        for col in scale_targets:
            scaler_name = decision_map[col]
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
