"""OutlierHandler — pick (detector, action) per numeric column.

Typed OutlierEvidence + Pydantic OutlierResponse with the four content checks,
one revision, fall-through to (iqr, scale).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.evidence import OutlierColumnEvidence, OutlierEvidence, render
from pipeline.responses import OutlierResponse, call_with_revision
from pipeline.state import PipelineState

STRATEGY_DEFINITIONS: dict[str, str] = {
    "iqr": "Detector. Flags rows where the value falls outside Q1-1.5·IQR or above Q3+1.5·IQR.",
    "zscore": "Detector. Flags rows whose standard-score |z| exceeds the configured threshold.",
    "isolation_forest": "Detector. Fits an isolation-forest and flags rows scored as anomalous.",
    "scale": "Action. Applies RobustScaler to the column (centres by median, scales by IQR).",
    "flag": "Action. Adds a binary indicator column `<col>_is_outlier` and keeps the original.",
    "drop_row": "Action. Drops the offending rows and their matching target values atomically.",
}

OUTLIER_PROMPT = """For each numeric column pick exactly one action and, when relevant, one detector.

## STRATEGY DEFINITIONS (mechanical descriptions only)
{strategy_definitions}

## RULES
- `detector` is REQUIRED when action is `flag` or `drop_row` — the detector's row mask is what gets flagged or dropped.
- `detector` is OPTIONAL (omit the field) when action is `scale` — the whole column is RobustScaled regardless of which rows would have been flagged.

## RESPONSE SHAPE
Return ONLY a JSON object of this shape:
{{
  "decisions": [
    {{
      "column": "<name>",
      "detector": "<iqr|zscore|isolation_forest>",
      "action": "<scale|flag|drop_row>",
      "rationale": "<at least {min_rationale_chars} characters citing fields from EVIDENCE>",
      "evidence_cited": ["<field paths from EVIDENCE you used>"],
      "alternatives_considered": ["<other detector/action pairs you weighed>"]
    }}
  ]
}}

{evidence_block}
"""


def _strategy_definitions_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in STRATEGY_DEFINITIONS.items())


def _extreme_pairs(series: pd.Series, target: pd.Series, k: int, ascending: bool) -> list[tuple[float, object]]:
    aligned = pd.concat([series.rename("v"), target.rename("t")], axis=1).dropna(subset=["v"])
    aligned = aligned.sort_values("v", ascending=ascending).head(k)
    return [(float(v), t.item() if hasattr(t, "item") else t) for v, t in zip(aligned["v"], aligned["t"])]


class OutlierHandler(BaseTool):
    def __init__(self, model_call: Callable[[str], str] | None):
        # model_call is None => deterministic mode (no LLM): (iqr, default_action)
        # for every numeric column, matching the LLM fallback policy.
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

        # Deterministic mode: rule-based (iqr detector, configured default action).
        if self.model_call is None:
            state.last_llm_source = "deterministic"
            decision_map = {c: ("iqr", cfg.outlier.default_action) for c in numeric_cols}
            self._apply_decisions(df, state, numeric_cols, decision_map)
            return

        per_col: list[OutlierColumnEvidence] = []
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            clean = series.dropna()
            if clean.empty:
                hist = [0] * 10
            else:
                counts, _ = np.histogram(clean.to_numpy(), bins=10)
                hist = [int(c) for c in counts.tolist()]
            mi = state.profile.get(col, {}).get("mi_with_target") or 0.0
            try:
                tcorr = float(series.corr(pd.to_numeric(state.target, errors="coerce")))
                if np.isnan(tcorr):
                    tcorr = 0.0
            except Exception:
                tcorr = 0.0
            per_col.append(
                OutlierColumnEvidence(
                    name=col,
                    histogram_10bin=hist,
                    extreme_top5=_extreme_pairs(series, state.target, 5, ascending=False),
                    extreme_bottom5=_extreme_pairs(series, state.target, 5, ascending=True),
                    mi_with_target=float(mi),
                    target_corr=tcorr,
                )
            )

        packet = OutlierEvidence(
            columns=per_col,
            downstream_model_hint=cfg.pipeline.downstream_model_hint,
        )
        evidence_block, sent_fields = render(packet, truncate_after_chars=int(cfg.llm.max_tokens * 0.7 * 4))

        prompt = OUTLIER_PROMPT.format(
            strategy_definitions=_strategy_definitions_block(),
            min_rationale_chars=cfg.validation.min_rationale_chars,
            evidence_block=evidence_block,
        )

        parsed, source, failures = call_with_revision(
            self.model_call, prompt, OutlierResponse, sent_fields, cfg,
            caller="OutlierHandler",
        )
        state.last_llm_source = source

        if parsed is None or source == "fallback":
            state.warnings.append(
                f"OutlierHandler fell through to (iqr, scale) (failures={failures})"
            )
            decision_map = {c: ("iqr", cfg.outlier.default_action) for c in numeric_cols}
        else:
            decision_map = {
                d.column: (d.detector, d.action) for d in parsed.decisions  # type: ignore[attr-defined]
            }
            for c in numeric_cols:
                decision_map.setdefault(c, ("iqr", cfg.outlier.default_action))

        self._apply_decisions(df, state, numeric_cols, decision_map)

    def _apply_decisions(
        self,
        df: pd.DataFrame,
        state: PipelineState,
        numeric_cols: list[str],
        decision_map: dict,
    ) -> None:
        """Apply (detector, action) per column. Shared by deterministic and LLM modes."""
        cfg = state.config
        drop_indices: set[int] = set()
        for col in numeric_cols:
            detector, action = decision_map[col]
            # detector is optional for scale; coerce missing detector to a
            # default for the few code paths that still want a mask (none of
            # them act on the mask when action=scale).
            if detector is None and action != "scale":
                detector = "iqr"
            mask = self._detect(df[col], detector, cfg) if detector else None
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
