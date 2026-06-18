"""MissingValueHandler — pick an imputation strategy per column with nulls.

Typed MissingValueEvidence + Pydantic MissingValueResponse with the four
content checks, one revision, fall-through to median/mode.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.evidence import MissingValueEvidence, NullColumnEvidence, render
from pipeline.responses import MissingValueResponse, call_with_revision
from pipeline.state import PipelineState

# Mechanical-only strategy descriptions. No when-to-use guidance.
STRATEGY_DEFINITIONS: dict[str, str] = {
    "median": "Fills nulls with the column's median value.",
    "mode": "Fills nulls with the most frequent non-null value.",
    "knn": "Fills nulls using the mean of the k nearest non-null rows in feature space.",
    "iterative": "Fits a regression of this column on the others and uses predictions for the nulls (MICE).",
    "drop": "Removes the column from the dataframe entirely; no further transforms reference it.",
}

IMPUTE_PROMPT = """You pick a single imputation strategy for each column that has nulls.

## STRATEGY DEFINITIONS (mechanical descriptions only)
{strategy_definitions}

## RESPONSE SHAPE
Return ONLY a JSON object of this shape:
{{
  "decisions": [
    {{
      "column": "<name>",
      "strategy": "<one of median|mode|knn|iterative|drop>",
      "rationale": "<at least {min_rationale_chars} characters citing fields from EVIDENCE>",
      "evidence_cited": ["<field paths from EVIDENCE you used>"],
      "alternatives_considered": ["<other strategies you weighed>"]
    }}
  ]
}}

{evidence_block}
"""


def _null_run_lengths(mask: pd.Series) -> list[int]:
    """Histogram of consecutive-null streak lengths."""
    runs: list[int] = []
    current = 0
    for v in mask.to_numpy():
        if v:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0
    if current > 0:
        runs.append(current)
    return runs


def _null_mask_corr_top5(df: pd.DataFrame, col: str) -> dict[str, float]:
    null_mask = df[col].isna().astype(int)
    out: dict[str, float] = {}
    for other in df.columns:
        if other == col:
            continue
        other_num = pd.to_numeric(df[other], errors="coerce")
        if other_num.notna().sum() < 5:
            continue
        try:
            c = float(null_mask.corr(other_num.fillna(other_num.median())))
            if not np.isnan(c) and abs(c) > 0.1:
                out[other] = c
        except Exception:
            pass
    return dict(sorted(out.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5])


def _target_rates(target: pd.Series, null_mask: pd.Series, task: str) -> tuple[float | None, float | None]:
    if task != "classification":
        return None, None
    try:
        positive = (target == target.mode().iloc[0]).astype(int) if target.dtype == object else (target == 1).astype(int)
        rate_when_null = float(positive[null_mask.astype(bool)].mean()) if null_mask.any() else None
        rate_when_present = float(positive[~null_mask.astype(bool)].mean()) if (~null_mask.astype(bool)).any() else None
        return rate_when_null, rate_when_present
    except Exception:
        return None, None


def _strategy_definitions_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in STRATEGY_DEFINITIONS.items())


class MissingValueHandler(BaseTool):
    def __init__(self, model_call: Callable[[str], str] | None):
        # model_call is None => deterministic mode (no LLM): rule-based strategy
        # per column using the same defaults as the LLM fallback path.
        self.model_call = model_call

    def precondition(self, state: PipelineState) -> None:
        if state.column_types is None:
            raise PreconditionError("MissingValueHandler: state.column_types is None")

    def run(self, state: PipelineState) -> None:
        df = state.df
        cfg = state.config
        drop_threshold = cfg.imputation.null_drop_threshold

        # Target imputation first (code only; not a model decision).
        if state.target.isna().any():
            if state.task == "classification":
                fill = state.target.mode().iloc[0]
            else:
                fill = state.target.median()
            state.target = state.target.fillna(fill)
            state.transformers.append(
                {"step": "imputation", "column": state.target_column, "strategy": "target_fill", "fill_value": _safe(fill)}
            )

        # Spec §4 "Null detection in categoricals": categorical and binary
        # columns are excluded from model-driven null detection. Tokens like
        # "NA" in those columns are legitimate category labels, not nulls.
        # Only numeric/datetime/target columns participate in model imputation;
        # any stray true-NaN in a categorical column is mopped up by the final
        # mode-fill pass below.
        cols_with_nulls = [
            c for c in df.columns
            if df[c].isna().any()
            and state.column_types.get(c) not in {"categorical", "binary"}
        ]
        if not cols_with_nulls:
            self._final_pass(df, state)
            return

        # Deterministic mode: rule-based strategy per column (median for numeric,
        # mode otherwise). The hard drop-threshold rule below still applies.
        if self.model_call is None:
            state.last_llm_source = "deterministic"
            decision_map: dict[str, str] = {
                col: ("median" if state.column_types.get(col, "numeric") == "numeric" else "mode")
                for col in cols_with_nulls
            }
            self._apply_decisions(df, state, cols_with_nulls, decision_map, drop_threshold)
            self._final_pass(df, state)
            return

        per_col: list[NullColumnEvidence] = []
        for col in cols_with_nulls:
            p = state.profile.get(col, {})
            null_mask = df[col].isna()
            rate_null, rate_present = _target_rates(state.target, null_mask, state.task)
            present_values = df[col].dropna().sample(
                n=min(10, df[col].dropna().shape[0]),
                random_state=cfg.pipeline.random_state,
            ).astype(str).tolist() if df[col].notna().any() else []
            per_col.append(
                NullColumnEvidence(
                    name=col,
                    null_rate=float(p.get("null_rate", null_mask.mean())),
                    null_run_lengths=_null_run_lengths(null_mask)[:20],
                    null_mask_corr_top5=_null_mask_corr_top5(df, col),
                    target_rate_when_null=rate_null,
                    target_rate_when_present=rate_present,
                    random_present_values=present_values,
                    dtype=str(df[col].dtype),
                    semantic_type=state.column_types.get(col, "numeric"),
                )
            )

        packet = MissingValueEvidence(columns=per_col)
        evidence_block, sent_fields = render(packet, truncate_after_chars=int(cfg.llm.max_tokens * 0.7 * 4))

        prompt = IMPUTE_PROMPT.format(
            strategy_definitions=_strategy_definitions_block(),
            min_rationale_chars=cfg.validation.min_rationale_chars,
            evidence_block=evidence_block,
        )

        parsed, source, failures = call_with_revision(
            self.model_call, prompt, MissingValueResponse, sent_fields, cfg,
            caller="MissingValueHandler",
        )
        state.last_llm_source = source

        decision_map: dict[str, str] = {}
        if parsed is None or source == "fallback":
            state.warnings.append(
                f"MissingValueHandler fell through to median/mode (failures={failures})"
            )
            for col in cols_with_nulls:
                t = state.column_types.get(col, "numeric")
                decision_map[col] = "median" if t == "numeric" else "mode"
        else:
            for d in parsed.decisions:  # type: ignore[attr-defined]
                decision_map[d.column] = d.strategy

        self._apply_decisions(df, state, cols_with_nulls, decision_map, drop_threshold)
        self._final_pass(df, state)

    @staticmethod
    def _apply_decisions(
        df: pd.DataFrame,
        state: PipelineState,
        cols_with_nulls: list[str],
        decision_map: dict[str, str],
        drop_threshold: float,
    ) -> None:
        """Apply per-column imputation strategies. Shared by deterministic and LLM modes."""
        cfg = state.config
        # Hard rule: drop above the configured null threshold regardless of strategy source.
        for col in cols_with_nulls:
            if state.profile.get(col, {}).get("null_rate", 0.0) > drop_threshold:
                decision_map[col] = "drop"

        knn_cols: list[str] = []
        iterative_cols: list[str] = []
        for col in cols_with_nulls:
            strategy = decision_map.get(col, "median")
            if strategy == "drop":
                df.drop(columns=[col], inplace=True)
                state.dropped_columns.append(col)
                state.column_types.pop(col, None)
                state.warnings.append(f"{col} dropped (null_rate={state.profile[col].get('null_rate', 0.0):.2f})")
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
                    max_iter=cfg.imputation.iterative_max_iter,
                    random_state=cfg.pipeline.random_state,
                )
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                for col in target_cols:
                    state.transformers.append(
                        {"step": "imputation", "column": col, "strategy": "iterative", "random_state": cfg.pipeline.random_state}
                    )

    @staticmethod
    def _final_pass(df: pd.DataFrame, state: PipelineState) -> None:
        """Mode-fill any leftover nulls — including stray empty-string NaNs in
        categorical columns that were excluded from model-driven imputation."""
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
