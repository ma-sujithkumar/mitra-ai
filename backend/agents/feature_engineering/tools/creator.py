"""FeatureCreator — propose operation specs, then run pre/post phases.

Typed FeatureCreatorEvidence + Pydantic FeatureCreatorResponse with the four
content checks, one revision, fall-through to skip. Surviving specs are passed
to the Judge Agent for ranking/capping.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from backend.agents.feature_engineering.base import BaseTool, PreconditionError
from backend.agents.feature_engineering.evidence import (
    CreatorColumnEvidence,
    FeatureCreatorEvidence,
    render,
)
from backend.agents.feature_engineering.parallel import compute_mini_profile
from backend.agents.feature_engineering.responses import FeatureCreatorResponse, call_with_revision
from backend.agents.feature_engineering.state import PipelineState

# Operation vocabulary (15 items). Mechanical descriptions only.
STRATEGY_DEFINITIONS: dict[str, str] = {
    "ratio": "Divide column A by column B (zeros in B are treated as missing).",
    "difference": "Subtract column B from column A.",
    "product": "Multiply column A by column B.",
    "sum_group": "Element-wise sum of two or more numeric columns.",
    "square": "Square the column's values.",
    "sqrt": "Square root of the absolute value of the column.",
    "log1p": "Natural log of (1 + |value|).",
    "row_mean": "Per-row mean across the listed columns.",
    "row_max": "Per-row maximum across the listed columns.",
    "row_count_positive": "Per-row count of strictly positive values across the listed columns.",
    "days_since": "Days elapsed between now and the parsed datetime in the source column.",
    "is_recent": "Binary indicator: 1 if the source datetime falls within the last 365 days.",
    "equal_width_bins": "Bin the numeric source into equal-width buckets.",
    "quantile_bins": "Bin the numeric source into equal-frequency (quantile) buckets.",
    "cross_categorical": "Concatenate two categorical columns into a single string label. PRE-ENCODING.",
}

CREATOR_PROMPT = """Propose new feature operations.

## STRATEGY DEFINITIONS (mechanical descriptions only)
{strategy_definitions}

## RULES
- `cross_categorical` MUST be marked temporal_class="pre_encoding".
- All other operations MUST be temporal_class="post_encoding".
- `sources` must reference columns from the EVIDENCE block.
- Pick at most {cap} operations.

## RESPONSE SHAPE
Return ONLY a JSON object of this shape:
{{
  "specs": [
    {{
      "operation": "<one of the operations above>",
      "sources": ["<col>", "..."],
      "name": "<new column name>",
      "temporal_class": "<pre_encoding|post_encoding>",
      "rationale": "<at least {min_rationale_chars} characters citing fields from EVIDENCE>",
      "evidence_cited": ["<field paths from EVIDENCE you used>"],
      "alternatives_considered": ["<other operations you weighed>"]
    }}
  ]
}}

{evidence_block}
"""


def _strategy_definitions_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in STRATEGY_DEFINITIONS.items())


def _correlated_top3(profile: dict, col: str) -> dict[str, float]:
    cm = profile.get("_correlation_matrix") or {}
    row = cm.get(col, {})
    items = [
        (k, float(v)) for k, v in row.items()
        if k != col and isinstance(v, (int, float)) and not pd.isna(v)
    ]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return {k: v for k, v in items[:3]}


def _co_occurring_pairs(profile: dict, target_col: str, top_n: int = 10) -> list[tuple[str, str, float]]:
    """Top column pairs by joint MI with target.

    Prefers the profiler's true joint-MI ranking (`_joint_mi_pairs`) when
    present, since that is computed on the concatenated feature columns
    against the target. Falls back to the MI-product proxy when the profiler
    key is absent (plan ambiguity #29).
    """
    precomputed = profile.get("_joint_mi_pairs")
    if precomputed:
        return [(a, b, float(s)) for a, b, s in precomputed][:top_n]

    cols = [c for c in profile if not c.startswith("_") and c != target_col]
    scored: list[tuple[str, str, float]] = []
    for i, a in enumerate(cols):
        mi_a = profile.get(a, {}).get("mi_with_target") or 0.0
        for b in cols[i + 1:]:
            mi_b = profile.get(b, {}).get("mi_with_target") or 0.0
            scored.append((a, b, float(mi_a * mi_b)))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:top_n]


class FeatureCreator(BaseTool):
    def __init__(self, model_call: Callable[[str], str], judge=None):
        self.model_call = model_call
        self.judge = judge
        self._specs: list[dict] | None = None
        self._proposed_pre: list[dict] = []
        self._proposed_post: list[dict] = []

    def precondition(self, state: PipelineState) -> None:
        pass

    def run(self, state: PipelineState) -> None:
        pass

    def postcondition(self, state: PipelineState) -> None:
        pass

    def __call__(self, state: PipelineState) -> PipelineState:
        return state  # FeatureCreator dispatches through run_pre / run_post

    def create_deterministic(self, state: PipelineState) -> None:
        """Deterministic feature engineering (no LLM): degree-2 ratio + product
        features from the top mutual-information numeric columns, capped by
        feature_creation.max_created_features. Run AFTER encoding so every source
        is numeric. Matches DESIGN_PLAN.md §9 (degree-2 poly + ratio only).
        """
        cfg = state.config
        cap = cfg.feature_creation.max_created_features
        if cap <= 0:
            return
        df = state.df
        numeric_cols = [
            c for c in df.columns
            if c != state.target_column
            and state.column_types.get(c) == "numeric"
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric_cols) < 2:
            return
        # Rank by mutual information with the target (from the profile) and bound
        # the base set to keep the pair count small/cheap.
        ranked = sorted(
            numeric_cols,
            key=lambda c: state.profile.get(c, {}).get("mi_with_target") or 0.0,
            reverse=True,
        )
        top = ranked[: min(len(ranked), 6)]
        specs: list[dict] = []
        for i, a in enumerate(top):
            for b in top[i + 1:]:
                specs.append({"operation": "ratio", "sources": [a, b], "name": f"{a}_div_{b}", "temporal_class": "post_encoding"})
                specs.append({"operation": "product", "sources": [a, b], "name": f"{a}_x_{b}", "temporal_class": "post_encoding"})
        specs = specs[:cap]

        new_cols: list[str] = []
        for spec in specs:
            if self._execute(state, spec):
                new_cols.append(spec["name"])
        if new_cols:
            mini = compute_mini_profile(state.df, new_cols)
            for nc, p in mini.items():
                p["mi_with_target"] = 0.0
                p["null_mask_corr"] = {}
                state.profile[nc] = p
                state.column_types[nc] = "numeric"

    def _ensure_specs(self, state: PipelineState) -> None:
        if self._specs is not None:
            return
        feature_cols = [c for c in state.df.columns if c != state.target_column]
        if not feature_cols:
            self._specs = []
            return
        cfg = state.config

        per_col: list[CreatorColumnEvidence] = []
        for col in feature_cols:
            p = state.profile.get(col, {})
            per_col.append(
                CreatorColumnEvidence(
                    name=col,
                    semantic_type=state.column_types.get(col, "numeric"),
                    mi_with_target=float(p.get("mi_with_target") or 0.0),
                    nunique=int(p.get("nunique") or 0),
                    correlated_with_top3=_correlated_top3(state.profile, col),
                    decomposed_from=p.get("decomposed_from"),
                )
            )
        packet = FeatureCreatorEvidence(
            columns=per_col,
            co_occurring_pairs=_co_occurring_pairs(state.profile, state.target_column),
        )
        evidence_block, sent_fields = render(packet, truncate_after_chars=int(cfg.llm.max_tokens * 0.7 * 4))

        cap = cfg.feature_creation.max_created_features
        prompt = CREATOR_PROMPT.format(
            strategy_definitions=_strategy_definitions_block(),
            min_rationale_chars=cfg.validation.min_rationale_chars,
            cap=cap,
            evidence_block=evidence_block,
        )

        parsed, source, failures = call_with_revision(
            self.model_call, prompt, FeatureCreatorResponse, sent_fields, cfg,
            caller="FeatureCreator",
        )
        state.last_llm_source = source

        if parsed is None or source == "fallback":
            state.warnings.append(
                f"FeatureCreator fell through to no-creation (failures={failures})"
            )
            self._specs = []
            return

        valid_specs: list[dict] = []
        for spec in parsed.specs:  # type: ignore[attr-defined]
            sources = list(spec.sources or [])
            if not sources or not all(s in state.df.columns for s in sources):
                state.warnings.append(f"FeatureCreator rejected spec (sources missing): {spec.name}")
                continue
            if not spec.name:
                state.warnings.append("FeatureCreator rejected spec with empty name")
                continue
            valid_specs.append({
                "operation": spec.operation,
                "sources": sources,
                "name": spec.name,
                "temporal_class": spec.temporal_class,
            })

        # Judge ranks / caps the surviving specs.
        if self.judge is not None and valid_specs:
            kept, judge_source = self.judge.rank(
                specs=valid_specs,
                profile=state.profile,
                target_column=state.target_column,
                task=state.task,
                cap=cap,
            )
            state.warnings.append(f"FeatureCreator ranking source={judge_source}, kept={len(kept)}/{len(valid_specs)}")
            valid_specs = kept
        else:
            def proxy_mi(sp):
                scores = [state.profile.get(s, {}).get("mi_with_target") or 0.0 for s in sp["sources"]]
                return float(np.mean(scores)) if scores else 0.0
            valid_specs.sort(key=proxy_mi, reverse=True)
            valid_specs = valid_specs[:cap]

        self._specs = valid_specs
        self._proposed_pre = [s for s in valid_specs if s["temporal_class"] == "pre_encoding"]
        self._proposed_post = [s for s in valid_specs if s["temporal_class"] == "post_encoding"]

    def run_pre(self, state: PipelineState) -> None:
        if state.column_types is None:
            raise PreconditionError("FeatureCreator.run_pre: column_types is None")
        self._ensure_specs(state)
        for spec in self._proposed_pre:
            self._execute(state, spec)
        state.pre_encoding_done = True

    def run_post(self, state: PipelineState) -> None:
        if not state.pre_encoding_done:
            raise PreconditionError("FeatureCreator.run_post: pre_encoding_done is False")
        self._ensure_specs(state)
        new_cols: list[str] = []
        for spec in self._proposed_post:
            ok = self._execute(state, spec)
            if ok:
                new_cols.append(spec["name"])

        if new_cols:
            mini = compute_mini_profile(state.df, new_cols)
            for nc, p in mini.items():
                p["mi_with_target"] = 0.0
                p["null_mask_corr"] = {}
                state.profile[nc] = p
                state.column_types[nc] = "numeric"

    def _execute(self, state: PipelineState, spec: dict) -> bool:
        df = state.df
        op = spec["operation"]
        sources = spec["sources"]
        name = spec["name"]
        try:
            if name in df.columns:
                state.warnings.append(f"Skipped {name}: column already exists")
                return False
            if op == "ratio" and len(sources) == 2:
                a, b = pd.to_numeric(df[sources[0]], errors="coerce"), pd.to_numeric(df[sources[1]], errors="coerce")
                df[name] = a / b.replace(0, np.nan)
                df[name] = df[name].fillna(0.0)
            elif op == "difference" and len(sources) == 2:
                df[name] = pd.to_numeric(df[sources[0]], errors="coerce") - pd.to_numeric(df[sources[1]], errors="coerce")
                df[name] = df[name].fillna(0.0)
            elif op == "product" and len(sources) == 2:
                df[name] = pd.to_numeric(df[sources[0]], errors="coerce") * pd.to_numeric(df[sources[1]], errors="coerce")
                df[name] = df[name].fillna(0.0)
            elif op == "sum_group" and len(sources) >= 2:
                df[name] = sum(pd.to_numeric(df[s], errors="coerce").fillna(0.0) for s in sources)
            elif op == "square" and len(sources) == 1:
                df[name] = pd.to_numeric(df[sources[0]], errors="coerce").fillna(0.0) ** 2
            elif op == "sqrt" and len(sources) == 1:
                df[name] = np.sqrt(np.abs(pd.to_numeric(df[sources[0]], errors="coerce").fillna(0.0)))
            elif op == "log1p" and len(sources) == 1:
                df[name] = np.log1p(np.abs(pd.to_numeric(df[sources[0]], errors="coerce").fillna(0.0)))
            elif op == "row_mean" and len(sources) >= 2:
                df[name] = df[sources].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.0)
            elif op == "row_max" and len(sources) >= 2:
                df[name] = df[sources].apply(pd.to_numeric, errors="coerce").max(axis=1).fillna(0.0)
            elif op == "row_count_positive" and len(sources) >= 1:
                df[name] = (df[sources].apply(pd.to_numeric, errors="coerce") > 0).sum(axis=1)
            elif op in {"days_since", "is_recent"} and len(sources) == 1:
                parsed = pd.to_datetime(df[sources[0]], errors="coerce")
                if op == "days_since":
                    df[name] = (pd.Timestamp.utcnow().tz_localize(None) - parsed).dt.days.fillna(0)
                else:
                    days_since = (pd.Timestamp.utcnow().tz_localize(None) - parsed).dt.days
                    df[name] = (days_since < 365).astype(int).fillna(0)
            elif op == "equal_width_bins" and len(sources) == 1:
                df[name] = pd.cut(
                    pd.to_numeric(df[sources[0]], errors="coerce"),
                    bins=state.config.feature_creation.equal_width_bins,
                    labels=False,
                ).fillna(-1).astype(int)
            elif op == "quantile_bins" and len(sources) == 1:
                try:
                    df[name] = pd.qcut(
                        pd.to_numeric(df[sources[0]], errors="coerce"),
                        q=state.config.feature_creation.quantile_bins,
                        labels=False,
                        duplicates="drop",
                    ).fillna(-1).astype(int)
                except Exception:
                    df[name] = 0
            elif op == "cross_categorical" and len(sources) == 2:
                df[name] = df[sources[0]].astype(str) + "_" + df[sources[1]].astype(str)
                state.column_types[name] = "categorical"
            else:
                state.warnings.append(f"Unsupported op/source combo: {spec}")
                return False
            state.created_columns.append({"name": name, "operation": op, "sources": sources})
            return True
        except Exception as e:
            state.warnings.append(f"Feature creation failed for {name}: {e}")
            return False
