import json
import re
from typing import Callable

import numpy as np
import pandas as pd

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.parallel import compute_mini_profile
from pipeline.state import PipelineState

VALID_OPS = {
    "ratio", "difference", "product", "sum_group", "square", "sqrt", "log1p",
    "row_mean", "row_max", "row_count_positive", "days_since", "is_recent",
    "equal_width_bins", "quantile_bins", "cross_categorical",
}

CREATOR_PROMPT = """You propose new features. For each proposal, output an operation spec.

Available operations: ratio, difference, product, sum_group, square, sqrt, log1p,
row_mean, row_max, row_count_positive, days_since, is_recent, equal_width_bins,
quantile_bins, cross_categorical.

Each spec MUST include `temporal_class`:
- "pre_encoding" for cross_categorical (executes before encoding).
- "post_encoding" for all other operations.

Columns:
{column_summary}

Respond with ONLY a JSON array (max {cap} items), no prose:
[{{"operation": "<op>", "sources": ["<col1>", "<col2>"], "name": "<new_col_name>", "temporal_class": "<pre_encoding|post_encoding>"}}, ...]
"""


def _extract_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array: {text[:200]}")
    return json.loads(m.group(0))


class FeatureCreator(BaseTool):
    def __init__(self, model_call: Callable[[str], str], judge=None):
        self.model_call = model_call
        self.judge = judge  # JudgeAgent | None — ranks/caps proposals
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
        # FeatureCreator uses run_pre / run_post; standard chain is a no-op
        return state

    def _ensure_specs(self, state: PipelineState) -> None:
        if self._specs is not None:
            return
        feature_cols = [c for c in state.df.columns if c != state.target_column]
        if not feature_cols:
            self._specs = []
            return
        summary_lines = []
        for col in feature_cols:
            p = state.profile.get(col, {})
            summary_lines.append(
                f"- {col}: type={state.column_types.get(col)}, mi_with_target={p.get('mi_with_target')}, "
                f"skewness={p.get('skewness')}, nunique={p.get('nunique')}"
            )
        cap = state.config.feature_creation.max_created_features
        prompt = CREATOR_PROMPT.format(column_summary="\n".join(summary_lines), cap=cap)
        try:
            response = self.model_call(prompt)
            raw_specs = _extract_json_array(response)
        except Exception as e:
            state.warnings.append(f"FeatureCreator parse failed: {e}; skipping feature creation")
            raw_specs = []

        valid_specs: list[dict] = []
        for spec in raw_specs:
            op = spec.get("operation")
            sources = spec.get("sources", [])
            name = spec.get("name")
            tc = spec.get("temporal_class")
            if op not in VALID_OPS or tc not in {"pre_encoding", "post_encoding"} or not name:
                state.warnings.append(f"Rejected spec (invalid): {spec}")
                continue
            if not isinstance(sources, list) or not all(s in state.df.columns for s in sources):
                state.warnings.append(f"Rejected spec (sources missing): {spec}")
                continue
            valid_specs.append(spec)

        # Judge Agent (Solution F): ranks proposals and caps to `cap` items.
        # Falls back to proxy-MI ranking if the Judge LLM is unavailable.
        if self.judge is not None:
            kept, source = self.judge.rank(
                specs=valid_specs,
                profile=state.profile,
                target_column=state.target_column,
                task=state.task,
                cap=cap,
            )
            state.warnings.append(f"FeatureCreator ranking source={source}, kept={len(kept)}/{len(valid_specs)}")
            valid_specs = kept
        else:
            def proxy_mi(spec):
                scores = [state.profile.get(s, {}).get("mi_with_target") or 0.0 for s in spec["sources"]]
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
