"""SemanticTypeInfer — one model call to assign a type per column.

Builds a typed SemanticTypeInferEvidence packet, sends a prompt that includes
STRATEGY_DEFINITIONS (mechanical descriptions only — no when-to-use guidance),
parses the response via the Pydantic SemanticTypeInferResponse with the four
content checks, retries once on content failure, falls through to dtype-based
inference on a second failure.
"""
from __future__ import annotations

import random
import re
from typing import Callable

import pandas as pd

from pipeline.base import BaseTool, PostconditionError, PreconditionError
from pipeline.evidence import (
    ColumnTypeEvidence,
    SemanticTypeInferEvidence,
    render,
)
from pipeline.parallel import compute_mini_profile
from pipeline.responses import (
    SemanticTypeInferResponse,
    call_with_revision,
)
from pipeline.state import PipelineState

# STRATEGY_DEFINITIONS lives at module level. Mechanical only — no when-to-use
# guidance. The prompt injects this dict verbatim.
STRATEGY_DEFINITIONS: dict[str, str] = {
    "numeric": "Continuous or count values stored as int/float.",
    "categorical": "Discrete values with finite, typically small, vocabulary; encoded one-of-K downstream.",
    "datetime": "Timestamps or date strings parseable into year/month/day/quarter components.",
    "id": "Unique row identifiers carrying no learnable signal (primary keys, UUIDs).",
    "text": "Free-form unstructured strings without finite vocabulary.",
    "binary": "Two-valued column (booleans, yes/no, 0/1).",
    "target": "Reserved label for the target column; assigned by the caller, never by you.",
}

INFER_PROMPT = """You assign a single semantic type to each column.

## STRATEGY DEFINITIONS (mechanical descriptions only)
{strategy_definitions}

## RESPONSE SHAPE
Return ONLY a JSON object of this shape:
{{
  "assignments": [
    {{
      "column": "<name>",
      "type": "<one of numeric|categorical|datetime|id|text|binary|target>",
      "rationale": "<at least {min_rationale_chars} characters explaining the choice using fields from the EVIDENCE block>",
      "evidence_cited": ["<field paths from the EVIDENCE you actually used>"],
      "alternatives_considered": ["<other types you considered before picking>"]
    }}
  ]
}}

{evidence_block}
"""

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$")
_PHONE_RE = re.compile(r"^\+?\d[\d\-\s\(\)]{6,}$")
_NUMERIC_STR_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _regex_signature(values: list) -> dict[str, int]:
    counts = {"uuid": 0, "email": 0, "iso_date": 0, "phone": 0, "numeric_string": 0}
    for v in values:
        s = str(v)
        if _UUID_RE.match(s):
            counts["uuid"] += 1
        if _EMAIL_RE.match(s):
            counts["email"] += 1
        if _ISO_DATE_RE.match(s):
            counts["iso_date"] += 1
        if _PHONE_RE.match(s):
            counts["phone"] += 1
        if _NUMERIC_STR_RE.match(s):
            counts["numeric_string"] += 1
    return counts


def _decompose_datetime(df: pd.DataFrame, col: str) -> list[str]:
    parsed = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"] = parsed.dt.year
    df[f"{col}_month"] = parsed.dt.month
    df[f"{col}_day"] = parsed.dt.day
    df[f"{col}_quarter"] = parsed.dt.quarter
    new_cols = [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_quarter"]
    df.drop(columns=[col], inplace=True)
    return new_cols


def _strategy_definitions_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in STRATEGY_DEFINITIONS.items())


class SemanticTypeInfer(BaseTool):
    def __init__(self, model_call: Callable[[str], str]):
        self.model_call = model_call

    def precondition(self, state: PipelineState) -> None:
        if state.profile is None:
            raise PreconditionError("SemanticTypeInfer: state.profile is None")

    def run(self, state: PipelineState) -> None:
        df = state.df
        cfg = state.config

        # Build EvidencePacket.
        rng = random.Random(cfg.pipeline.random_state)
        per_col: list[ColumnTypeEvidence] = []
        for col in df.columns:
            p = state.profile.get(col, {})
            dropped_vals = df[col].dropna()
            samples = dropped_vals.sample(
                n=min(5, len(dropped_vals)),
                random_state=cfg.pipeline.random_state,
            ).astype(str).tolist() if len(dropped_vals) else []
            per_col.append(
                ColumnTypeEvidence(
                    name=col,
                    dtype=str(p.get("dtype", df[col].dtype)),
                    null_rate=float(p.get("null_rate", 0.0)),
                    nunique=int(p.get("nunique", 0)),
                    top_values=[str(v) for v in (p.get("top_values") or [])][:5],
                    random_samples=samples,
                    regex_signature=_regex_signature(samples),
                )
            )
        packet = SemanticTypeInferEvidence(columns=per_col)
        evidence_block, sent_fields = render(packet, truncate_after_chars=int(cfg.llm.max_tokens * 0.7 * 4))

        prompt = INFER_PROMPT.format(
            strategy_definitions=_strategy_definitions_block(),
            min_rationale_chars=cfg.validation.min_rationale_chars,
            evidence_block=evidence_block,
        )

        parsed, source, failures = call_with_revision(
            self.model_call, prompt, SemanticTypeInferResponse, sent_fields, cfg,
            caller="SemanticTypeInfer",
        )

        if parsed is None or source == "fallback":
            state.warnings.append(
                f"SemanticTypeInfer fell through to dtype inference (failures={failures})"
            )
            assignments_iter = [
                (c, self._dtype_fallback(df[c]), None) for c in df.columns
            ]
        else:
            assignments_iter = [
                (a.column, a.type, a) for a in parsed.assignments  # type: ignore[attr-defined]
            ]

        column_types: dict[str, str] = {}
        for col, typ, _ in assignments_iter:
            if col in df.columns:
                column_types[col] = typ
        for col in df.columns:
            column_types.setdefault(col, self._dtype_fallback(df[col]))

        # Caller's target column overrides any model output.
        column_types[state.target_column] = "target"

        cols_to_drop = [
            c for c, t in column_types.items() if t in {"id", "text"} and c != state.target_column
        ]
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
            for nc, mp in mini.items():
                mp["mi_with_target"] = 0.0
                mp["null_mask_corr"] = {}
                state.profile[nc] = mp

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
