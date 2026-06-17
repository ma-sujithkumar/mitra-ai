"""Pydantic response models + validate_response helper.

- Every model response is parsed via a Pydantic model in this module.
- Each per-item decision carries rationale, evidence_cited, alternatives_considered.
- validate_response does parse → content checks → joint-strategy-tuple degeneracy check.
- Returns (parsed_model | None, failure_reasons: list[str]).

Callers wrap the (one-revision → fall-through) loop around this helper.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

# ---------- raw response logger ----------
# Orchestrator calls set_raw_log(path) once at startup. Every LLM call site
# (call_with_revision, Judge.plan, Judge.rank) writes one entry per attempt so
# failures can be diagnosed without re-running the pipeline.
_raw_log_path: str | None = None
_RAW_SNIPPET_CHARS = 15000


def set_raw_log(path: str | None) -> None:
    global _raw_log_path
    _raw_log_path = path


def log_raw(caller: str, attempt: str, raw: str | None, status: str, failures: list[str] | None = None) -> None:
    """Append one entry to raw_responses.txt. Safe no-op when log is unset."""
    if _raw_log_path is None:
        return
    body = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
    snippet = body[:_RAW_SNIPPET_CHARS]
    truncated = "" if len(body) <= _RAW_SNIPPET_CHARS else f"\n... [truncated, total {len(body)} chars]"
    try:
        with open(_raw_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n===== caller={caller} attempt={attempt} status={status} "
                f"failures={failures or []} =====\n"
            )
            f.write(snippet)
            f.write(truncated)
            f.write("\n")
    except Exception:
        pass


# Allowed feature-creation operations
CREATOR_VALID_OPS = (
    "ratio",
    "difference",
    "product",
    "sum_group",
    "square",
    "sqrt",
    "log1p",
    "row_mean",
    "row_max",
    "row_count_positive",
    "days_since",
    "is_recent",
    "equal_width_bins",
    "quantile_bins",
    "cross_categorical",
)


# ---------- decision items ----------


class DecisionItem(BaseModel):
    rationale: str = Field(min_length=1)
    evidence_cited: list[str]
    alternatives_considered: list[str]


class TypeAssignment(DecisionItem):
    column: str
    type: Literal["numeric", "categorical", "datetime", "id", "text", "binary", "target"]


class SemanticTypeInferResponse(BaseModel):
    assignments: list[TypeAssignment]


class ImputationDecision(DecisionItem):
    column: str
    strategy: Literal["median", "mode", "knn", "iterative", "drop"]


class MissingValueResponse(BaseModel):
    decisions: list[ImputationDecision]


class OutlierDecision(DecisionItem):
    column: str
    detector: Literal["iqr", "zscore", "isolation_forest"]
    action: Literal["scale", "flag", "drop_row"]


class OutlierResponse(BaseModel):
    decisions: list[OutlierDecision]


class ScalerDecision(DecisionItem):
    column: str
    scaler: Literal["standard", "robust", "minmax", "power"]


class ScalerResponse(BaseModel):
    decisions: list[ScalerDecision]


class CreatorSpec(DecisionItem):
    operation: Literal[
        "ratio",
        "difference",
        "product",
        "sum_group",
        "square",
        "sqrt",
        "log1p",
        "row_mean",
        "row_max",
        "row_count_positive",
        "days_since",
        "is_recent",
        "equal_width_bins",
        "quantile_bins",
        "cross_categorical",
    ]
    sources: list[str]
    name: str
    temporal_class: Literal["pre_encoding", "post_encoding"]


class FeatureCreatorResponse(BaseModel):
    specs: list[CreatorSpec]


class ClusterAction(DecisionItem):
    cluster_id: int
    action: Literal["mrmr", "pca", "mrmr_then_pca", "drop", "lasso", "rf_importance"]


class SelectionPlanResponse(BaseModel):
    plan: list[ClusterAction]


# ---------- validation helper ----------


def _strategy_tuple(item: BaseModel) -> tuple:
    """The joint of every Literal field on the item — used for degeneracy check.

    For example, `(detector, action)` for OutlierDecision, `(operation,
    temporal_class)` for CreatorSpec.
    """
    # Heuristic: collect fields whose name is one of the known strategy-like names.
    candidates = ("type", "strategy", "detector", "action", "scaler", "operation", "temporal_class")
    return tuple(getattr(item, c) for c in candidates if hasattr(item, c) and getattr(item, c) is not None)


def _extract_json(raw: str) -> dict | list | None:
    """Best-effort JSON extraction. Accepts a bare JSON object, a fenced block,
    or text with embedded JSON."""
    if not raw:
        return None
    raw = raw.strip()
    # Strip code fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        # First object or array in the text
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        candidate = m.group(1) if m else raw
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _items_of(parsed: BaseModel) -> list[BaseModel]:
    """Return the list-of-DecisionItem regardless of which response container."""
    if hasattr(parsed, "assignments"):
        return list(parsed.assignments)
    if hasattr(parsed, "decisions"):
        return list(parsed.decisions)
    if hasattr(parsed, "specs"):
        return list(parsed.specs)
    if hasattr(parsed, "plan"):
        return list(parsed.plan)
    return []


def validate_response(
    model_cls: type[BaseModel],
    raw_text: str,
    sent_field_names: set[str],
    cfg,
) -> tuple[BaseModel | None, list[str]]:
    """Parse + content-check the model response.

    Returns (parsed_model, failure_reasons).

    Failure modes:
      - 'parse'        — JSON missing or Pydantic ValidationError.
      - 'evidence'     — empty or unknown-field entries in evidence_cited.
      - 'rationale'    — too short or matches boilerplate denylist.
      - 'alternatives' — too few alternatives_considered.
      - 'lazy'         — > lazy_response_threshold of items share one strategy tuple.

    Empty failure list ⇒ response is acceptable.
    """
    raw_json = _extract_json(raw_text)
    if raw_json is None:
        return None, ["parse"]

    # Allow both `{...}` and `[...]` shapes — wrap arrays into the container's
    # singular list field if the model returned a bare list.
    if isinstance(raw_json, list):
        list_field = _list_field_name(model_cls)
        if list_field is None:
            return None, ["parse"]
        raw_json = {list_field: raw_json}

    try:
        parsed = model_cls(**raw_json)
    except ValidationError:
        return None, ["parse"]

    items = _items_of(parsed)
    if not items:
        return parsed, []  # nothing to check

    failures: list[str] = []
    denylist = [s.lower() for s in cfg.validation.boilerplate_denylist]
    min_rationale = cfg.validation.min_rationale_chars
    min_alts = cfg.validation.min_alternatives

    for item in items:
        rationale = (item.rationale or "").strip()
        if len(rationale) < min_rationale:
            failures.append("rationale")
            break
        if any(bad in rationale.lower() for bad in denylist):
            failures.append("rationale")
            break

    if "rationale" not in failures:
        for item in items:
            cited = item.evidence_cited or []
            if not cited:
                failures.append("evidence")
                break
            # Unknown-field check: every cited field must appear in sent_field_names.
            if sent_field_names and not all(_field_known(c, sent_field_names) for c in cited):
                failures.append("evidence")
                break

    if "alternatives" not in failures:
        for item in items:
            if len(item.alternatives_considered or []) < min_alts:
                failures.append("alternatives")
                break

    # Degeneracy check (joint strategy tuple)
    if len(items) >= 3:
        counts = Counter(_strategy_tuple(it) for it in items)
        top_count = counts.most_common(1)[0][1]
        if top_count / len(items) > cfg.validation.lazy_response_threshold:
            failures.append("lazy")

    return parsed, failures


def _field_known(cited: str, sent: set[str]) -> bool:
    """Allow exact, dotted-prefix, or suffix matches against the sent whitelist."""
    if not cited:
        return False
    if cited in sent:
        return True
    # Permit the leaf-name form (e.g. `null_rate` when sent includes `columns.null_rate`).
    return any(s.endswith("." + cited) or s == cited for s in sent)


def _list_field_name(model_cls: type[BaseModel]) -> str | None:
    """Return the name of the singular list field for a response container."""
    for fname in ("assignments", "decisions", "specs", "plan"):
        if fname in model_cls.model_fields:
            return fname
    return None


def call_with_revision(
    model_call,
    prompt: str,
    model_cls: type[BaseModel],
    sent_field_names: set[str],
    cfg,
    caller: str = "unknown",
) -> tuple[BaseModel | None, str, list[str]]:
    """Standard call → validate → one revision → return contract.

    Returns (parsed_model | None, source_tag, last_failures).
    source_tag is 'ok' if first attempt passed, 'ok:revised' if second passed,
    'fallback' if both failed. Callers apply their deterministic default when
    source_tag == 'fallback'.

    Every attempt is recorded to raw_responses.txt via log_raw(...) if the
    orchestrator has called set_raw_log() at startup.
    """
    try:
        raw = model_call(prompt)
    except Exception as e:
        log_raw(caller, "first", f"<exception: {e}>", "fallback", ["parse"])
        return None, "fallback", ["parse"]

    parsed, failures = validate_response(model_cls, raw, sent_field_names, cfg)
    if parsed is not None and not failures:
        log_raw(caller, "first", raw, "ok", [])
        return parsed, "ok", []
    log_raw(caller, "first", raw, "rejected", failures)

    revision_prompt = (
        prompt
        + "\n\n## REVISION\nprior_response_was_uninformative=true\n"
        + "Your previous response was rejected because: "
        + ", ".join(failures or ["parse"])
        + ".\nProduce a fresh response that names exact EvidencePacket field "
        + "names in `evidence_cited` and a rationale of at least "
        + f"{cfg.validation.min_rationale_chars} characters per item."
    )
    try:
        raw2 = model_call(revision_prompt)
    except Exception as e:
        log_raw(caller, "revision", f"<exception: {e}>", "fallback", failures or ["parse"])
        return None, "fallback", failures or ["parse"]

    parsed2, failures2 = validate_response(model_cls, raw2, sent_field_names, cfg)
    if parsed2 is not None and not failures2:
        log_raw(caller, "revision", raw2, "ok:revised", [])
        return parsed2, "ok:revised", []
    log_raw(caller, "revision", raw2, "fallback", failures2 or ["parse"])
    return None, "fallback", failures2 or ["parse"]
