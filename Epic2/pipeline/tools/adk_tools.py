"""ADK tool wrappers. Each function is registered as a tool on the orchestrator agent.

The shared PipelineState is injected once at startup via set_pipeline_state().
Tools mutate state in place and return small dicts so the ADK agent can decide what to call next.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable

import pandas as pd

from pipeline.base import PostconditionError, PreconditionError
from pipeline.state import PipelineState
from pipeline.tools.creator import FeatureCreator
from pipeline.tools.encoder import Encoder
from pipeline.tools.imputer import MissingValueHandler
from pipeline.tools.infer import SemanticTypeInfer
from pipeline.tools.outlier import OutlierHandler
from pipeline.tools.profiler import DataProfiler
from pipeline.tools.reporter import FeatureReporter
from pipeline.tools.scaler import Scaler
from pipeline.tools.selector import FeatureSelector
from pipeline.tools.validator import FeatureValidator

_state: PipelineState | None = None
_model_call: Callable[[str], str] | None = None
_log_path: str | None = None
_creator_instance: FeatureCreator | None = None
_selector_instance: FeatureSelector | None = None


def set_pipeline_state(
    state: PipelineState,
    model_call: Callable[[str], str],
    judge_agent: Any | None = None,
) -> None:
    global _state, _model_call, _log_path, _creator_instance, _selector_instance
    _state = state
    _model_call = model_call
    _log_path = str(state.output_dir / "execution_log.txt")
    _creator_instance = FeatureCreator(model_call, judge=judge_agent)
    _selector_instance = FeatureSelector(model_call, judge=judge_agent)


def _log(tool: str, status: str, detail: str, elapsed: float) -> None:
    if _log_path is None:
        return
    one_line = " | ".join(s.strip() for s in str(detail).splitlines() if s.strip())
    with open(_log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {tool} {status} ({elapsed:.2f}s) {one_line}\n")


def _wrap(tool_name: str, fn: Callable[[], Any], already_done: Callable[[PipelineState], bool] | None = None) -> dict:
    """Run `fn` under a uniform try/except + log wrapper.

    `already_done(state)` is an optional postcondition predicate. When it
    returns True the wrapper short-circuits and does NOT invoke `fn` — every
    tool must be safe to skip if its outcome is already present, since the
    ADK agent may re-dispatch a tool more than once (spec §5 "Tool
    idempotency", §7-Z).
    """
    if _state is None:
        return {"status": "error", "detail": "pipeline state not initialised"}
    if already_done is not None:
        try:
            if already_done(_state):
                _log(tool_name, "ok", "already done", 0.0)
                return {"status": "ok", "detail": "already done"}
        except Exception:
            pass
    start = time.perf_counter()
    try:
        # Per-call LLM source tag (spec §6 "Observability detail").
        _state.last_llm_source = None
        result = fn()
        elapsed = time.perf_counter() - start
        src = _state.last_llm_source
        suffix = f" llm={src}" if src else ""
        _log(tool_name, "ok", (str(result) if result else "") + suffix, elapsed)
        return {"status": "ok", "detail": result if result else "completed"}
    except (PreconditionError, PostconditionError) as e:
        elapsed = time.perf_counter() - start
        _log(tool_name, "error", str(e), elapsed)
        return {"status": "error", "detail": str(e)}
    except Exception as e:
        elapsed = time.perf_counter() - start
        _log(tool_name, "error", f"{e}\n{traceback.format_exc()}", elapsed)
        return {"status": "error", "detail": str(e)}


# ---------- per-tool idempotency predicates ----------


def _types_done(state: PipelineState) -> bool:
    return state.column_types is not None


def _missing_done(state: PipelineState) -> bool:
    if state.df is None or state.column_types is None:
        return False
    cols = [c for c in state.df.columns
            if state.column_types.get(c) not in {"categorical", "binary"}]
    return state.df[cols].isna().sum().sum() == 0 if cols else True


def _outliers_done(state: PipelineState) -> bool:
    return state.row_count_after_outlier is not None


def _pre_done(state: PipelineState) -> bool:
    return state.pre_encoding_done


def _encoded_done(state: PipelineState) -> bool:
    if state.df is None:
        return False
    return all(pd.api.types.is_numeric_dtype(state.df[c]) for c in state.df.columns)


def _post_done(state: PipelineState) -> bool:
    if _creator_instance is None or _creator_instance._specs is None:
        return False
    posts = _creator_instance._proposed_post
    if not posts:
        return True
    return all(s["name"] in state.df.columns for s in posts)


def _scaled_done(state: PipelineState) -> bool:
    return any(t.get("step") == "scaling" for t in state.transformers)


def _selected_done(state: PipelineState) -> bool:
    return state.selected_columns is not None


def _validated_done(state: PipelineState) -> bool:
    if state.df is None or state.df.empty:
        return False
    last_col_is_target = (
        state.target_column in state.df.columns
        and state.df.columns[-1] == state.target_column
    )
    return last_col_is_target


def _report_done(state: PipelineState) -> bool:
    if state.output_dir is None:
        return False
    return (state.output_dir / "report.md").exists()


# ---------- inter-tool data hygiene ----------


def _coerce_object_numeric(state: PipelineState) -> int:
    """Spec §4 'Numeric placeholder normalization' / plan ambiguity #29.

    Coerce every column typed `numeric` whose pandas dtype is not numeric.
    `"NA"`-style tokens in numeric-meaning columns become real NaN so the
    imputer sees them. After coercion, refresh `state.profile[col]['null_rate']`
    and `dtype` because the imputer's drop-threshold check reads from the
    profile cache. Idempotent: a column already numeric is unchanged.
    Returns the number of columns coerced.
    """
    if state.df is None or state.column_types is None:
        return 0
    coerced = 0
    for col, typ in list(state.column_types.items()):
        if col not in state.df.columns or typ != "numeric":
            continue
        if pd.api.types.is_numeric_dtype(state.df[col]):
            continue
        state.df[col] = pd.to_numeric(state.df[col], errors="coerce")
        if state.profile is not None and col in state.profile:
            state.profile[col]["null_rate"] = float(state.df[col].isna().mean())
            state.profile[col]["dtype"] = str(state.df[col].dtype)
        coerced += 1
    return coerced


def profile_data() -> dict:
    """Run DataProfiler on the current dataset. Computes per-column stats and correlations."""
    def _do():
        DataProfiler()(_state)
        n_cols = sum(1 for k in _state.profile if not k.startswith("_"))
        return f"profiled {n_cols} columns"
    return _wrap("profile_data", _do, already_done=lambda s: s.profile is not None)


def infer_types() -> dict:
    """Run SemanticTypeInfer: assigns numeric/categorical/datetime/id/text/binary/target per column."""
    def _do():
        SemanticTypeInfer(_model_call)(_state)
        return f"typed {len(_state.column_types)} columns"
    return _wrap("infer_types", _do, already_done=_types_done)


def handle_missing() -> dict:
    """Run MissingValueHandler: pick strategy (median/mode/knn/iterative/drop) per column and fill.

    Before the imputer runs, the wrapper applies the spec §4 numeric-placeholder
    normalization pass: every column typed `numeric` with non-numeric dtype is
    coerced via `pd.to_numeric(..., errors='coerce')` so `"NA"`-style tokens
    become real NaN. The imputer then sees the nulls and decides per-column.
    """
    def _do():
        coerced = _coerce_object_numeric(_state)
        MissingValueHandler(_model_call)(_state)
        suffix = f", coerced {coerced} object→numeric" if coerced else ""
        return f"imputed; {len(_state.dropped_columns)} cols dropped total{suffix}"
    return _wrap("handle_missing", _do, already_done=_missing_done)


def handle_outliers() -> dict:
    """Run OutlierHandler: pick detector+action per numeric column and apply."""
    def _do():
        OutlierHandler(_model_call)(_state)
        return f"row_count={_state.row_count_after_outlier}"
    return _wrap("handle_outliers", _do, already_done=_outliers_done)


def create_features_pre() -> dict:
    """Run FeatureCreator.run_pre: execute pre_encoding feature operations (cross_categorical)."""
    def _do():
        _creator_instance.run_pre(_state)
        return f"pre_encoding done, df cols={len(_state.df.columns)}"
    return _wrap("create_features_pre", _do, already_done=_pre_done)


def encode_features() -> dict:
    """Run Encoder: LabelEncoder for categorical/binary columns and target."""
    def _do():
        Encoder()(_state)
        return "all columns numeric"
    return _wrap("encode_features", _do, already_done=_encoded_done)


def create_features_post() -> dict:
    """Run FeatureCreator.run_post: execute post_encoding feature operations."""
    def _do():
        _creator_instance.run_post(_state)
        return f"created {len(_state.created_columns)} columns total"
    return _wrap("create_features_post", _do, already_done=_post_done)


def scale_features() -> dict:
    """Run Scaler: pick scaler per numeric column and fit/transform."""
    def _do():
        Scaler(_model_call)(_state)
        return "scaling complete"
    return _wrap("scale_features", _do, already_done=_scaled_done)


def select_features() -> dict:
    """Run FeatureSelector: Judge picks per-cluster actions; code executes them."""
    def _do():
        _selector_instance(_state)
        return f"method={_state.selection_method}, k={len(_state.selected_columns)}"
    return _wrap("select_features", _do, already_done=_selected_done)


def validate_features() -> dict:
    """Run FeatureValidator: coerce to float (no LabelEncode), check NaNs, finalise output dataframe."""
    def _do():
        FeatureValidator()(_state)
        return f"final cols={len(_state.df.columns)}, rows={len(_state.df)}"
    return _wrap("validate_features", _do, already_done=_validated_done)


def write_report() -> dict:
    """Run FeatureReporter: produce report.md from structured pipeline summary."""
    def _do():
        FeatureReporter(_model_call)(_state)
        return "report.md written"
    return _wrap("write_report", _do, already_done=_report_done)


ALL_TOOLS = [
    profile_data,
    infer_types,
    handle_missing,
    handle_outliers,
    create_features_pre,
    encode_features,
    create_features_post,
    scale_features,
    select_features,
    validate_features,
    write_report,
]
