"""ADK tool wrappers. Each function is registered as a tool on the orchestrator agent.

The shared PipelineState is injected once at startup via set_pipeline_state().
Tools mutate state in place and return small dicts so the ADK agent can decide what to call next.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable

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


def set_pipeline_state(
    state: PipelineState,
    model_call: Callable[[str], str],
    judge_agent: Any | None = None,
) -> None:
    global _state, _model_call, _log_path, _creator_instance
    _state = state
    _model_call = model_call
    _log_path = str(state.output_dir / "execution_log.txt")
    _creator_instance = FeatureCreator(model_call, judge=judge_agent)


def _log(tool: str, status: str, detail: str, elapsed: float) -> None:
    if _log_path is None:
        return
    # Collapse newlines and runs of whitespace so each tool stays on one line.
    one_line = " | ".join(s.strip() for s in str(detail).splitlines() if s.strip())
    with open(_log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {tool} {status} ({elapsed:.2f}s) {one_line}\n")


def _wrap(tool_name: str, fn: Callable[[], Any]) -> dict:
    if _state is None:
        return {"status": "error", "detail": "pipeline state not initialised"}
    start = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - start
        _log(tool_name, "ok", str(result) if result else "", elapsed)
        return {"status": "ok", "detail": result if result else "completed"}
    except (PreconditionError, PostconditionError) as e:
        elapsed = time.perf_counter() - start
        _log(tool_name, "error", str(e), elapsed)
        return {"status": "error", "detail": str(e)}
    except Exception as e:
        elapsed = time.perf_counter() - start
        _log(tool_name, "error", f"{e}\n{traceback.format_exc()}", elapsed)
        return {"status": "error", "detail": str(e)}


def profile_data() -> dict:
    """Run DataProfiler on the current dataset. Computes per-column stats and correlations."""
    def _do():
        DataProfiler()(_state)
        return f"profiled {len(_state.profile) - 1} columns"
    return _wrap("profile_data", _do)


def infer_types() -> dict:
    """Run SemanticTypeInfer: assigns numeric/categorical/datetime/id/text/binary/target per column."""
    def _do():
        SemanticTypeInfer(_model_call)(_state)
        return f"typed {len(_state.column_types)} columns"
    return _wrap("infer_types", _do)


def handle_missing() -> dict:
    """Run MissingValueHandler: pick strategy (median/mode/knn/iterative/drop) per column and fill."""
    def _do():
        MissingValueHandler(_model_call)(_state)
        return f"imputed; {len(_state.dropped_columns)} cols dropped total"
    return _wrap("handle_missing", _do)


def handle_outliers() -> dict:
    """Run OutlierHandler: pick detector+action per numeric column and apply."""
    def _do():
        OutlierHandler(_model_call)(_state)
        return f"row_count={_state.row_count_after_outlier}"
    return _wrap("handle_outliers", _do)


def create_features_pre() -> dict:
    """Run FeatureCreator.run_pre: execute pre_encoding feature operations (cross_categorical)."""
    def _do():
        _creator_instance.run_pre(_state)
        return f"pre_encoding done, df cols={len(_state.df.columns)}"
    return _wrap("create_features_pre", _do)


def encode_features() -> dict:
    """Run Encoder: LabelEncoder for categorical/binary columns and target."""
    def _do():
        Encoder()(_state)
        return "all columns numeric"
    return _wrap("encode_features", _do)


def create_features_post() -> dict:
    """Run FeatureCreator.run_post: execute post_encoding feature operations."""
    def _do():
        _creator_instance.run_post(_state)
        return f"created {len(_state.created_columns)} columns total"
    return _wrap("create_features_post", _do)


def scale_features() -> dict:
    """Run Scaler: pick scaler per numeric column and fit/transform."""
    def _do():
        Scaler(_model_call)(_state)
        return "scaling complete"
    return _wrap("scale_features", _do)


def select_features() -> dict:
    """Run FeatureSelector: pick method and select top-k features."""
    def _do():
        FeatureSelector(_model_call)(_state)
        return f"method={_state.selection_method}, k={len(_state.selected_columns)}"
    return _wrap("select_features", _do)


def validate_features() -> dict:
    """Run FeatureValidator: coerce to float, check NaNs, finalise output dataframe."""
    def _do():
        FeatureValidator()(_state)
        return f"final cols={len(_state.df.columns)}, rows={len(_state.df)}"
    return _wrap("validate_features", _do)


def write_report() -> dict:
    """Run FeatureReporter: produce report.md from structured pipeline summary."""
    def _do():
        FeatureReporter(_model_call)(_state)
        return "report.md written"
    return _wrap("write_report", _do)


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
