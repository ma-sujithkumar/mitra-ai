"""Feature-engineering stage router.

Bridges the web flow to PipelinePrep (the same pre-training prep the CLI uses):
feature engineering -> Dataset2Vec query -> feature selection -> train/test split
-> model selection -> model_config.json. This is the stage the upload flow was
missing, which is why training previously ran on a 3-model deterministic fallback
with no feature engineering, no Dataset2Vec, and no agent reasoning.

Execution model: PipelinePrep.run() is long (multiple LLM calls), so it runs in a
single-worker background thread. The HTTP POST returns immediately with
status="accepted"; the frontend then polls:
  - GET /api/runs/{sid}/feature-engineering  (11-step detail + reasoning, files)
  - GET /api/feature-engineering/status       (job lifecycle: running/done/error)

Hard-fail policy: if PipelinePrep raises, the job is marked error and NO fallback
model_config.json is written, so the subsequent training call cannot proceed.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.activity_log import ActivityLog
from backend.agents.metadata_gen_agent import LlmSettings
from backend.agents.metadata_gen_agent import LlmSettingsResolver
from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader
from backend.dependencies import get_job_registry
from backend.dependencies import get_session_manager
from backend.jobs import JobRegistry
from backend.jobs import format_sse_event
from backend.services.pipeline_prep import PipelinePrep
from backend.session import SessionManager


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feature-engineering"])

JOB_TYPE = "feature_engineering"

# Single worker: only one feature-engineering run at a time (parallelization = 1).
# Module-level so the executor outlives the request that submits the job.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fe-prep")


class FeatureEngineeringRequest(BaseModel):
    session_id: str
    target_col: str | None = None
    problem_type: str | None = None
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    gateway_url: str | None = None


@router.post("/feature-engineering")
def start_feature_engineering(
    feature_request: FeatureEngineeringRequest,
    config_loader: ConfigLoader = Depends(get_config_loader),
    session_manager: SessionManager = Depends(get_session_manager),
    job_registry: JobRegistry = Depends(get_job_registry),
) -> dict[str, object]:
    """Kick off PipelinePrep (feature engineering + D2V + model selection) async."""
    session_path = _get_existing_session_path(
        session_manager=session_manager,
        session_id=feature_request.session_id,
    )
    metadata_path = session_path / "reports" / "metadata.json"
    if not metadata_path.is_file():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "METADATA_REQUIRED",
                "message": "Run metadata generation before feature engineering.",
            },
        )
    raw_data_path = session_path / "data" / "data.csv"

    # Resolve the prediction target: request wins, else fall back to metadata.
    target_column = _resolve_target_column(
        requested=feature_request.target_col,
        metadata_path=metadata_path,
    )
    if not target_column:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "TARGET_REQUIRED",
                "message": "A target column is required for feature engineering.",
            },
        )

    llm_settings = _resolve_llm_settings(
        feature_request=feature_request,
        config_loader=config_loader,
        job_registry=job_registry,
    )
    _ensure_credentials(
        feature_request=feature_request,
        llm_settings=llm_settings,
        job_registry=job_registry,
    )

    job_registry.start_job(session_id=feature_request.session_id, job_type=JOB_TYPE)
    job_registry.append_event(
        session_id=feature_request.session_id,
        job_type=JOB_TYPE,
        event={
            "type": "progress",
            "step": "llm_settings_resolved",
            "message": f"Using {llm_settings.provider}/{llm_settings.model}",
            "provider": llm_settings.provider,
            "model": llm_settings.model,
        },
    )
    ActivityLog(session_path=session_path).record(
        stage="feature_engineering",
        message=f"Feature engineering started (target={target_column})",
    )

    _executor.submit(
        _run_pipeline_prep,
        config_loader=config_loader,
        session_path=session_path,
        session_id=feature_request.session_id,
        raw_data_path=raw_data_path,
        metadata_path=metadata_path,
        target_column=target_column,
        llm_settings=llm_settings,
        job_registry=job_registry,
    )

    return {
        "session_id": feature_request.session_id,
        "status": "accepted",
        "llm": {
            "provider": llm_settings.provider,
            "model": llm_settings.model,
            "source": llm_settings.source,
        },
    }


@router.get("/feature-engineering/status")
def feature_engineering_status(
    session_id: str,
    job_registry: JobRegistry = Depends(get_job_registry),
) -> dict[str, object]:
    """Return the job lifecycle (running/done/error/idle) plus last error message."""
    state = job_registry.get_state(session_id=session_id, job_type=JOB_TYPE)
    last_error = next(
        (
            event.payload.get("message")
            for event in reversed(state.events)
            if event.payload.get("type") == "error"
        ),
        None,
    )
    return {
        "session_id": session_id,
        "status": state.status,
        "message": last_error,
    }


@router.get("/feature-engineering/events")
def stream_feature_engineering_events(
    session_id: str,
    job_registry: JobRegistry = Depends(get_job_registry),
) -> StreamingResponse:
    """Replay buffered job events (snapshot) as SSE, mirroring the metadata router."""
    events = job_registry.get_events(session_id=session_id, job_type=JOB_TYPE)
    return StreamingResponse(
        (format_sse_event(event.payload) for event in events),
        media_type="text/event-stream",
    )


def _run_pipeline_prep(
    *,
    config_loader: ConfigLoader,
    session_path: Path,
    session_id: str,
    raw_data_path: Path,
    metadata_path: Path,
    target_column: str,
    llm_settings: LlmSettings,
    job_registry: JobRegistry,
) -> None:
    """Background task: run PipelinePrep and record job lifecycle events.

    Runs in a worker thread; PipelinePrep's LLM calls create their own event loop
    (see feature_engineering/orchestrator._make_model_call), so this is thread-safe.
    """
    activity_log = ActivityLog(session_path=session_path)
    try:
        job_registry.append_event(
            session_id=session_id,
            job_type=JOB_TYPE,
            event={
                "type": "progress",
                "step": "feature_engineering",
                "message": "Running feature engineering, Dataset2Vec, and model selection",
            },
        )
        prep = PipelinePrep(
            config_loader=config_loader,
            session_dir=session_path,
            llm_settings=llm_settings,
        )
        model_config_path = prep.run(
            raw_data_path=raw_data_path,
            target_column=target_column,
            metadata_path=metadata_path,
            max_models=config_loader.pipeline.max_ml_models,
        )
        job_registry.append_event(
            session_id=session_id,
            job_type=JOB_TYPE,
            event={
                "type": "done",
                "step": "model_selection",
                "artifact": "model_config.json",
                "message": f"model_config.json written: {model_config_path.name}",
            },
        )
        job_registry.mark_done(session_id=session_id, job_type=JOB_TYPE)
        activity_log.record(
            stage="feature_engineering",
            message="Feature engineering + model selection complete",
        )
    except Exception as prep_error:  # noqa: BLE001 - surface any failure to the UI
        # Hard-fail: mark the job errored and write NO fallback model_config.json,
        # so the downstream training call cannot proceed on partial inputs.
        message = f"{type(prep_error).__name__}: {prep_error}"
        logger.exception("=> feature engineering failed: session=%s", session_id)
        job_registry.mark_error(session_id=session_id, job_type=JOB_TYPE, message=message)
        activity_log.record(
            stage="feature_engineering",
            level="ERROR",
            message=f"Feature engineering failed: {message}",
        )


def _resolve_target_column(requested: str | None, metadata_path: Path) -> str | None:
    if requested:
        return requested
    metadata = _read_json_or_none(metadata_path) or {}
    for key in ("target_col", "target_column"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    output_cols = metadata.get("output_cols")
    if isinstance(output_cols, list) and output_cols:
        return str(output_cols[0])
    return None


def _get_existing_session_path(
    session_manager: SessionManager,
    session_id: str,
) -> Path:
    try:
        session_path = session_manager.get_session_path(session_id=session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"},
        ) from exc
    if not (session_path / "data" / "data.csv").is_file():
        raise HTTPException(
            status_code=404,
            detail={"error": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"},
        )
    return session_path


def _resolve_llm_settings(
    feature_request: FeatureEngineeringRequest,
    config_loader: ConfigLoader,
    job_registry: JobRegistry,
) -> LlmSettings:
    resolver = LlmSettingsResolver(config_loader=config_loader)
    try:
        return resolver.resolve(
            provider=feature_request.provider,
            model=feature_request.model,
            api_key=feature_request.api_key,
            gateway_url=feature_request.gateway_url,
        )
    except ValueError as exc:
        job_registry.mark_error(
            session_id=feature_request.session_id,
            job_type=JOB_TYPE,
            message="LLM configuration is unavailable.",
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "LLM_CONFIGURATION_UNAVAILABLE",
                "message": "LLM configuration is unavailable.",
            },
        ) from exc


def _ensure_credentials(
    feature_request: FeatureEngineeringRequest,
    llm_settings: LlmSettings,
    job_registry: JobRegistry,
) -> None:
    if llm_settings.api_key or llm_settings.gateway_url:
        return
    job_registry.mark_error(
        session_id=feature_request.session_id,
        job_type=JOB_TYPE,
        message="LLM credentials required.",
    )
    raise HTTPException(
        status_code=503,
        detail={
            "error": "LLM_CREDENTIALS_REQUIRED",
            "message": "Provide an LLM API key or gateway URL.",
        },
    )


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
