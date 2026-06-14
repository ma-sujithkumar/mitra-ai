from __future__ import annotations

import json
import ssl
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agents.metadata_gen_agent import LlmSettings
from backend.agents.metadata_gen_agent import LlmSettingsResolver
from backend.agents.metadata_gen_agent import MetadataAgentRunner
from backend.agents.metadata_gen_agent import MetadataGenerationError
from backend.agents.metadata_gen_agent import MetadataGenerationInput
from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader
from backend.dependencies import get_job_registry
from backend.dependencies import get_metadata_agent_runner
from backend.dependencies import get_session_manager
from backend.jobs import JobRegistry
from backend.jobs import format_sse_event
from backend.session import SessionManager


router = APIRouter(prefix="/api", tags=["metadata"])


class MetadataRequest(BaseModel):
    session_id: str
    description: str | None = None
    target_col: str | None = None
    problem_type: str | None = None
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    gateway_url: str | None = None


@router.post("/metadata")
def start_metadata(
    metadata_request: MetadataRequest,
    config_loader: ConfigLoader = Depends(get_config_loader),
    session_manager: SessionManager = Depends(get_session_manager),
    job_registry: JobRegistry = Depends(get_job_registry),
    metadata_agent_runner: MetadataAgentRunner = Depends(get_metadata_agent_runner),
) -> dict[str, object]:
    session_path = _get_existing_session_path(
        session_manager=session_manager,
        session_id=metadata_request.session_id,
    )
    _load_passing_validation_report(session_path=session_path)
    job_registry.start_job(
        session_id=metadata_request.session_id,
        job_type="metadata",
    )

    llm_settings = _resolve_llm_settings(
        metadata_request=metadata_request,
        config_loader=config_loader,
        job_registry=job_registry,
    )
    _ensure_credentials(
        metadata_request=metadata_request,
        llm_settings=llm_settings,
        job_registry=job_registry,
    )
    user_metadata_context = _read_user_metadata_context(
        session_path=session_path,
        max_characters=config_loader.metadata_agent.metadata_context_char_limit,
    )

    job_registry.append_event(
        session_id=metadata_request.session_id,
        job_type="metadata",
        event={
            "type": "progress",
            "step": "llm_settings_resolved",
            "provider": llm_settings.provider,
            "model": llm_settings.model,
        },
    )
    generation_input = MetadataGenerationInput(
        session_id=metadata_request.session_id,
        workspace_root=config_loader.paths.workspace_root,
        llm_settings=llm_settings,
        description=metadata_request.description,
        target_col=metadata_request.target_col,
        problem_type=metadata_request.problem_type,
        user_metadata_context=user_metadata_context,
    )

    try:
        job_registry.append_event(
            session_id=metadata_request.session_id,
            job_type="metadata",
            event={
                "type": "progress",
                "step": "metadata_agent_started",
            },
        )
        result = metadata_agent_runner.generate_metadata(
            generation_input=generation_input
        )
    except MetadataGenerationError as exc:
        return _metadata_generation_failed(
            metadata_request=metadata_request,
            job_registry=job_registry,
            exception=exc,
        )
    except Exception as exc:
        return _metadata_generation_failed(
            metadata_request=metadata_request,
            job_registry=job_registry,
            exception=exc,
        )

    job_registry.append_event(
        session_id=metadata_request.session_id,
        job_type="metadata",
        event={
            "type": "done",
            "artifact": "metadata.json",
            "metadata_fields": sorted(result.metadata.keys()),
        },
    )
    job_registry.mark_done(
        session_id=metadata_request.session_id,
        job_type="metadata",
    )
    return {
        "session_id": metadata_request.session_id,
        "status": "accepted",
        "artifact": "metadata.json",
        "llm": {
            "provider": llm_settings.provider,
            "model": llm_settings.model,
            "source": llm_settings.source,
        },
    }


@router.get("/metadata/events")
def stream_metadata_events(
    session_id: str,
    job_registry: JobRegistry = Depends(get_job_registry),
) -> StreamingResponse:
    events = job_registry.get_events(session_id=session_id, job_type="metadata")
    return StreamingResponse(
        (format_sse_event(event.payload) for event in events),
        media_type="text/event-stream",
    )


def _get_existing_session_path(
    session_manager: SessionManager,
    session_id: str,
) -> Path:
    try:
        session_path = session_manager.get_session_path(session_id=session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Session not found: {session_id}",
            },
        ) from exc

    if not (session_path / "data" / "data.csv").is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Session not found: {session_id}",
            },
        )
    return session_path


def _load_passing_validation_report(session_path: Path) -> dict[str, Any]:
    validation_report_path = session_path / "reports" / "validation_report.json"
    if not validation_report_path.is_file():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "VALIDATION_REQUIRED",
                "message": "Run validation before metadata generation.",
            },
        )

    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if validation_report.get("passed") is not True:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "VALIDATION_FAILED",
                "message": "Metadata generation requires passing validation.",
            },
        )
    return validation_report


def _resolve_llm_settings(
    metadata_request: MetadataRequest,
    config_loader: ConfigLoader,
    job_registry: JobRegistry,
) -> LlmSettings:
    resolver = LlmSettingsResolver(config_loader=config_loader)
    try:
        return resolver.resolve(
            provider=metadata_request.provider,
            model=metadata_request.model,
            api_key=metadata_request.api_key,
            gateway_url=metadata_request.gateway_url,
        )
    except ValueError as exc:
        failure_message = _llm_configuration_failure_message(exception=exc)
        job_registry.mark_error(
            session_id=metadata_request.session_id,
            job_type="metadata",
            message=failure_message,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "LLM_CONFIGURATION_UNAVAILABLE",
                "message": failure_message,
            },
        ) from exc


def _llm_configuration_failure_message(exception: ValueError) -> str:
    if "LLM_CA_BUNDLE" in str(exception):
        return (
            "LLM_CA_BUNDLE must point to a PEM file containing at least one "
            "root or intermediate CA certificate."
        )
    return "LLM configuration is unavailable."


def _ensure_credentials(
    metadata_request: MetadataRequest,
    llm_settings: LlmSettings,
    job_registry: JobRegistry,
) -> None:
    if llm_settings.api_key or llm_settings.gateway_url:
        return

    job_registry.mark_error(
        session_id=metadata_request.session_id,
        job_type="metadata",
        message="LLM credentials required.",
    )
    raise HTTPException(
        status_code=503,
        detail={
            "error": "LLM_CREDENTIALS_REQUIRED",
            "message": "Provide an LLM API key or gateway URL.",
        },
    )


def _read_user_metadata_context(
    session_path: Path,
    max_characters: int,
) -> str | None:
    for metadata_filename in ["user_metadata.json", "user_metadata.csv"]:
        metadata_path = session_path / "data" / metadata_filename
        if metadata_path.is_file():
            return metadata_path.read_text(encoding="utf-8")[:max_characters]
    return None


def _metadata_generation_failed(
    metadata_request: MetadataRequest,
    job_registry: JobRegistry,
    exception: Exception,
) -> None:
    failure_message = _metadata_failure_message(exception=exception)
    job_registry.mark_error(
        session_id=metadata_request.session_id,
        job_type="metadata",
        message=failure_message,
    )
    raise HTTPException(
        status_code=503,
        detail={
            "error": "METADATA_GENERATION_FAILED",
            "message": failure_message,
        },
    ) from exception


def _metadata_failure_message(exception: Exception) -> str:
    if _has_llm_quota_error(exception=exception):
        return (
            "LLM provider quota exceeded or rate limited. Check the provider "
            "billing/quota for this API key, or choose another key, model, "
            "provider, or gateway."
        )
    if _has_ssl_certificate_error(exception=exception):
        return (
            "LLM HTTPS certificate verification failed. Configure LLM_CA_BUNDLE "
            "with a PEM bundle containing your local root CA and restart the backend."
        )
    return "Metadata generation failed."


def _has_llm_quota_error(exception: Exception) -> bool:
    for current_exception in _iter_exception_chain(exception=exception):
        exception_class_name = current_exception.__class__.__name__.lower()
        exception_message = str(current_exception).lower()
        if "ratelimit" in exception_class_name or "rate_limit" in exception_message:
            return True
        if "insufficient_quota" in exception_message:
            return True
        if "exceeded your current quota" in exception_message:
            return True
    return False


def _has_ssl_certificate_error(exception: Exception) -> bool:
    for current_exception in _iter_exception_chain(exception=exception):
        if isinstance(current_exception, ssl.SSLCertVerificationError):
            return True
        if "SSLCertVerificationError" in current_exception.__class__.__name__:
            return True
    return False


def _iter_exception_chain(exception: BaseException) -> list[BaseException]:
    visited_exception_ids: set[int] = set()
    current_exception: BaseException | None = exception
    exception_chain: list[BaseException] = []
    while current_exception is not None:
        current_exception_id = id(current_exception)
        if current_exception_id in visited_exception_ids:
            break
        visited_exception_ids.add(current_exception_id)
        exception_chain.append(current_exception)

        current_exception = current_exception.__cause__ or current_exception.__context__
    return exception_chain
