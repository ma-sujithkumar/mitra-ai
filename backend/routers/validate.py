from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader
from backend.dependencies import get_job_registry
from backend.dependencies import get_session_manager
from backend.jobs import JobRegistry
from backend.jobs import format_sse_event
from backend.session import SessionManager
from backend.user_metadata import find_user_metadata_path
from backend.validator import DataValidator


router = APIRouter(prefix="/api", tags=["validate"])


class ValidationRequest(BaseModel):
    session_id: str
    target_col: str | None = None
    validation_split: float | None = None


@router.post("/validate")
def start_validation(
    validation_request: ValidationRequest,
    config_loader: ConfigLoader = Depends(get_config_loader),
    session_manager: SessionManager = Depends(get_session_manager),
    job_registry: JobRegistry = Depends(get_job_registry),
) -> dict[str, object]:
    session_path = session_manager.get_session_path(
        session_id=validation_request.session_id
    )
    data_file = session_path / "data" / "data.csv"
    if not data_file.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Session not found: {validation_request.session_id}",
            },
        )

    validation_split = (
        validation_request.validation_split
        if validation_request.validation_split is not None
        else config_loader.pipeline.train_test_split
    )
    _write_json(
        path=session_path / "reports" / "run_config.json",
        data={
            "session_id": validation_request.session_id,
            "target_col": validation_request.target_col,
            "validation_split": validation_split,
        },
    )

    job_registry.start_job(
        session_id=validation_request.session_id,
        job_type="validate",
    )
    validator = DataValidator(
        min_rows=config_loader.upload.min_rows,
        null_threshold=config_loader.upload.null_threshold,
        pii_patterns=config_loader.upload.pii_patterns,
        metadata_match_min_overlap=config_loader.upload.metadata_match_min_overlap,
        chunk_size_rows=config_loader.upload.chunk_size_rows,
    )
    user_metadata_path = find_user_metadata_path(session_path=session_path)
    check_results = list(
        validator.validate(
            data_file=data_file,
            session_id=validation_request.session_id,
            target_col=validation_request.target_col,
            user_metadata_path=user_metadata_path,
        )
    )
    for check_result in check_results:
        job_registry.append_event(
            session_id=validation_request.session_id,
            job_type="validate",
            event={
                "type": "check",
                **check_result.to_dict(),
            },
        )

    validation_report = validator.build_report(
        session_id=validation_request.session_id,
        checks=check_results,
    )
    _write_json(
        path=session_path / "reports" / "validation_report.json",
        data=validation_report.to_dict(),
    )
    job_registry.append_event(
        session_id=validation_request.session_id,
        job_type="validate",
        event={
            "type": "done",
            "artifact": "validation_report.json",
            "passed": validation_report.passed,
        },
    )
    job_registry.mark_done(
        session_id=validation_request.session_id,
        job_type="validate",
    )

    return {
        "session_id": validation_request.session_id,
        "status": "accepted",
    }


@router.get("/validate/events")
def stream_validation_events(
    session_id: str,
    job_registry: JobRegistry = Depends(get_job_registry),
) -> StreamingResponse:
    events = job_registry.get_events(session_id=session_id, job_type="validate")
    return StreamingResponse(
        (format_sse_event(event.payload) for event in events),
        media_type="text/event-stream",
    )


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
