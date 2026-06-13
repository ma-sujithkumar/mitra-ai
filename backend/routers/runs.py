from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import Depends

from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader
from backend.dependencies import get_session_manager
from backend.session import SessionManager


router = APIRouter(prefix="/api", tags=["runs"])


@router.get("/runs")
def list_runs(
    limit: int | None = None,
    config_loader: ConfigLoader = Depends(get_config_loader),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, object]:
    resolved_limit = limit or config_loader.upload.recent_upload_limit
    uploads = session_manager.list_recent_uploads(limit=resolved_limit)
    runs = [
        _build_run_record(
            workspace_root=config_loader.paths.workspace_root,
            upload_record=upload_record,
        )
        for upload_record in uploads
    ]
    return {"runs": runs}


@router.get("/runs/stats")
def run_stats(
    config_loader: ConfigLoader = Depends(get_config_loader),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, object]:
    uploads = session_manager.list_recent_uploads(limit=1000000)
    run_records = [
        _build_run_record(
            workspace_root=config_loader.paths.workspace_root,
            upload_record=upload_record,
        )
        for upload_record in uploads
    ]
    return {
        "total_uploads": len(run_records),
        "validated_runs": sum(
            1 for run_record in run_records
            if run_record["validation_status"] == "passed"
        ),
        "metadata_runs": sum(
            1 for run_record in run_records
            if run_record["metadata_status"] == "complete"
        ),
        "leaderboard_runs": 0,
    }


def _build_run_record(
    workspace_root: Path,
    upload_record: dict[str, object],
) -> dict[str, object]:
    session_id = str(upload_record["session_id"])
    reports_dir = workspace_root / session_id / "reports"
    return {
        **upload_record,
        "validation_status": _validation_status(reports_dir=reports_dir),
        "metadata_status": _metadata_status(reports_dir=reports_dir),
        "leaderboard_status": "pending",
    }


def _validation_status(reports_dir: Path) -> str:
    validation_report_path = reports_dir / "validation_report.json"
    if not validation_report_path.is_file():
        return "pending"

    validation_report = _read_json(path=validation_report_path)
    if validation_report.get("passed") is True:
        return "passed"
    return "failed"


def _metadata_status(reports_dir: Path) -> str:
    if (reports_dir / "metadata.json").is_file():
        return "complete"
    return "pending"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
