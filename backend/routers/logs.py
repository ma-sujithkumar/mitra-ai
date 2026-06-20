from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import FileResponse

from backend.activity_log import ACTIVITY_LOG_FILENAME
from backend.activity_log import ActivityLog
from backend.dependencies import get_session_manager
from backend.session import SessionManager


router = APIRouter(prefix="/api", tags=["logs"])


@router.get("/runs/{session_id}/activity")
def read_activity_log(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, object]:
    session_path = _require_session_path(
        session_manager=session_manager, session_id=session_id
    )
    activity_log = ActivityLog(session_path=session_path)
    return {"session_id": session_id, "entries": activity_log.read()}


@router.get("/logs/download/{session_id}")
def download_activity_log(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> FileResponse:
    session_path = _require_session_path(
        session_manager=session_manager, session_id=session_id
    )
    log_path = session_path / ACTIVITY_LOG_FILENAME
    if not log_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "ACTIVITY_LOG_NOT_FOUND",
                "message": f"No activity log for session: {session_id}",
            },
        )
    return FileResponse(
        path=log_path,
        media_type="text/plain",
        filename=f"{session_id}_activity.log",
    )


def _require_session_path(
    session_manager: SessionManager,
    session_id: str,
):
    session_path = session_manager.get_session_path(session_id=session_id)
    if not session_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Session not found: {session_id}",
            },
        )
    return session_path
