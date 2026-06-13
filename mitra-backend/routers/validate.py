import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from session import SessionManager
from validator import DataValidator

logger = logging.getLogger(__name__)
router = APIRouter()


class ValidateRequest(BaseModel):
    session_id: str
    target_col: Optional[str] = ""


async def _stream_validation(session_id: str, target_col: str) -> AsyncGenerator[str, None]:
    if not SessionManager.session_exists(session_id):
        error_event = {"type": "error", "message": f"Session '{session_id}' not found"}
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    validator = DataValidator()
    loop = asyncio.get_event_loop()

    def run_validation():
        return list(validator.validate(session_id, target_col or None))

    results = await loop.run_in_executor(None, run_validation)

    for result in results:
        event_data = {
            "type": "check",
            "key": result.key,
            "status": result.status,
            "detail": result.detail,
            **({"warn_message": result.warn_message} if result.warn_message else {}),
        }
        yield f"data: {json.dumps(event_data)}\n\n"
        await asyncio.sleep(0)

    done_event = {"type": "done", "artifact": "validation_report.json"}
    yield f"data: {json.dumps(done_event)}\n\n"


@router.post("/validate")
async def validate_dataset(request: ValidateRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_validation(request.session_id, request.target_col or ""),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
