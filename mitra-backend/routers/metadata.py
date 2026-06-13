import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.metadata_gen_agent import MetadataGenAgent
from session import SessionManager

logger = logging.getLogger(__name__)
router = APIRouter()

_metadata_agent: Optional[MetadataGenAgent] = None


def get_metadata_agent() -> MetadataGenAgent:
    global _metadata_agent
    if _metadata_agent is None:
        _metadata_agent = MetadataGenAgent()
    return _metadata_agent


class MetadataRequest(BaseModel):
    session_id: str
    description: str
    target_col: Optional[str] = ""
    problem_type: Optional[str] = "auto"


async def _stream_metadata(request: MetadataRequest):
    if not SessionManager.session_exists(request.session_id):
        error_event = {"type": "error", "message": f"Session '{request.session_id}' not found"}
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    from main import llm_smoke_test_passed
    if not llm_smoke_test_passed:
        error_event = {"type": "error", "message": "LLM_SMOKE_TEST_FAILED"}
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    agent = get_metadata_agent()
    async for event in agent.run(
        session_id=request.session_id,
        description=request.description,
        target_col=request.target_col or None,
        problem_type=request.problem_type or "auto",
    ):
        yield f"data: {json.dumps(event)}\n\n"


@router.post("/metadata")
async def generate_metadata(request: MetadataRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_metadata(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
