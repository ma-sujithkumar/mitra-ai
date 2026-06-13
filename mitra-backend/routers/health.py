import time
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_server_start_time = time.time()


@router.get("/health")
async def health_check() -> JSONResponse:
    from main import llm_smoke_test_passed

    uptime_seconds = int(time.time() - _server_start_time)
    return JSONResponse(content={
        "llm_smoke_test": "ok" if llm_smoke_test_passed else "failed",
        "uptime_seconds": uptime_seconds,
    })
