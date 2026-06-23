"""SSE endpoint for Epic-3 live training progress."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from backend.orchestration.events import TrainingEventBus

router = APIRouter(prefix="/api", tags=["training-events"])


def get_training_event_bus(request: Request) -> TrainingEventBus:
    return request.app.state.training_event_bus


@router.get("/training/events")
def stream_training_events(
    session_id: str = Query(min_length=1),
    replay: bool = True,
    heartbeat_sec: float = Query(default=15.0, gt=0.0, le=60.0),
    event_bus: TrainingEventBus = Depends(get_training_event_bus),
) -> StreamingResponse:
    """Replay existing events, then stream new events until the session closes."""

    subscription = event_bus.subscribe(session_id=session_id, replay=replay)
    return StreamingResponse(
        subscription.iter_sse(heartbeat_sec=heartbeat_sec),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
