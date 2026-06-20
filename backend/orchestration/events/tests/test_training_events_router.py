from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.training_events import router
from backend.orchestration.events import TrainingEvent, TrainingEventBus


def test_training_sse_endpoint_replays_and_finishes_closed_session() -> None:
    bus = TrainingEventBus()
    bus.emit(
        TrainingEvent(
            session_id="api-session",
            model_id="model_001",
            model_name="LogisticRegression",
            status="completed",
            msg="training completed",
            pct=100,
        )
    )
    bus.emit(
        TrainingEvent(
            session_id="api-session",
            status="all_completed",
            msg="all models completed",
            pct=100,
        )
    )
    bus.close_session("api-session")

    app = FastAPI()
    app.state.training_event_bus = bus
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/training/events?session_id=api-session")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert "event: training" in response.text
    assert '"status": "completed"' in response.text
    assert '"status": "all_completed"' in response.text
