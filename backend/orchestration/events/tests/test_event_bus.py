from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from backend.orchestration.events import TrainingEvent, TrainingEventBus, format_training_sse


def _event(*, status: str = "running", model_id: str | None = "model_001") -> TrainingEvent:
    return TrainingEvent(
        session_id="session-1",
        model_id=model_id,
        model_name="RandomForestClassifier" if model_id else None,
        status=status,
        msg=f"status={status}",
        pct=25 if status == "running" else 100,
    )


def test_event_contract_validates_required_fields_and_progress() -> None:
    with pytest.raises(ValidationError):
        TrainingEvent(
            session_id="",
            status="running",
            msg="bad",
            pct=101,
        )


def test_bus_replays_ordered_events_and_closes_stream() -> None:
    bus = TrainingEventBus()
    first = bus.emit(_event(status="running"))
    second = bus.emit(_event(status="completed"))
    bus.close_session("session-1")

    assert first.sequence == 1
    assert second.sequence == 2

    subscription = bus.subscribe("session-1", replay=True)
    frames = list(subscription.iter_sse(heartbeat_sec=0.01))

    assert len(frames) == 2
    assert frames[0].startswith("id: 1\nevent: training\ndata: ")
    payload = json.loads(frames[1].split("data: ", 1)[1])
    assert payload["status"] == "completed"
    assert payload["sequence"] == 2


def test_disconnected_subscriber_does_not_block_publishers() -> None:
    bus = TrainingEventBus()
    subscription = bus.subscribe("session-1", replay=False)
    assert bus.subscriber_count("session-1") == 1

    subscription.close()
    assert bus.subscriber_count("session-1") == 0

    published = bus.emit(_event(status="completed"))
    assert published.sequence == 1
    assert [item.status for item in bus.history("session-1")] == ["completed"]


def test_each_subscriber_receives_its_own_copy() -> None:
    bus = TrainingEventBus()
    first = bus.subscribe("session-1", replay=False)
    second = bus.subscribe("session-1", replay=False)

    bus.emit(_event(status="running"))

    assert first.get(timeout=0.1).status == "running"
    assert second.get(timeout=0.1).status == "running"
    first.close()
    second.close()


def test_format_training_sse_contains_no_secret_fields() -> None:
    event = _event(status="completed")
    frame = format_training_sse(event.model_copy(update={"sequence": 4}))

    assert "id: 4" in frame
    assert "event: training" in frame
    assert "api_key" not in frame
