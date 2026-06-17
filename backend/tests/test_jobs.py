import json

import pytest

from backend.jobs import JobEvent, JobRegistry, format_sse_event


def test_job_registry_stores_and_replays_events() -> None:
    registry = JobRegistry()
    registry.start_job(session_id="sid", job_type="validate")
    registry.append_event(
        session_id="sid",
        job_type="validate",
        event={"type": "check", "key": "format", "status": "pass"},
    )

    events = registry.get_events(session_id="sid", job_type="validate")

    assert len(events) == 1
    assert events[0].payload["key"] == "format"


def test_job_registry_replaces_existing_job_for_same_session_and_type() -> None:
    registry = JobRegistry()
    registry.start_job(session_id="sid", job_type="validate")
    registry.append_event(
        session_id="sid",
        job_type="validate",
        event={"type": "check", "key": "format"},
    )

    registry.start_job(session_id="sid", job_type="validate")

    assert registry.get_events(session_id="sid", job_type="validate") == []
    assert registry.get_state(session_id="sid", job_type="validate").status == "running"


def test_job_registry_marks_done_and_error() -> None:
    registry = JobRegistry()
    registry.start_job(session_id="sid", job_type="metadata")

    registry.mark_done(session_id="sid", job_type="metadata")

    assert registry.get_state(session_id="sid", job_type="metadata").status == "done"

    registry.start_job(session_id="sid", job_type="metadata")
    registry.mark_error(session_id="sid", job_type="metadata", message="failed")

    state = registry.get_state(session_id="sid", job_type="metadata")
    assert state.status == "error"
    assert state.events[-1].payload["message"] == "failed"


def test_format_sse_event_serializes_json_payload() -> None:
    payload = {"type": "done", "artifact": "validation_report.json"}

    sse_event = format_sse_event(payload)

    assert sse_event.startswith("data: ")
    assert sse_event.endswith("\n\n")
    assert json.loads(sse_event.removeprefix("data: ").strip()) == payload


def test_job_event_rejects_secret_payload_keys() -> None:
    safe_event = JobEvent(sequence=1, payload={"type": "check"})

    assert safe_event.payload == {"type": "check"}

    with pytest.raises(ValueError, match="Secret keys"):
        JobEvent(sequence=2, payload={"type": "check", "api_key": "secret"})
