from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from threading import Lock
from typing import Any


SECRET_PAYLOAD_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "token",
}


@dataclass(frozen=True)
class JobEvent:
    sequence: int
    payload: dict[str, Any]
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        secret_keys = [
            key
            for key in self.payload
            if key.lower() in SECRET_PAYLOAD_KEYS
        ]
        if secret_keys:
            raise ValueError(f"Secret keys are not allowed in job events: {secret_keys}")


@dataclass
class JobState:
    session_id: str
    job_type: str
    status: str
    events: list[JobEvent] = field(default_factory=list)


class JobRegistry:
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], JobState] = {}
        self._lock = Lock()

    def start_job(self, session_id: str, job_type: str) -> JobState:
        with self._lock:
            state = JobState(
                session_id=session_id,
                job_type=job_type,
                status="running",
            )
            self._states[(session_id, job_type)] = state
            return state

    def append_event(
        self,
        session_id: str,
        job_type: str,
        event: dict[str, Any],
    ) -> JobEvent:
        with self._lock:
            state = self._get_or_create_state(
                session_id=session_id,
                job_type=job_type,
            )
            job_event = JobEvent(
                sequence=len(state.events) + 1,
                payload=event,
            )
            state.events.append(job_event)
            return job_event

    def mark_done(self, session_id: str, job_type: str) -> None:
        with self._lock:
            state = self._get_or_create_state(
                session_id=session_id,
                job_type=job_type,
            )
            state.status = "done"

    def mark_error(self, session_id: str, job_type: str, message: str) -> None:
        with self._lock:
            state = self._get_or_create_state(
                session_id=session_id,
                job_type=job_type,
            )
            state.status = "error"
            job_event = JobEvent(
                sequence=len(state.events) + 1,
                payload={"type": "error", "message": message},
            )
            state.events.append(job_event)

    def get_state(self, session_id: str, job_type: str) -> JobState:
        with self._lock:
            return self._get_or_create_state(
                session_id=session_id,
                job_type=job_type,
            )

    def get_events(self, session_id: str, job_type: str) -> list[JobEvent]:
        with self._lock:
            state = self._get_or_create_state(
                session_id=session_id,
                job_type=job_type,
            )
            return list(state.events)

    def _get_or_create_state(self, session_id: str, job_type: str) -> JobState:
        state_key = (session_id, job_type)
        if state_key not in self._states:
            self._states[state_key] = JobState(
                session_id=session_id,
                job_type=job_type,
                status="idle",
            )
        return self._states[state_key]


def format_sse_event(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, sort_keys=True)}\n\n"
