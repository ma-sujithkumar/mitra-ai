"""Event-sink protocol and no-op implementation for optional SSE wiring."""

from __future__ import annotations

from typing import Protocol

from .contracts import TrainingEvent


class TrainingEventSink(Protocol):
    def emit(self, event: TrainingEvent) -> object:
        """Publish one event."""

    def close_session(self, session_id: str) -> None:
        """Signal that no more events will be produced for this session."""


class NullTrainingEventSink:
    """Default sink used when the API/event bus has not been connected."""

    def emit(self, event: TrainingEvent) -> None:
        del event

    def close_session(self, session_id: str) -> None:
        del session_id
