"""Thread-safe, unbounded event bus and SSE subscriptions.

The training orchestrator and Ray collector run synchronously, while FastAPI
streams events to browsers.  A standard-library unbounded ``Queue`` provides a
safe boundary between those worlds.  Publishers never wait for a client, and a
disconnected subscription can be removed without affecting training.
"""

from __future__ import annotations

import json
from collections import defaultdict
from queue import Empty, Queue
from threading import RLock
from typing import Iterator

from .contracts import TrainingEvent

_SENTINEL = object()


class TrainingEventSubscription:
    """One independent consumer queue for a session's event stream."""

    def __init__(
        self,
        *,
        bus: "TrainingEventBus",
        session_id: str,
        event_queue: Queue[TrainingEvent | object],
    ) -> None:
        self._bus = bus
        self.session_id = session_id
        self._queue = event_queue
        self._closed = False

    def get(self, timeout: float | None = None) -> TrainingEvent | None:
        """Return the next event, or ``None`` when the stream is closed."""

        item = self._queue.get(timeout=timeout)
        if item is _SENTINEL:
            self._closed = True
            return None
        if not isinstance(item, TrainingEvent):
            return None
        return item

    def iter_sse(self, *, heartbeat_sec: float = 15.0) -> Iterator[str]:
        """Yield standards-compliant SSE frames until the session closes.

        Heartbeat comments keep proxies from considering an idle stream dead.
        ``finally`` always unsubscribes, including browser disconnects.
        """

        try:
            while not self._closed:
                try:
                    event = self.get(timeout=heartbeat_sec)
                except Empty:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    break
                yield format_training_sse(event)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            self._bus._unsubscribe(self.session_id, self._queue)
            return
        self._closed = True
        self._bus._unsubscribe(self.session_id, self._queue)


class TrainingEventBus:
    """In-memory replayable event bus with one unbounded queue per subscriber."""

    def __init__(self) -> None:
        self._history: dict[str, list[TrainingEvent]] = defaultdict(list)
        self._subscribers: dict[
            str, set[Queue[TrainingEvent | object]]
        ] = defaultdict(set)
        self._closed_sessions: set[str] = set()
        self._lock = RLock()

    def emit(self, event: TrainingEvent) -> TrainingEvent:
        """Publish an event without waiting for any SSE consumer."""

        with self._lock:
            if event.session_id in self._closed_sessions:
                # A completed session is immutable.  Silently ignore late
                # worker callbacks so they cannot disrupt training cleanup.
                return event
            sequenced = event.model_copy(
                update={"sequence": len(self._history[event.session_id]) + 1}
            )
            self._history[event.session_id].append(sequenced)
            subscribers = tuple(self._subscribers[event.session_id])

        for event_queue in subscribers:
            try:
                event_queue.put_nowait(sequenced)
            except Exception:
                # A broken/disconnected consumer must never block publishers.
                self._unsubscribe(event.session_id, event_queue)
        return sequenced

    def subscribe(
        self,
        session_id: str,
        *,
        replay: bool = True,
    ) -> TrainingEventSubscription:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")

        event_queue: Queue[TrainingEvent | object] = Queue(maxsize=0)
        with self._lock:
            history = tuple(self._history.get(session_id, ())) if replay else ()
            closed = session_id in self._closed_sessions
            if not closed:
                self._subscribers[session_id].add(event_queue)

        for event in history:
            event_queue.put_nowait(event)
        if closed:
            event_queue.put_nowait(_SENTINEL)

        return TrainingEventSubscription(
            bus=self,
            session_id=session_id,
            event_queue=event_queue,
        )

    def history(self, session_id: str) -> list[TrainingEvent]:
        with self._lock:
            return list(self._history.get(session_id, ()))

    def close_session(self, session_id: str) -> None:
        """Finish all current/future streams after already-published events."""

        with self._lock:
            self._closed_sessions.add(session_id)
            subscribers = tuple(self._subscribers.pop(session_id, set()))
        for event_queue in subscribers:
            try:
                event_queue.put_nowait(_SENTINEL)
            except Exception:
                pass

    def reset_session(self, session_id: str, *, clear_history: bool = True) -> None:
        """Re-open a session for an explicit retry/re-run."""

        with self._lock:
            self._closed_sessions.discard(session_id)
            if clear_history:
                self._history.pop(session_id, None)

    def subscriber_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._subscribers.get(session_id, ()))

    def _unsubscribe(
        self,
        session_id: str,
        event_queue: Queue[TrainingEvent | object],
    ) -> None:
        with self._lock:
            subscribers = self._subscribers.get(session_id)
            if subscribers is None:
                return
            subscribers.discard(event_queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)


def format_training_sse(event: TrainingEvent) -> str:
    """Serialize one typed event as a standard SSE frame."""

    payload = json.dumps(event.model_dump(mode="json"), sort_keys=True)
    return f"id: {event.sequence}\nevent: training\ndata: {payload}\n\n"
