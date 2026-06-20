# MITRA Epic-3 Training SSE Events

**Owner: Onkar**

This package publishes live model-training lifecycle events without coupling
training success to a connected browser. The event bus uses one unbounded queue
per subscriber, keeps replay history per session, and safely removes clients
that disconnect.

## Lifecycle

```text
queued -> submitted -> running -> completed | failed | timed_out | cancelled
                                                        |
                                                        -> all_completed
```

Local execution emits `queued`, `running`, a terminal model event, and the final
`all_completed` event. Ray execution additionally emits `submitted`, and the
Ray collector invokes a callback as each model finishes so completion-order
updates are available immediately.

## Event schema

```json
{
  "session_id": "iris-ray-demo",
  "stage": "training",
  "level": "info",
  "msg": "RandomForestClassifier training completed",
  "pct": 100,
  "status": "completed",
  "ts": "2026-06-17T14:30:00Z",
  "sequence": 8,
  "model_id": "model_001",
  "model_name": "RandomForestClassifier",
  "details": {
    "validation_score": 0.93,
    "training_time_sec": 1.42,
    "model_path": ".mitra/iris-ray-demo/model_001/model.pkl"
  }
}
```

## FastAPI endpoint

```text
GET /api/training/events?session_id=<session-id>
Content-Type: text/event-stream
```

The endpoint replays already-published events and then waits for new ones. The
stream closes after the orchestrator writes `training_summary.json` and emits
`all_completed`. Heartbeat comments keep idle connections alive.

## Programmatic integration

```python
from epic_3.events import TrainingEventBus
from epic_3.training_orchestrator import TrainingOrchestrator

bus = TrainingEventBus()
orchestrator = TrainingOrchestrator("model_library", event_sink=bus)
summary = orchestrator.prepare_and_execute_ray(...)
```

Event publication is best-effort. A broken event sink or disconnected browser
never changes model execution, persisted job state, or the final summary.

## Frontend helper

```javascript
import { streamTrainingEvents } from './api/events';

const source = streamTrainingEvents(sessionId, {
  onEvent: (event) => console.log(event),
  onDone: (event) => console.log('training finished', event),
});
```

## Tests

```bash
python -m pytest -q epic_3/events/tests
```
