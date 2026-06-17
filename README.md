# deeplearning-repo

## Team Members (In No Order):

1. Sujithkumar M A, Texas Instruments
2. ⁠Avinash Bhargav, Siemens
3. ⁠Shiva Priya, Bosch
4. ⁠Meena M, Bosch
5. ⁠Sebin Francis, Cisco
6. ⁠Onkar Shamsunder Biyani, SMILe
7. ⁠Subhasis Mahana, Samsung
8. ⁠Vidhi Kant Gupta, NPCI

## Implemented Components

- [Epic-3 Model Selection](epic_3/model_selection/README.md) — agent-based,
  registry-constrained selection. Every emitted `model_name` is validated against
  `model_library/ml_kit.py::MODEL_REGISTRY` before `model_config.json` is written.
- [Epic-3 Local Training Pipeline](epic_3/training/README.md) — consumes one
  `TrainingJob`, trains the exact MLKit registry model, writes `model.pkl` and
  `train_metrics.json`, and returns a typed `TrainingResult`.
- [Epic-3 Training Orchestrator](epic_3/training_orchestrator/README.md) —
  validates and routes models, prepares `training_jobs.json`, integrates the
  local training worker, persists job statuses, isolates per-model failures,
  and writes `training_summary.json`.
- [Epic-3 Training SSE Events](epic_3/events/README.md) — publishes
  replayable `queued`/`running`/terminal events, streams them through
  `/api/training/events`, and keeps training independent of browser clients.

## Epic-3 Page 2 live training UI

The Pipeline navigation now opens the live Page-2 training view. It subscribes
to `GET /api/training/events?session_id=<id>`, replays queued events, displays
one card per selected model, and updates status, progress, validation score,
training duration, artifact path, and failure details without a page refresh.
The browser's native `EventSource` reconnection remains enabled for temporary
network interruptions. The final `all_completed` event unlocks navigation to
the leaderboard.

Frontend verification:

```bash
cd frontend
npm test
npm run build
```
