# MITRA Training Orchestrator — Local + Ray Result Integration

**Owner: Subhasis**

This package connects Model Selection/Model Routing to Onkar's local training
worker and Ray infrastructure. It owns job lifecycle state and session-level
result aggregation while the Ray wrapper owns cluster lifecycle, resources,
timeouts, and remote worker execution.

## Flow

```text
metadata.json + model_config.json + Epic-2 train/test splits
  -> validate selected models against MODEL_REGISTRY
  -> create TrainingJob objects
  -> write training_jobs.json
  -> update queued -> running
  -> execute locally OR submit all jobs to Ray
  -> collect TrainingResult objects (completion order may differ)
  -> map results back by model_id
  -> update completed / failed
  -> isolate per-model failures and timeouts
  -> emit live SSE lifecycle events
  -> write training_summary.json and close the event stream
```

## Inputs

- `metadata.json`
- `model_config.json`
- Epic-2 train split
- Epic-2 test split

## Outputs

```text
.mitra/<session-id>/
├── training_jobs.json          # persisted lifecycle state
├── training_summary.json       # session-level aggregation
├── model_001/
│   ├── model.pkl
│   └── train_metrics.json
└── model_002/
    ├── model.pkl               # present only when training succeeds
    └── train_metrics.json
```

Supported job states:

```text
queued -> running -> completed | failed
```

One failed/timed-out model does not remove successful results. The final summary
status is `completed`, `partial_failure`, or `failed`.

## Prepare jobs only

```bash
python -m epic_3.training_orchestrator.cli \
  --session-id iris-demo \
  --metadata epic_3/model_selection/fixtures/iris_metadata.json \
  --model-config /tmp/model_config.json \
  --train /tmp/train.csv \
  --test /tmp/test.csv \
  --session-dir /tmp/.mitra/iris-demo \
  --model-library-root model_library
```

## Prepare and execute locally

```bash
python -m epic_3.training_orchestrator.cli \
  --session-id iris-demo \
  --metadata epic_3/model_selection/fixtures/iris_metadata.json \
  --model-config /tmp/model_config.json \
  --train /tmp/train.csv \
  --test /tmp/test.csv \
  --session-dir /tmp/.mitra/iris-demo \
  --model-library-root model_library \
  --target-column species \
  --execute-local
```

## Prepare and execute with Ray

```bash
python -m epic_3.training_orchestrator.cli \
  --session-id iris-ray-demo \
  --metadata epic_3/model_selection/fixtures/iris_metadata.json \
  --model-config /tmp/model_config.json \
  --train /tmp/train.csv \
  --test /tmp/test.csv \
  --session-dir /tmp/.mitra/iris-ray-demo \
  --model-library-root model_library \
  --target-column species \
  --ray-timeout-sec 300 \
  --execute-ray
```

Ray settings are loaded from the root `config.ini` `[ray]` section. Local mode
remains available as an explicit fallback when Ray is not desired.

## Programmatic Ray integration

```python
from epic_3.events import TrainingEventBus

bus = TrainingEventBus()
summary = TrainingOrchestrator(
    "model_library",
    event_sink=bus,
).execute_ray(
    manifest,
    target_column="species",
    timeout_sec=300,
)
```

The orchestrator lazily imports `RayExecutor`, calls `start()`, `submit_all()`,
`collect()`, persists final statuses, writes the summary, and safely closes the
executor. An injected executor may be kept open with `close_executor=False`.

## Tests

```bash
python -m pytest -q \
  epic_3/model_selection/tests \
  epic_3/training_orchestrator/tests \
  epic_3/training/tests \
  epic_3/ray_wrapper/tests \
  epic_3/events/tests
```
