# MITRA Training Orchestrator — Local Result Integration

**Owner: Subhasis**

This package connects Model Selection/Model Routing to Onkar's local training
worker. It now covers both job preparation and local result aggregation.

## Flow

```text
metadata.json + model_config.json + Epic-2 train/test splits
  -> validate selected models against MODEL_REGISTRY
  -> create TrainingJob objects
  -> write training_jobs.json
  -> update queued -> running
  -> call LocalTrainingWorker.run(job)
  -> update completed / failed
  -> isolate per-model failures
  -> write training_summary.json
```

Ray execution, cluster/resource management, SSE events, and Page-2 UI remain
separate Epic-3 work items.

## Inputs

- `metadata.json`
- `model_config.json`
- Epic-2 train split
- Epic-2 test split

## Outputs

```text
.mitra/<session-id>/
├── training_jobs.json          # updated after every job-state transition
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

One model failure does not stop later models. The final summary status is one
of `completed`, `partial_failure`, or `failed`.

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

Example `training_summary.json`:

```json
{
  "session_id": "iris-demo",
  "status": "partial_failure",
  "total_models": 3,
  "completed": 2,
  "failed": 1,
  "models": [
    {
      "model_id": "model_001",
      "model_name": "RandomForestClassifier",
      "status": "completed",
      "validation_score": 0.93,
      "model_path": "/tmp/.mitra/iris-demo/model_001/model.pkl",
      "training_time_sec": 1.2,
      "metrics": {},
      "error": null
    }
  ]
}
```

## Tests

```bash
python -m pytest -q \
  epic_3/model_selection/tests \
  epic_3/training_orchestrator/tests \
  epic_3/training/tests
```
