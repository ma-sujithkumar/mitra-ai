# MITRA Training Orchestrator — Job Preparation

This work item is the boundary between Model Selection and the training/Ray
implementation.

It reads:

- `metadata.json`
- `model_config.json`
- Epic-2 train and test split paths

It writes:

- `training_jobs.json`
- one empty output directory per queued model

The router validates every selected model against
`model_library/ml_kit.py::MODEL_REGISTRY`, uses model-library defaults as the
source of truth, and produces the `TrainingJob` contract that the training/Ray
layer will consume.

No estimator is instantiated and no training or Ray execution happens in this
component.

## Run

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

## Test

```bash
pytest -q epic_3/training_orchestrator/tests
```
