# Epic-3 Local Training Pipeline — Owner: Onkar

This package implements only the local, single-model training work item.
It consumes the `TrainingJob` produced by Subhasis's router/orchestrator and
returns a structured `TrainingResult`.

## Flow

```text
TrainingJob
  -> load pre-split CSV or NPZ data
  -> validate exact model against MODEL_REGISTRY
  -> train through MLKit
  -> compute train and validation metrics
  -> save model.pkl
  -> write train_metrics.json
  -> return TrainingResult
```

Ray execution, SSE events, Page-2 UI, and session-level aggregation are not part
of this package.

## Data hand-off

- CSV files require a header and numeric feature columns. Pass `--target-column`
  when the target is not named `target`, `label`, `species`, `class`, `y`, or
  `output`; otherwise the final column is used.
- NPZ files may contain `X`/`y`, or split-specific keys
  `X_train`/`y_train` and `X_test`/`y_test`.

## Run one job

```bash
python -m epic_3.training.cli \
  --job path/to/training_job.json \
  --model-library-root model_library \
  --target-column species
```

Artifacts are written inside the job's `output_dir`:

```text
model.pkl
train_metrics.json
```

## Tests

```bash
python -m pytest -q epic_3/training/tests
```
