# Judge Agent - Input Format Requirement

This document defines the adapter schema that `run_judge.py` and `JudgeAgent` consume.
It is decoupled from all upstream tool output formats; the `UpstreamAdapter` translates
upstream JSONs (e.g. `overfitting_analysis.json`) into this schema.

---

## Top-level structure

```json
{
  "dataset_id": "<string or null>",
  "candidates": [ ... ],
  "minidata": { ... },
  "metadata": { ... }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `dataset_id` | string | No | Identifier for the dataset run. |
| `candidates` | list[CandidateModel] | Yes | 5-10 model entries to rank. |
| `minidata` | object | No | Output of `pd.describe()` as a dict. Context only - not scored. |
| `metadata` | object | No | User/pipeline metadata. Format is open; treated as opaque context. |

---

## CandidateModel

Each entry in `candidates`:

```json
{
  "model_name": "XGBClassifier",
  "task_type": "classification",
  "metrics": {
    "accuracy": 0.90,
    "f1_macro": 0.89,
    "f1_weighted": 0.90,
    "precision_macro": 0.90,
    "recall_macro": 0.89
  },
  "overfitting": {
    "is_overfitted": false,
    "gap": 0.09,
    "train_vs_cv_gap": 0.08
  },
  "complexity": {
    "n_params": 300000,
    "depth": 6,
    "family_rank": 9
  },
  "shap_summary": { ... },
  "hyperparam_sensitivity": { ... }
}
```

### `metrics`

Classification keys (use for `task_type: classification`):
- `accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`

Regression keys (use for `task_type: regression`):
- `mse`, `rmse`, `mae`, `r2`

Field names match `model_library/metrics/evaluators.py::MetricsResult`.

### `overfitting`

Adapted from the Overfitting Analysis Tool output (`overfitting_analysis.json`).

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `is_overfitted` | bool | Yes | From overfitting tool verdict. |
| `gap` | float | Yes | Primary metric gap (direction-aware). Positive => train > test. |
| `train_vs_cv_gap` | float or null | No | Train score minus CV mean. Null if K-fold was skipped. |

### `complexity`

Explicit complexity descriptor - must be supplied per model. Not inferred.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `n_params` | int | Yes | Number of model parameters (weights, trees, etc.). |
| `depth` | int | Yes | Max tree depth or network layer count. Use 0 for non-applicable models. |
| `family_rank` | int | Yes | Model family complexity rank: 1 = simplest, N = most complex. Example order: Linear=1, Ridge=2, KNN=3, DTree=4, RF=6, GBM=8, XGB=9. |

### `shap_summary` (context-only, optional)

Not scored in v1. Passed as context to the LLM for rationale and flag generation.

```json
{
  "top_features": ["feature_a", "feature_b"],
  "mean_abs_shap": {"feature_a": 0.42, "feature_b": 0.35},
  "feature_concentration": 0.77
}
```

### `hyperparam_sensitivity` (context-only, optional)

Not scored in v1. Passed as context to the LLM.

```json
{
  "learning_rate": {"0.01": 0.85, "0.1": 0.91},
  "sensitivity_score": 0.06,
  "most_sensitive_param": "learning_rate"
}
```

---

## Alternative: raw adapter-list format

`run_judge.py` also accepts the upstream raw format (a list of per-model dicts with
an `overfitting_json` key). This triggers `UpstreamAdapter.adapt_judge_input()`:

```json
{
  "dataset_id": "...",
  "candidate_models": [
    {
      "model_name": "...",
      "task_type": "...",
      "metrics": { ... },
      "overfitting_json": { <exact overfitting_analysis.json content> },
      "complexity": { ... },
      "shap_summary": { ... },
      "hyperparam_sensitivity": { ... }
    }
  ],
  "minidata": { ... },
  "metadata": { ... }
}
```

The adapter maps `overfitting_json` (the upstream tool's raw output) into the
`OverfittingInfo` fields above.
