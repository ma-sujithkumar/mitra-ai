# Overfitting Analysis Tool

Analyzes whether a trained ML model has overfit by computing direction-aware metric gaps
(train vs. test) and K-fold cross validation. Designed as a step in an AutoML pipeline.

Backed by `model_library` (MLKit + compute_metrics) — no training or metric logic is
reimplemented here.

---

## Prerequisites

- Python binary: `/home/sujithma/venv/bin/python` (set in `config/config.ini`)
- `model_library` must be present at the path in `config/config.ini -> [paths] model_library_root`
- Required packages: `scikit-learn`, `numpy`, `pyyaml` (already available in the venv)

---

## Input format

Prepare a JSON file with these fields:

```json
{
  "model_type": "classification",
  "model_name": "DecisionTreeClassifier",
  "dataset_path": "/abs/path/to/dataset.npz",
  "train_metrics": {},
  "test_metrics":  {}
}
```

| Field | Required | Description |
|---|---|---|
| `model_type` | yes | `"classification"` or `"regression"` |
| `model_name` | yes | One of the 60 names registered in MLKit (see `model_library/ml_kit.py::MODEL_REGISTRY`) |
| `dataset_path` | yes | Absolute path to a `.npz` file containing arrays `X_train`, `y_train`, `X_test`, `y_test` |
| `train_metrics` | no | Precomputed train metrics dict (keys match `MetricsResult` fields). If omitted or `{}`, the tool trains via MLKit and computes them. |
| `test_metrics` | no | Same as above for test split. Both must be non-empty to use precomputed path. |

### Preparing the .npz dataset

```python
import numpy as np
np.savez(
    "/path/to/dataset.npz",
    X_train=X_train,   # shape (n_train, n_features), dtype float32
    y_train=y_train,   # shape (n_train,)
    X_test=X_test,     # shape (n_test, n_features), dtype float32
    y_test=y_test,     # shape (n_test,)
)
```

---

## Running the tool

```bash
cd /home/sujithma/mitra/epic_4/overfitting_analysis_tool

/home/sujithma/venv/bin/python overfitting_analysis.py \
  -i /path/to/input.json \
  -o /path/to/output_dir \
  -v
```

| Arg | Required | Description |
|---|---|---|
| `-i / --input_json` | yes | Path to the input JSON |
| `-o / --output_dir` | yes | Directory where `overfitting_analysis.json` is written (created if absent) |
| `-v / --verbose` | no | Enable DEBUG-level logging |

---

## Output

`<output_dir>/overfitting_analysis.json`:

```json
{
  "model_name": "DecisionTreeClassifier",
  "model_type": "classification",
  "is_overfitted": true,
  "gap_threshold": 0.1,
  "primary_metric": "accuracy",
  "gaps": {
    "accuracy": 0.15,
    "f1_macro": 0.13
  },
  "rel_rmse_gap": null,
  "train_metrics": { "accuracy": 1.0, "f1_macro": 1.0 },
  "test_metrics":  { "accuracy": 0.85, "f1_macro": 0.87 },
  "k_fold_cross_validation_results": {
    "k": 5,
    "scoring": "accuracy",
    "per_fold_scores": [0.81, 0.79, 0.83, 0.80, 0.82],
    "mean": 0.81,
    "std": 0.014,
    "train_vs_cv_gap": 0.19
  },
  "cv_skipped_reason": null
}
```

Key fields:

| Field | Description |
|---|---|
| `is_overfitted` | `true` if `primary_metric` gap exceeds `gap_threshold` (config.yaml) |
| `gaps` | Direction-aware gap per metric: positive always means train beat test |
| `rel_rmse_gap` | `(test_rmse - train_rmse) / train_rmse` — regression only, else `null` |
| `train_vs_cv_gap` | Gap between holdout train score and CV mean (used as fallback verdict signal) |
| `cv_skipped_reason` | Non-null string if K-fold was skipped (e.g. too few samples) |

---

## Calling from another Python script (pipeline integration)

```python
import sys
sys.path.insert(0, "/home/sujithma/mitra/epic_4/overfitting_analysis_tool")

from overfitting_analysis import OverfittingAnalyzer

analyzer = OverfittingAnalyzer(
    input_json_path="/path/to/input.json",
    output_dir="/path/to/output_dir",
    verbose=False,
)
result = analyzer.run()

if result["is_overfitted"]:
    print(f"=> Model overfit. Primary gap: {result['gaps'][result['primary_metric']]:.4f}")
```

---

## Controllables (config/config.yaml)

| Key | Default | Description |
|---|---|---|
| `k_folds` | 5 | Number of CV folds |
| `shuffle` | true | Shuffle data before splitting |
| `random_state` | 42 | Seed for reproducible folds |
| `cv_data_source` | `train_test_concat` | Data used for CV: `train_test_concat`, `train_only`, `test_only` |
| `gap_threshold` | 0.1 | Threshold above which `is_overfitted` is set to true |
| `classification.primary_metric` | `accuracy` | Metric driving the verdict for classifiers |
| `regression.primary_metric` | `r2` | Metric driving the verdict for regressors |
| `metric_direction_map` | see config | `higher_is_better` / `lower_is_better` per metric |

---

## Running the test suite

```bash
cd /home/sujithma/mitra/epic_4/overfitting_analysis_tool
/home/sujithma/venv/bin/python tests/test_overfitting.py -v
```

Covers: overfit detection (classification), well-fit detection (classification + regression),
and K-fold reproducibility. All 4 cases must pass before integrating into a pipeline.

---

## Supported model names

Any of the 60 names in `model_library/ml_kit.py::MODEL_REGISTRY`. Common examples:

- Classification: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier`
- Regression: `LinearRegression`, `Ridge`, `RandomForestRegressor`, `XGBRegressor`

`model_type` must match the model family — passing a classifier name with `"regression"` raises
a clear error.
