# SPEC: Overfitting Analysis Tool

## 1. GOAL
Provide a Python tool that, given a trained-able model and its dataset, computes
regression/classification metrics on train vs. test, runs K-fold cross validation,
and decides whether the model has overfitted. Designed to be called as a step in
an AutoML pipeline.

## 2. APPLICATION CONTEXT
1. Runs as a step inside an AutoML pipeline.
2. Reuses existing MITRA components (do NOT reimplement metrics or training):
   - `model_library/ml_kit.py::MLKit` => the "training class". It exposes
     `train()` and `test()` over a `DataBundle`. This replaces the placeholder.
   - `model_library/metrics/evaluators.py::compute_metrics` + `MetricsResult`
     => standard metric computation (accuracy/f1/precision/recall for
     classification; mse/rmse/mae/r2 for regression).
   - `model_library/core/data_bundle.py::CommonData`, `DataBundle`
     => data container expected by MLKit.

## 3. PACKAGES TO BE USED
- scikit-learn (KFold / StratifiedKFold, cross-validation splitting)
- numpy
- pyyaml (controllables)
- MITRA model_library (MLKit, compute_metrics, DataBundle)

## 4. REQUIREMENTS
1. Compute a direction-aware overfitting gap per metric (see Section 7):
   - Higher-is-better metrics (accuracy, f1, precision, recall, r2):
     `gap = train_score - test_score` (positive => train better than test).
   - Lower-is-better metrics (mse, rmse, mae):
     `gap = test_error - train_error` (positive => test worse than train).
   - Also report a scale-independent relative RMSE gap for regression:
     `rel_rmse_gap = (test_rmse - train_rmse) / max(train_rmse, eps)`.
2. Perform K-fold cross validation by re-training the model on each fold via
   MLKit. Report per-fold scores, mean, and std for the configured scoring metric.
3. Emit an overfitting verdict `is_overfitted: bool`, decided by comparing the
   primary gap (and/or the train-vs-CV gap) against a configurable threshold.
4. Interface with the training class through MLKit (no placeholder). If the model
   cannot be retrained for a given input, K-fold is skipped and the tool falls
   back to the holdout gap only, with a `cv_skipped_reason` field.

## 5. INPUT FORMAT
A single JSON file (path via `-i`):
```json
{
  "model_type": "classification|regression",
  "model_name": "XGBClassifier",
  "dataset_path": "/abs/path/to/dataset.npz",
  "train_metrics": {},
  "test_metrics": {}
}
```
- `model_name`: one of the 60 names registered in MLKit (REQUIRED).
- `dataset_path`: REQUIRED. An `.npz` containing arrays `X_train`, `y_train`,
  `X_test`, `y_test`. Loaded into `CommonData`/`DataBundle` for MLKit. K-fold runs
  on the concatenation of train+test by default (configurable in config.yaml).
- `train_metrics`, `test_metrics`: OPTIONAL precomputed holdout metrics whose keys
  match `MetricsResult` fields. When absent, the tool computes them itself by
  training via MLKit and calling `compute_metrics`. When present, they are used
  as-is for the holdout gap (no retraining needed for the gap).

## 6. OUTPUT FORMAT
A JSON written to `<output_dir>/overfitting_analysis.json`:
```json
{
  "model_name": "XGBClassifier",
  "model_type": "classification",
  "is_overfitted": true,
  "gap_threshold": 0.1,
  "primary_metric": "accuracy",
  "gaps": {
    "accuracy": 0.18,
    "f1_macro": 0.15
  },
  "rel_rmse_gap": null,
  "train_metrics": {},
  "test_metrics": {},
  "k_fold_cross_validation_results": {
    "k": 5,
    "scoring": "accuracy",
    "per_fold_scores": [0.81, 0.79, 0.83, 0.80, 0.82],
    "mean": 0.81,
    "std": 0.014,
    "train_vs_cv_gap": 0.16
  },
  "cv_skipped_reason": null
}
```
- `gaps`: direction-aware gap for every metric available for the task type.
- `rel_rmse_gap`: populated for regression only, else `null`.
- `train_vs_cv_gap`: train score minus CV mean (direction-aware), the primary
  overfitting signal when holdout metrics are unavailable.

## 7. CONTROLLABLES (config/config.yaml)
- `k_folds`: number of folds (e.g. 5).
- `stratified`: true for classification, false for regression (auto by model_type
  unless overridden).
- `scoring_metric`: metric scored per fold (e.g. accuracy / r2).
- `primary_metric`: metric used for the overfitting verdict.
- `gap_threshold`: float; `is_overfitted = primary_gap > gap_threshold`.
- `cv_data_source`: `train_test_concat` | `train_only` | `test_only`.
- `random_state`: seed for reproducible folds.
- `metric_direction_map`: hash map of metric => `higher_is_better` |
  `lower_is_better`. Avoids an if-else ladder when computing direction-aware gaps.
- `epsilon`: small value for the relative RMSE denominator.

Python binary and paths live in `config/config.ini` (`[python] PYTHON=...`,
`[paths] ...`), matching the model_library convention. One config.ini per project.

## 8. CLI ARGS
- `-i, --input_json <path>`: REQUIRED. Path to the input JSON. Error if missing.
- `-o, --output_dir <path>`: REQUIRED. Created with `mkdir -p`. Error if missing.
- `-v, --verbose`: enable debug logging.

## 9. DEVELOPMENT OUTPUTS
1. `config/config.ini` - python binary + paths.
2. `config/config.yaml` - all controllables (Section 7).
3. `overfitting_analysis.py` - OOP implementation; imports MLKit, compute_metrics,
   DataBundle at top of file; direction-aware gaps via the config map.
4. `tests/` - test suite that builds a small synthetic dataset (.npz), runs the
   tool end-to-end through MLKit for both a classifier and a regressor, and asserts
   the output JSON schema and that a deliberately overfit model is flagged.

## 10. ACCEPTANCE CRITERIA
1. Runs end-to-end on the test suite for both classification and regression and
   produces a schema-valid `overfitting_analysis.json`.
2. K-fold results contain per-fold scores, mean, and std and are reproducible
   under a fixed `random_state`.
3. A deliberately overfit model yields `is_overfitted: true`; a well-fit model
   yields `is_overfitted: false`.
4. No metric or training logic is reimplemented; MLKit and compute_metrics are
   reused.

## 11. OPEN ITEMS / ASSUMPTIONS
- Dataset is assumed to fit in memory as a single `.npz`. Streaming/large datasets
  are out of scope for v1.
- `train_vs_cv_gap` vs `primary gap` precedence for the verdict is configurable;
  default uses the holdout `primary gap` when holdout metrics exist, else
  `train_vs_cv_gap`.
