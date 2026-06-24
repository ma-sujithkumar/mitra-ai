# Plan: Overfitting Analysis Tool (epic-4)

## Context
`epic_4/overfitting_analysis_tool/SPEC.md` (just revised) asks for a tool that takes a
`model_name` + `dataset_path`, computes train/test metrics, runs K-fold cross validation
by retraining the model, and emits a verdict on whether the model overfit. The whole point
is that this is an AutoML pipeline step that **reuses** the already-built `model_library`
(MLKit) rather than reimplementing training or metrics. Exploration confirmed every reuse
point exists and is importable. This plan implements the tool against those components.

## Reuse (do NOT reimplement)
- `model_library/ml_kit.py::MLKit` => the training class. `MLKit(model_name, DataBundle).train()`,
  `.test()` (returns predictions), `.model.predict(X)` for train-set predictions.
- `model_library/core/data_bundle.py::CommonData(X_train,y_train,X_test,y_test)`, `DataBundle(common, hyperparameters)`.
- `model_library/metrics/evaluators.py::compute_metrics(y_true,y_pred,task_type,model_name)` -> `MetricsResult`
  (classification: accuracy/f1_macro/f1_weighted/precision_macro/recall_macro; regression: mse/rmse/mae/r2).
- `model_library/core/validators.py::validate_model_name` (raises with suggestions for bad names).
- Import bootstrap mirrors `model_library/tests/test_*.py`: `sys.path.insert(0, model_library_root)`.

## Directory layout
```
epic_4/overfitting_analysis_tool/
  config/config.ini          # [python] PYTHON ; [paths] model_library_root, config_yaml
  config/config.yaml         # all controllables
  overfitting_analysis.py    # OOP implementation
  tests/test_overfitting.py  # runner script (argparse + main + asserts), no pytest
  docs/PLAN.md               # this file (git-tracked)
```

## config/config.ini
```
[python]
PYTHON = /home/sujithma/venv/bin/python

[paths]
model_library_root = /home/sujithma/mitra/model_library
config_yaml = config/config.yaml
```
`model_library_root` lives here so no paths are hardcoded in code.

## config/config.yaml (controllables)
```yaml
overfitting:
  k_folds: 5
  shuffle: true
  random_state: 42
  cv_data_source: train_test_concat        # train_test_concat | train_only | test_only
  gap_threshold: 0.1
  epsilon: 1.0e-9
  classification:
    stratified: true
    scoring_metric: accuracy
    primary_metric: accuracy
  regression:
    stratified: false
    scoring_metric: r2
    primary_metric: r2
  metric_direction_map:
    accuracy: higher_is_better
    f1_macro: higher_is_better
    f1_weighted: higher_is_better
    precision_macro: higher_is_better
    recall_macro: higher_is_better
    r2: higher_is_better
    mse: lower_is_better
    rmse: lower_is_better
    mae: lower_is_better
```

## overfitting_analysis.py
OOP, fully typed. Imports at top; a config-driven `sys.path` bootstrap before model_library
imports (same pattern as existing tests).

Module-level dispatch map (avoids if-else for metric direction):
```python
DIRECTION_GAP_FUNCS = {
    "higher_is_better": lambda train_score, test_score: train_score - test_score,
    "lower_is_better":  lambda train_score, test_score: test_score - train_score,
}
```

Class `OverfittingAnalyzer`:
- `__init__`: reads config.ini + config.yaml, parses input JSON, validates fields, calls
  `validate_model_name`, checks model_name vs model_type consistency.
- `load_dataset`: `np.load(.npz)`, requires X_train/y_train/X_test/y_test arrays.
- `compute_holdout_metrics`: uses precomputed metrics from input JSON when present; otherwise
  one MLKit train + `test()` for test preds + `kit.model.predict(X_train)` for train preds.
- `run_kfold`: assembles CV data per `cv_data_source`; StratifiedKFold/KFold per task type;
  retrains a fresh MLKit per fold; returns KFoldResult with per_fold_scores/mean/std/train_vs_cv_gap.
- `compute_gaps`: iterates non-None metric fields; direction-aware gap via DIRECTION_GAP_FUNCS;
  regression also computes `rel_rmse_gap`.
- `decide_verdict`: `primary_gap > gap_threshold`; falls back to `train_vs_cv_gap` if holdout missing.
- `write_output`: `mkdir -p output_dir`; writes `overfitting_analysis.json`.
- `run`: orchestrates all above.

CLI: `-i/--input_json` (required), `-o/--output_dir` (required), `-v/--verbose`.

## tests/test_overfitting.py
Runner script (no pytest). Four test cases:
1. Classification overfit: `DecisionTreeClassifier` on small data => `is_overfitted: true`.
2. Classification well-fit: `LogisticRegression` on well-separated data => `is_overfitted: false`.
3. Regression well-fit: `LinearRegression` on clean regression data — checks schema + `rel_rmse_gap`.
4. K-fold reproducibility: two runs with same `random_state` produce identical `per_fold_scores`.

## Edge cases handled
- Bad `model_name` => `validate_model_name` raises with suggestions.
- model_name/model_type mismatch => explicit ValueError.
- `.npz` missing required arrays => explicit error listing what was found.
- `k` too large for samples/min class count => explicit error before CV.

## Verification
```
cd epic_4/overfitting_analysis_tool
/home/sujithma/venv/bin/python tests/test_overfitting.py -v
```
All 4 test cases must pass.

## Notes / assumptions
- Dataset assumed in-memory `.npz` (v1 scope).
- `train_vs_cv_gap` uses the configured `scoring_metric` for comparison against `gap_threshold`
  as fallback when holdout metrics are absent.
