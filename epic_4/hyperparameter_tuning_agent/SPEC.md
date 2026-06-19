# SPEC: Hyperparameter Tuning Agent

## 1. GOAL

Automatically find the best hyperparameters for each machine learning model using Optuna, maximizing validation performance while preventing overfitting, and hand off tuned results to the Judge Agent.

## 2. Agent Objective

The Hyperparameter Tuning Agent must:
1. Accept the `model_config.json` produced by Epic-3's Model Selection Agent
2. For each model entry, perform an intelligent Optuna-driven hyperparameter search using the `hp_space` ranges defined in `model_config.json` as the search boundaries
3. Train each trial using `model_library/ml_kit.py::MLKit` - training code available in `epic_3\training\trainer.py`
4. Evaluate each trial using `model_library/metrics/evaluators.py::compute_metrics`
5. Select the best hyperparameter combination per model based on the primary validation metric
6. Persist tuned results as `hpt_results.json` under `.mitra/<session_id>/`
7. Hand off `hpt_results.json` to the Judge Agent (via `UpstreamAdapter`) for final ranking

## 3. CONSTRAINTS

### 3.1 Hard Constraints

| Constraint | Value / Rule |
|:---|:---|
| Max Optuna trials per model | `MAX_HPT_TRIALS` from `config.ini` [pipeline] (default: 5) |
| Max concurrent HPT workers | `MAX_CONCURRENT_RUNS` from `config.ini` [training_api] |
| Overfitting gate | A trial is penalized (not promoted) if `train_score - val_score > OVERFITTING_GAP_THRESHOLD` (configurable in HPT config section) |
| Test set is hidden | `test.csv` / `X_test` / `y_test` are NEVER used during tuning; only train/val splits are used |
| Reproducibility | Every Optuna study is seeded with `random_state: 42` from MLKit config |
| Search space source | `hp_space` from each entry in `model_config.json`; no hardcoded ranges in agent code |
| Python binary | `PYTHON` from `config.ini` [python] |
| Session output root | `.mitra/<session_id>/` as defined by `WORKSPACE_ROOT` in `config.ini` [paths] |
| No custom train/test logic | Must reuse `MLKit.train()` / `MLKit.test()` and `compute_metrics` exclusively |
| Logging | All runs produce structured logs; `-v` flag enables verbose mode |

### 3.2 Overfitting Check Integration

After each trial completes, the agent invokes `epic_4/overfitting_analysis_tool/overfitting_analysis.py` logic (or its equivalent Python API) to compute the overfitting gap per trial. A trial with `is_overfitted: true` is recorded but not promoted as the best trial. The final selected hyperparameters for each model MUST satisfy the overfitting gate.

### 3.3 MLKit Compatibility

- All 60 model names registered in `model_library/config/config.yaml` are valid inputs.
- `DataBundle` from `model_library/core/data_bundle.py` is the only data container used.
- Hyperparameters passed to MLKit override the `config.yaml` defaults for that trial.

## 4. INPUT DATA AND DEPENDENCIES

### 4.1 Inputs from Previous Epics

The agent reads all artifacts from `.mitra/<session_id>/`:

| **Input File** | **Source** | **Format** | **Description** |
|:---|:---|:---|:---|
| `metadata.json` | Epic-1 (Metadata Gen Agent) | JSON | Dataset characteristics: `problem_type`, `col_types`, `target_col`, `row_count`, `col_count`, `class_balance` |
| `data/train.csv` | Epic-2 (Pipeline) | CSV | Training split (80% of cleaned data, default from `TRAIN_TEST_SPLIT` in config.ini) |
| `data/test.csv` | Epic-2 (Pipeline) | CSV | Test split (20% of cleaned data) — HIDDEN during HPT |
| `model_config.json` | Epic-3 (Model Selection Agent) | JSON | Array of model entries; each entry provides `name`, `family`, `hp_space`, `priority` |

### 4.2 model_config.json Schema (consumed fields)

Each entry in `model_config.json` that the HPT agent uses:

```json
{
  "name": "xgb_v1",
  "family": "xgboost",
  "rationale": "...",
  "priority": 1,
  "hp_space": {
    "n_estimators": {"type": "int",   "low": 50,  "high": 500, "step": 50},
    "max_depth":    {"type": "int",   "low": 3,   "high": 10},
    "learning_rate":{"type": "float", "low": 0.01,"high": 0.3, "log": true},
    "subsample":    {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    "reg_lambda":   {"type": "float", "low": 1e-3,"high": 10.0,"log": true}
  }
}
```

`hp_space` parameter types supported (matches Epic-3 Model Selection schema):
- `{"type": "int",   "low": N, "high": M, "step": S}` — maps to `trial.suggest_int()`
- `{"type": "float", "low": N, "high": M, "log": true/false}` — maps to `trial.suggest_float()`
- `{"type": "categorical", "choices": [...]}` — maps to `trial.suggest_categorical()`

At most 6 hyperparameters per model (enforced by the `model_config.json` schema from Epic-3).

### 4.3 Hyperparameter Default Fallback

If a parameter present in `model_library/config/config.yaml` for the model is NOT included in `hp_space`, the MLKit config.yaml default is used verbatim. The agent must NOT widen or invent search ranges.

## 5. OUTPUT: hpt_results.json

Written atomically (`hpt_results.json.tmp` then `os.replace`) to `.mitra/<session_id>/`.

### 5.1 Schema

```json
[
  {
    "name": "xgb_v1",
    "model_class": "XGBClassifier",
    "family": "xgboost",
    "priority": 1,
    "best_params": {
      "n_estimators": 200,
      "max_depth": 6,
      "learning_rate": 0.05
    },
    "val_metrics": {
      "accuracy": 0.91,
      "f1_macro": 0.90,
      "f1_weighted": 0.91,
      "precision_macro": 0.91,
      "recall_macro": 0.90
    },
    "train_metrics": {
      "accuracy": 0.95,
      "f1_macro": 0.94
    },
    "overfitting": {
      "is_overfitted": false,
      "gap": 0.04,
      "train_vs_cv_gap": null
    },
    "complexity": {
      "n_params": 200,
      "depth": 6,
      "family_rank": 9
    },
    "n_trials": 5,
    "best_trial_number": 3,
    "optuna_study_name": "hpt_xgb_v1_<session_id>"
  }
]
```

One entry per model from `model_config.json`. All entries written regardless of overfitting status; the Judge Agent's rule engine applies the final gate.

### 5.2 Metric Keys

Metric keys in `val_metrics` and `train_metrics` match `model_library/metrics/evaluators.py::MetricsResult` exactly:

| Task type | Keys |
|:---|:---|
| classification | `accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro` |
| regression | `mse`, `rmse`, `mae`, `r2` |

The `problem_type` field from `metadata.json` selects which metric set is used.

### 5.3 Primary Optimization Metric

| `problem_type` | Primary metric (Optuna direction) |
|:---|:---|
| `classification` | `accuracy` (maximize) |
| `regression` | `r2` (maximize) |

The primary metric is configurable via `HPT_PRIMARY_METRIC_CLASSIFICATION` and `HPT_PRIMARY_METRIC_REGRESSION` in `config.ini` [hpt].

## 6. PROCESSING PIPELINE

```
model_config.json
       |
       v
For each model (ordered by priority):
  1. Build DataBundle from train.csv + val split (StratifiedKFold or KFold per problem_type)
  2. Create Optuna study (sampler=TPESampler, seed=42)
  3. For each trial (up to MAX_HPT_TRIALS):
       a. Sample hp dict from hp_space
       b. MLKit(model_name, data=DataBundle, hyperparameters=hp_dict).train()
       c. metrics = compute_metrics(y_val, y_pred, problem_type)
       d. Compute overfitting gap (train_score - val_score)
       e. Report metric to Optuna (penalize if overfitted)
  4. Select best non-overfitted trial; fall back to least-overfitted if all overfit
  5. Append result entry to hpt_results list
       |
       v
Write hpt_results.json (atomic)
       |
       v
Hand off path to Judge Agent orchestrator
```

## 7. CONFIG.INI ADDITIONS (new [hpt] section)

The following keys must be added to the project-wide `config.ini`:

```ini
[hpt]
HPT_PRIMARY_METRIC_CLASSIFICATION=accuracy
HPT_PRIMARY_METRIC_REGRESSION=r2
OVERFITTING_GAP_THRESHOLD=0.10
VAL_SPLIT_RATIO=0.2
HPT_N_JOBS=1
OPTUNA_SAMPLER=TPE
OPTUNA_SEED=42
HPT_OUTPUT_FILENAME=hpt_results.json
```

All HPT agent code reads from this section via the project-wide `config.ini`.

## 8. DOWNSTREAM: JUDGE AGENT CONTRACT

The Judge Agent's `UpstreamAdapter` reads `hpt_results.json` and maps each entry to the `CandidateModel` schema defined in `epic_4/judge_agent/input_format_requirement.md`:

| `hpt_results.json` field | Maps to Judge `CandidateModel` field |
|:---|:---|
| `model_class` | `model_name` |
| `problem_type` (from metadata) | `task_type` |
| `val_metrics` | `metrics` |
| `overfitting` | `overfitting` |
| `complexity` | `complexity` |
| _(not yet populated by HPT)_ | `shap_summary` — filled by SHAP tool separately |
| _(not yet populated by HPT)_ | `hyperparam_sensitivity` — context only, optional |

## 9. PACKAGES TO BE USED

| Package | Purpose |
|:---|:---|
| `optuna` | Hyperparameter search (TPESampler, study management) |
| `scikit-learn` | StratifiedKFold / KFold for validation splits |
| `numpy` | Array operations |
| `pyyaml` | Reading `model_library/config/config.yaml` for defaults |
| `model_library/ml_kit.py` | MLKit — training and inference |
| `model_library/metrics/evaluators.py` | `compute_metrics`, `MetricsResult` |
| `model_library/core/data_bundle.py` | `DataBundle`, `CommonData` |
| `configparser` | Reading `config.ini` |

## 10. IMPLEMENTATION NOTES

1. The agent is a Python class (`HyperparameterTuningAgent`) following project OOP conventions.
2. All imports at the top; no conditional imports.
3. No hardcoded paths, values, or hyperparameter ranges in code — config.ini and model_config.json are the only sources.
4. Descriptive variable names throughout (no single-letter names).
5. Verbose logging via `-v` switch; `logging` module only (no `print`).
6. Output directory created with `mkdir -p` before writing any file.
7. No unicode characters in print statements or logs — ASCII only.
8. Any standalone test/debug scripts go into `claude_scripts/` (gitignored).

