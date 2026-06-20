# Integration Plan: SHAP & Hyperparameter Tuning Outputs into Judge Agent

**Date**: 2026-06-20  
**Status**: COMPLETED  
**Test Results**: ✅ 2/2 smoke tests pass, ✅ 14/14 existing judge tests pass

---

## Overview

This plan documents the integration of SHAP feature importance and hyperparameter tuning (HPT) sensitivity outputs into the Judge Agent's decision pipeline. The Judge Agent already had schema fields for these data (`shap_summary`, `hyperparam_sensitivity`), but there was no code to populate them from real pipeline outputs. This integration wires the missing plumbing and enriches the LLM prompt so the judge's commentary meaningfully uses this information.

---

## Problem Statement

**Before**: The Judge Agent had empty `shap_summary` and `hyperparam_sensitivity` fields in candidate models, forcing the LLM to make decisions without access to feature importance and parameter sensitivity insights.

**After**: Each candidate model automatically gets:
- **SHAP summary**: Top N features by absolute SHAP value + concentration metric (% of total importance captured by top features)
- **Hyperparameter sensitivity**: Which parameter has the highest impact on model performance, with best hyperparameters embedded for context

---

## Implementation Summary

### 1. HPT Agent: Compute Hyperparameter Sensitivity

**File**: `epic_4/hyperparameter_tuning_agent/optuna_wrapper.py`

**Added Method**: `OptunaWrapper.compute_param_sensitivity()`
- Analyzes all completed Optuna trials to measure per-parameter sensitivity
- Computes `score_range` (max_score - min_score) for each parameter across trials
- Returns top-level `sensitivity_score` (maximum score_range) and `most_sensitive_param`
- Returns `None` if fewer than 2 trials (insufficient data for sensitivity analysis)
- Uses numpy for vectorized operations

**Integration**: In `agent.py`, calls this method after Optuna completes and stores result in `result_entry['hyperparam_sensitivity']`, which gets written to `hpt_results.json`.

### 2. Judge Adapter: Read SHAP CSVs and HPT Results

**File**: `epic_4/judge_agent/adapter.py`

**Added Methods**:

**`UpstreamAdapter.build_shap_summary_from_csv(csv_path, top_n=5)`**
- Reads `global_feature_importance.csv` (from SHAP module)
- CSV format: `feature_name, mean_absolute_shap_value` (sorted descending)
- Returns dict with:
  - `top_features`: List of top N feature names
  - `mean_abs_shap`: Dict mapping feature name → rounded SHAP value
  - `feature_concentration`: Fraction of total SHAP captured by top N features
- Returns `None` if file missing or empty (logs warning)

**`UpstreamAdapter.adapt_from_hpt_results(hpt_json_path, task_type, shap_dir=None, ...)`**
- New entry point: reads `hpt_results.json` directly (bypassing manual JSON construction)
- Maps each HPT entry:
  - `name` → `model_name`
  - `val_metrics` → `metrics`
  - `overfitting` dict (already matches `OverfittingInfo` schema)
  - `complexity` dict → `ComplexityDescriptor`
  - `hyperparam_sensitivity` with `best_params` embedded for prompt context
- Looks up SHAP CSVs via two-level fallback:
  1. Per-model: `<shap_dir>/<model_name>/csv/global_feature_importance.csv`
  2. Shared: `<shap_dir>/csv/global_feature_importance.csv`
- Returns fully populated `JudgeInput` with all enrichments

### 3. Judge Prompt: Structured Data Presentation

**File**: `epic_4/judge_agent/prompts/judge_prompt.jinja2`

**Enhancements**:

**Dataset Context Block**:
```
HPT run summary: N models tuned in X.Xs using <metric> as primary metric.
```

**Per-Model SHAP Block** (replaces raw JSON dump):
```
SHAP feature importance (top 4 features by mean |SHAP|):
  feature_a: 0.51
  feature_b: 0.29
  ...
Feature concentration: 0.80
  (=> low concentration, model may rely on many features diffusely)  [only if < 0.5]
```

**Per-Model Hyperparameter Sensitivity Block**:
```
Hyperparameter sensitivity:
  Most sensitive param: learning_rate
  Sensitivity score: 0.08
  (=> high, small param changes shift performance significantly)  [only if > 0.05]
  Best hyperparameters found: {...}
```

These blocks give the LLM concrete, human-readable facts to anchor its commentary and flag concerns (e.g., "high SHAP concentration on just 2 features suggests model may be overly reliant on those features").

### 4. Judge CLI: Session-Based Invocation

**File**: `epic_4/judge_agent/run_judge.py`

**New Flags**:
- `--hpt-json <path>`: Path to `hpt_results.json` (mutually exclusive with `-i`)
- `--shap-dir <path>`: Root of SHAP outputs for this session (optional, used with `--hpt-json`)
- `--task-type <classification|regression>`: Required when using `--hpt-json`

**Behavior**:
```bash
# Old way: pre-built JSON
python run_judge.py -i judge_input.json -o /tmp/out/ --no-llm

# New way: direct from HPT + SHAP
python run_judge.py --hpt-json .mitra/session_123/hpt_results.json \
  --shap-dir .mitra/session_123/shap_outputs/ \
  --task-type classification \
  -o /tmp/out/ --no-llm
```

### 5. Config

**File**: `epic_4/judge_agent/config/config.yaml`

Added:
```yaml
shap_top_n_features: 5
```

This controls how many top features are included in the `shap_summary` dict passed to the LLM.

---

## Testing & Verification

### Smoke Tests ✅

**File**: `claude_scripts/smoke_test_judge_integration.py`

Runs two end-to-end tests:

1. **Classification** (Iris, 150 samples, 4 features, 3 classes):
   - Trains RandomForest, GradientBoosting, LogisticRegression
   - Generates per-model SHAP CSVs using sklearn `feature_importances_`
   - Synthesizes `hpt_results.json` with realistic HPT entries (overfitting gaps, sensitivity scores, best_params)
   - Calls `adapter.adapt_from_hpt_results()` → `JudgeAgent.judge(use_llm=False)`
   - Assertions:
     - `selected_model` is not `None`
     - All ranked models have `rank > 0`
     - SHAP summaries are populated in candidates
     - Hyperparameter sensitivity is populated in candidates

2. **Regression** (Diabetes, 442 samples, 10 features):
   - Same workflow with RandomForestRegressor, GradientBoostingRegressor, Ridge
   - Uses `r2` as primary metric
   - Verifies feature concentration calculation (top 5 features)

**Results**: ✅ **2/2 PASS**
```
=> Smoke test results: 2/2 passed
```

### Existing Tests ✅

**File**: `epic_4/judge_agent/tests/`

All 14 existing judge agent tests still pass:
- 3 adapter tests (overfitting mapping, judge input construction)
- 11 rule engine + integration tests (gates, scoring, tie-break, end-to-end)

```
====== 14 passed, 1 skipped, 4 warnings ======
```

### Manual CLI Test ✅

```bash
python epic_4/judge_agent/run_judge.py \
  --hpt-json /tmp/test_hpt.json \
  --task-type classification \
  -o /tmp/judge_cli_out/ \
  --no-llm -v
```

Result: ✅ **2 candidates ranked, LogisticRegression selected** with correct scores (0.9093 vs 0.9067).

---

## Critical Files Modified

| File | Changes |
|---|---|
| `epic_4/hyperparameter_tuning_agent/optuna_wrapper.py` | Added `compute_param_sensitivity()` method |
| `epic_4/hyperparameter_tuning_agent/agent.py` | Call sensitivity computation, store in result_entry |
| `epic_4/judge_agent/adapter.py` | Added `build_shap_summary_from_csv()`, `adapt_from_hpt_results()` |
| `epic_4/judge_agent/prompts/judge_prompt.jinja2` | Enriched SHAP and HPT blocks, added HPT context |
| `epic_4/judge_agent/run_judge.py` | Added `--hpt-json`, `--shap-dir`, `--task-type` flags |
| `epic_4/judge_agent/config/config.yaml` | Added `shap_top_n_features: 5` |
| `claude_scripts/smoke_test_judge_integration.py` | New smoke test script (classification + regression) |

---

## Code Reuse & No Duplication

- **`adapter.adapt_candidate()`** — Reused inside `adapt_from_hpt_results()` for building each candidate (no new construction logic)
- **`JudgeAgent.judge()`** — Unchanged; adapter feeds into existing judge without modification
- **Overfitting adaptation** — HPT's `overfitting` dict already matches `OverfittingInfo` fields, so no double-adaptation needed; used directly

---

## Workflow Example: End-to-End Integration

```
1. User runs HPT Agent:
   HyperparameterTuningAgent.run() → writes to .mitra/<session>/hpt_results.json
   
2. User runs SHAP Module (parallel):
   SHAP pipeline → writes per-model CSVs to .mitra/<session>/shap_outputs/<model>/csv/
   
3. User invokes Judge Agent:
   run_judge.py --hpt-json ... --shap-dir ... --task-type classification
   
4. Adapter wires everything:
   - Reads hpt_results.json
   - For each model, looks up SHAP CSV and builds shap_summary
   - Embeds best_params into hyperparam_sensitivity
   
5. Judge Agent scores & ranks:
   - Rule engine: hard gates + weighted scoring (unchanged)
   - LLM enrichment (if enabled): reads prompt with all SHAP/HPT context
   
6. Output judge_decision.json with auditable trace
```

---

## Backward Compatibility

✅ **All existing interfaces unchanged**:
- Original `-i <json>` input still works
- `JudgeInput` schema unchanged (new fields already existed, now just populated)
- Rule engine logic unchanged (SHAP/HPT are context-only for LLM, not scored)
- All 14 existing tests pass without modification

✅ **New mode is opt-in**: `--hpt-json` is an alternative path, legacy JSON input still fully supported.

---

## Future Enhancements

1. **Scoring using SHAP concentration**: If a model has low feature concentration (relying on many features diffusely), apply a small penalty or flag in scoring.
2. **Hyperparameter sensitivity thresholds**: Reject models with high sensitivity (brittle performance) if threshold breached.
3. **Per-class SHAP summaries**: For multiclass problems, compute and present per-class SHAP summaries.
4. **SHAP interaction effects**: If available, embed top pairwise feature interactions in prompt.

---

## How to Use

### HPT Agent Output with Judge

```bash
# After HPT completes, you have:
# .mitra/session_123/hpt_results.json
# .mitra/session_123/shap_outputs/<model>/csv/global_feature_importance.csv

# Run Judge:
python epic_4/judge_agent/run_judge.py \
  --hpt-json .mitra/session_123/hpt_results.json \
  --shap-dir .mitra/session_123/shap_outputs/ \
  --task-type classification \
  -o .mitra/session_123/judge_output/ \
  -v

# Output:
# .mitra/session_123/judge_output/judge_decision.json
```

### Run Smoke Tests

```bash
python claude_scripts/smoke_test_judge_integration.py -v
```

Expected output: `=> Smoke test results: 2/2 passed`

---

## Summary

✅ **Complete end-to-end integration**  
✅ **SHAP & HPT data now flow seamlessly into Judge Agent**  
✅ **LLM has concrete, structured context for richer commentary**  
✅ **All existing tests pass + 2 new smoke tests pass**  
✅ **Backward compatible; new mode is opt-in**  
✅ **No code duplication; adapter pattern preserved**

The Judge Agent can now consume real SHAP and HPT outputs from the pipeline and provide richer, more informed model selection decisions.
