# Phase 5 Implementation Report
**Date:** 2026-06-17
**Branch:** epic4-shap
**Status:** Complete -- 187 tests passing (126 existing + 61 new)

---

## Summary

Phase 5 implemented the SHAP Explainer Layer: the two-module pair that selects and
constructs a SHAP explainer for the confirmed model family (ExplainerFactory) and
runs the SHAP computation to produce structured output DataFrames ready for Phase 6
CSV exporters and visualisers (SHAPService).

---

## Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| `src/shap_explainability/errors.py` | Modified | Added `SHAPExecutionError` |
| `config/model_type_detection.json` | Modified | Added 3 new sections for Phase 5 |
| `config/config.ini` | Modified | Added `[model]` section with `LINEAR_EXPLAINER_BACKGROUND_SAMPLES` |
| `src/shap_explainability/utils/logger.py` | Modified | Added `level` param to `log_shap_generation` for consistency |
| `src/shap_explainability/explainers/__init__.py` | Created | Package marker |
| `src/shap_explainability/explainers/explainer_factory.py` | Created | `ExplainerFactory`, `BuiltExplainer` |
| `src/shap_explainability/explainers/shap_service.py` | Created | `SHAPService`, `SHAPResult` |
| `tests/unit/test_explainer_factory.py` | Created | 16 tests for ExplainerFactory |
| `tests/unit/test_shap_service.py` | Created | 45 tests for SHAPService |

---

## Implementation Details

### errors.py

One new exception class added:

```
SHAPModuleError (base)
  ModelLoadError       (Phase 3)
  DatasetLoadError     (Phase 3)
  ModelValidationError (Phase 4)
  SchemaValidationError(Phase 4)
  SHAPExecutionError   (Phase 5)  <-- new
```

`SHAPExecutionError` is raised for: unsupported model family in config, SHAP
explainer construction failure, `shap_values()` failure, and shape normalisation
failure. Always raised after `session_context.mark_failed()` in the factory.

### config/model_type_detection.json

Three new sections added alongside existing Phase 4 sections:

```json
"model_family_to_explainer": {
    "XGBoost": "TreeExplainer",
    "RandomForest": "TreeExplainer",
    "LightGBM": "TreeExplainer",
    "CatBoost": "TreeExplainer",
    "LogisticRegression": "LinearExplainer"
},
"class_name_to_prediction_category": {
    "XGBClassifier": "classification",
    "XGBRegressor": "regression",
    "RandomForestClassifier": "classification",
    "RandomForestRegressor": "regression",
    "LGBMClassifier": "classification",
    "LGBMRegressor": "regression",
    "CatBoostClassifier": "classification",
    "CatBoostRegressor": "regression",
    "LogisticRegression": "classification"
},
"tree_explainer_kwargs_by_family": {
    "XGBoost": {},
    "RandomForest": {},
    "LightGBM": {},
    "CatBoost": {"check_additivity": false}
}
```

CatBoost's `check_additivity: false` entry mitigates R-02 from the pre-implementation
review without any if-else in code -- SHAPService reads these kwargs and unpacks them
into the `shap_values()` call.

### config/config.ini

New `[model]` section added:

```ini
[model]
LINEAR_EXPLAINER_BACKGROUND_SAMPLES = 200
```

The pipeline orchestrator (Phase 8) reads this value and passes it to
`ExplainerFactory.__init__(linear_background_samples=...)` at construction time.

### explainers/explainer_factory.py

**Class:** `BuiltExplainer` (frozen dataclass)

Fields: `explainer_object: Any`, `explainer_name: str`, `model_family: str`

**Class:** `ExplainerFactory`

| Method | Purpose |
|---|---|
| `__init__` | Load `model_family_to_explainer` from JSON; accepts optional `linear_background_samples` |
| `create` | Config-driven dispatch: look up explainer type, build, write to SessionContext, return `BuiltExplainer` |
| `_dispatch_explainer_build` | Hash-map dispatch to build method (no if-else ladder) |
| `_build_tree_explainer` | `shap.TreeExplainer(model_object)` -- identical for all 4 tree families |
| `_build_linear_explainer` | `shap.LinearExplainer(model_object, shap.maskers.Independent(background_df))` |
| `_load_family_to_explainer_config` | JSON load with error handling |

**LinearExplainer background sampling:** If `len(feature_dataframe) > linear_background_samples`,
a random subsample (seed=42) is used. Otherwise the full DataFrame is used. This resolves
`architecture.md [OPEN A-5]`.

**Unsupported family:** If `model_family` is not in the JSON map, `SHAPExecutionError`
is raised and `session_context.mark_failed()` is called before raising.

### explainers/shap_service.py

**Class:** `SHAPResult` (frozen dataclass)

Fields: `prediction_type`, `shap_values_array`, `feature_names`, `class_names`,
`global_importance_dataframe`, `mapping_dataframe`

**Class:** `SHAPService`

| Method | Purpose |
|---|---|
| `__init__` | Load `class_name_to_prediction_category` and `tree_explainer_kwargs_by_family` from JSON |
| `compute` | Orchestrates full SHAP computation: type detection, run, normalise, aggregate, build mapping |
| `detect_prediction_type` | Config-driven category lookup + `n_classes_`/`classes_` introspection |
| `_resolve_num_classes` | Inspects `n_classes_` then `classes_`, falls back to 2 with warning |
| `_run_explainer` | Calls `explainer_object.shap_values(feature_dataframe, **shap_kwargs)` |
| `_normalize_shap_values` | Dispatches to `_normalize_regression_shap`, `_normalize_binary_shap`, or `_normalize_multiclass_shap` |
| `_normalize_binary_shap` | list input: take `[1]`; ndarray: use as-is |
| `_normalize_multiclass_shap` | list: use as-is; 3D ndarray: slice into list by class axis |
| `_compute_global_importance` | Binary/Regression: `mean(abs, axis=0)`; Multiclass: per-class mean then mean across classes |
| `_build_mapping_dataframe` | Dispatches to `_build_flat_mapping` or `_build_multiclass_mapping` |
| `_get_class_names` | Multiclass: extract `model.classes_`, format integers as `class_{i}` |

### Design Decisions

**SHAP value normalisation (R-01 mitigation):**
All shape handling is centralised in three private static methods
(`_normalize_regression_shap`, `_normalize_binary_shap`, `_normalize_multiclass_shap`)
so per-library shape differences are isolated in one place. The canonical shapes are:
- Binary/Regression: `ndarray (n_samples, n_features)`
- Multiclass: `list[ndarray]` (one array per class, each `(n_samples, n_features)`)

**CatBoost `check_additivity=False` (R-02 mitigation):**
Configured in `tree_explainer_kwargs_by_family` JSON section. SHAPService looks up
the model family from `BuiltExplainer.model_family` and unpacks the kwargs dict into
`shap_values(**shap_kwargs)`. No if-else in code.

**`feature_dataframe` flow (R-08 / A-5 resolution):**
`feature_dataframe` is received as an explicit parameter by both `ExplainerFactory.create()`
and `SHAPService.compute()`. It is NOT stored on SessionContext (Option A from
pre-implementation doc) to avoid memory overhead for large datasets. Phase 8 pipeline
will carry it as a local variable.

**`prediction_type` on SessionContext:**
`SHAPResult.prediction_type` carries the value for downstream use. Phase 8 pipeline
writes it to `session_context.extra_metadata` or a new `prediction_type` field
(minor SessionContext extension deferred to Phase 8).

---

## Test Coverage

| Test File | Tests | Coverage |
|---|---|---|
| `test_explainer_factory.py` | 16 | Config loading, all 5 model families, unsupported family error, session context write, background sampling behaviour, construction failure wrapping |
| `test_shap_service.py` | 45 | Prediction type detection (all 9 class names), fallback to binary, all 3 normalisation paths (binary list, binary ndarray, multiclass list, multiclass 3D, regression), global importance values and sort order, mapping column schemas, mapping row counts, class name extraction (string labels, integer labels, missing classes_), CatBoost kwargs pass-through, XGBoost no-extra-kwargs, session context writes, frozen SHAPResult |

**Phase 5 total: 61 new tests**
**Full suite total: 187 tests, all passing**

### Key test scenarios verified

- All four tree families produce `explainer_name == "TreeExplainer"`
- LogisticRegression produces `explainer_name == "LinearExplainer"`
- Unsupported family raises `SHAPExecutionError` AND marks context FAILED
- Linear background sampling caps at `linear_background_samples` rows
- Binary list (RandomForest) normalisation takes index `[1]` (positive class)
- Multiclass 3D ndarray (XGBoost/CatBoost) sliced into list of K 2D arrays
- Global importance sorted descending by `mean_absolute_shap_value`
- Multiclass global importance averaged across all K class importance vectors
- Binary mapping has 4 columns (no `class_name`); multiclass has 5 columns
- Multiclass row count = n_samples x n_features x n_classes
- `record_id` values are sequential integers starting at 0
- CatBoost: `shap_values(df, check_additivity=False)` verified via mock
- XGBoost: `shap_values(df)` with no extra kwargs verified via mock
- Integer `model.classes_` formatted as `"class_0"`, `"class_1"`, ...
- String `model.classes_` used directly (e.g. `"cat"`, `"dog"`, `"fish"`)
- `session_context.shap_values`, `.global_feature_importance`, `.feature_shap_mapping` all written

---

## Acceptance Criteria Traceability

| AC | Description | Phase 5 Coverage |
|---|---|---|
| AC-05 | System selects appropriate SHAP explainer for detected model type | Covered by `ExplainerFactory` config-driven dispatch |
| AC-06 | SHAP values computed for all records in the feature-only dataset | Covered by `SHAPService.compute` |
| AC-07 | Binary classification SHAP values represent positive class contributions | `_normalize_binary_shap` selects index [1] for list output |
| AC-08 | Multiclass SHAP values decomposed per-class | `_normalize_multiclass_shap` produces `list[ndarray]` |
| AC-09 | Global feature importance computed as mean absolute SHAP value | `_compute_global_importance` with axis=0 mean |
| AC-10 | Feature-SHAP mapping table built per Sec 17.2 schema | `_build_mapping_dataframe` / `_build_flat_mapping` / `_build_multiclass_mapping` |
| AC-11 | CatBoost additivity check suppressed | `tree_explainer_kwargs_by_family` + `_run_explainer` kwargs unpacking |

AC-12 through AC-15 remain pending Phase 6-7 (exporters, visualisations, metadata).

---

## What Phase 6 Can Now Assume

After Phase 5, the following are available for direct consumption:

- `SHAPResult.global_importance_dataframe` -- `feature_name`, `mean_absolute_shap_value`
  columns; n_features rows; sorted descending. Ready for `to_csv()` (Spec Sec 17.1).
- `SHAPResult.mapping_dataframe` -- per Sec 17.2 column contract for the detected
  prediction type. Ready for `to_csv()` (Spec Sec 17.2).
- `SHAPResult.shap_values_array` -- canonical normalised form accepted directly by
  `shap.summary_plot()` for all three prediction types.
- `SHAPResult.feature_names` and `SHAPResult.class_names` -- axis label strings for
  visualiser and metadata exporter.
- `SHAPResult.prediction_type` -- controls plot type parameter and CSV schema selection.
- `SessionContext.shap_values`, `.global_feature_importance`, `.feature_shap_mapping` --
  populated and available to any Phase 6+ component reading from context.

---

## Bug Fixed During Implementation

**`log_shap_generation` missing `level` parameter:**
`ExecutionLogger.log_shap_generation()` did not accept a `level` argument, unlike the
other log methods (`log_schema_validation`, `log_model_validation`, etc.) which all
accept an optional `level: int = logging.INFO`. Fixed by adding the `level` parameter
to bring the method into line with the established pattern. Two tests caught this
during the first test run.
