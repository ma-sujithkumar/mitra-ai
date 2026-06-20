# Phase 5 Pre-Implementation Review
**Date:** 2026-06-17
**Branch:** epic4-shap
**Scope:** `explainers/explainer_factory.py`, `explainers/shap_service.py`
**Spec references:** Sections 6.1, 14, 15 (steps 7-9), 17, 18, 24
**Architecture references:** Sections 3, 4, 5 (steps 7-9)

---

## 1. Architecture Review

### What Phase 4 delivered (available to Phase 5)

| SessionContext field | Set by | Phase 5 reads |
|---|---|---|
| `detected_model_type` | ModelLoader | Yes — class name for prediction type detection |
| `model_name_validation_status` | ModelValidator | No — metadata concern only |
| `feature_names` | SchemaValidator | Yes — used for importance aggregation and mapping |
| `target_column_name` | SchemaValidator | No — already excluded from DataFrame |
| `num_samples` / `num_features` | SchemaValidator | Verification only |
| `explainer_name` | ExplainerFactory (Phase 5) | Written here |
| `shap_values` | SHAPService (Phase 5) | Written here |
| `global_feature_importance` | SHAPService (Phase 5) | Written here |
| `feature_shap_mapping` | SHAPService (Phase 5) | Written here |

`SchemaValidationResult.feature_dataframe` (cleaned, target-excluded) is the primary input
to both ExplainerFactory and SHAPService. It is NOT currently on SessionContext — see
Section 9 (Gap: SessionContext fields).

### Module boundary (star-shaped dependency rule)

Phase 5 modules (`explainer_factory.py`, `shap_service.py`) must:
- Import from `shap_explainability.session_context`, `shap_explainability.errors`,
  `shap_explainability.utils.logger` (cross-cutting)
- NOT import from each other
- NOT import from loaders or validators
- Receive `feature_dataframe` and `model_object` as explicit arguments from the pipeline
  orchestrator (`SHAPPipeline`, Phase 8)

---

## 2. Proposed Classes

### 2.1 ExplainerFactory (`explainers/explainer_factory.py`)

**Responsibility:** Map a confirmed model family to a constructed SHAP explainer instance.
No SHAP value computation occurs here.

**Result container** (frozen dataclass):
```
BuiltExplainer
    explainer_object: Any          # shap.TreeExplainer or shap.LinearExplainer instance
    explainer_name: str            # "TreeExplainer" / "LinearExplainer" (for metadata)
    model_family: str              # e.g. "XGBoost" (for traceability)
```

**Class:** `ExplainerFactory`

| Method | Signature | Purpose |
|---|---|---|
| `__init__` | `(execution_logger, model_type_config_path=None)` | Load JSON config, read `model_family_to_explainer` section |
| `create` | `(model_family, model_object, feature_dataframe) -> BuiltExplainer` | Select and construct explainer |
| `_build_tree_explainer` | `(model_object) -> Any` | Construct `shap.TreeExplainer(model_object)` |
| `_build_linear_explainer` | `(model_object, feature_dataframe) -> Any` | Construct with background masker (see Section 5) |
| `_load_type_detection_config` | `(config_path) -> dict` | Read JSON, raise on missing |

### 2.2 SHAPService (`explainers/shap_service.py`)

**Responsibility:** Run the explainer, normalize SHAP value shapes, detect prediction type,
compute global importance, produce the long-form per-record/per-feature mapping table.

**Result container** (frozen dataclass):
```
SHAPResult
    prediction_type: str                   # "binary_classification" / "multiclass_classification" / "regression"
    shap_values_array: Any                 # normalized: ndarray (n_samples, n_features) for binary/regression;
                                           # list of K ndarrays for multiclass
    feature_names: tuple[str, ...]         # ordered feature names (echoed from input for downstream use)
    class_names: Optional[tuple[str, ...]] # class labels for multiclass (None for binary/regression)
    global_importance_dataframe: pd.DataFrame  # columns: feature_name, mean_absolute_shap_value
    mapping_dataframe: pd.DataFrame            # columns: per Spec Sec 17.2 schema for detected prediction type
```

**Class:** `SHAPService`

| Method | Signature | Purpose |
|---|---|---|
| `__init__` | `(execution_logger, model_type_config_path=None)` | Load `class_name_to_prediction_category` from JSON config |
| `compute` | `(built_explainer, feature_dataframe, feature_names, model_object, detected_class_name, session_context) -> SHAPResult` | Full computation: detect type, run SHAP, normalize, aggregate |
| `detect_prediction_type` | `(model_object, detected_class_name) -> str` | Config-driven category lookup + `n_classes_`/`classes_` introspection |
| `_run_explainer` | `(explainer_object, feature_dataframe) -> Any` | Call `.shap_values()` or `.__call__()` |
| `_normalize_shap_values` | `(raw_shap_output, prediction_type) -> Any` | Normalize to canonical internal shape |
| `_compute_global_importance` | `(shap_values_array, feature_names, prediction_type) -> pd.DataFrame` | Mean absolute SHAP per feature |
| `_build_mapping_dataframe` | `(shap_values_array, feature_dataframe, feature_names, prediction_type, class_names) -> pd.DataFrame` | Long-form per Sec 17.2 |
| `_get_class_names` | `(model_object, prediction_type) -> Optional[tuple[str, ...]]` | Extract from `model.classes_` or generate `class_0, class_1, ...` |

---

## 3. SHAP Explainer Selection Strategy

Config-driven mapping added to `config/model_type_detection.json` under a new key
`model_family_to_explainer`. No if-else ladder in code (CLAUDE.md rule 4).

```
Model Family       => Explainer
XGBoost            => TreeExplainer
RandomForest       => TreeExplainer
LightGBM           => TreeExplainer
CatBoost           => TreeExplainer
LogisticRegression => LinearExplainer
```

`ExplainerFactory` reads this mapping at `__init__` time. The `create()` method:
1. Looks up `model_family` in the config map.
2. Dispatches to `_build_tree_explainer` or `_build_linear_explainer`.
3. Returns a `BuiltExplainer` with the explainer name string for metadata.

If `model_family` is not in the map, `ExplainerFactory` raises `SHAPExecutionError`
(new exception class needed — see Section 9) and marks `session_context.mark_failed()`.

---

## 4. TreeExplainer Mapping

All four tree-based families use `shap.TreeExplainer(model_object)` with identical
construction. No background data is required for TreeExplainer.

| Family | Class examples | TreeExplainer notes |
|---|---|---|
| XGBoost | XGBClassifier, XGBRegressor | Native SHAP support via XGBoost's internal implementation |
| RandomForest | RandomForestClassifier, RandomForestRegressor | sklearn's tree ensemble; may return list-per-class or 3D array |
| LightGBM | LGBMClassifier, LGBMRegressor | Native SHAP support; similar shape behavior to XGBoost |
| CatBoost | CatBoostClassifier, CatBoostRegressor | Native SHAP support; `shap.TreeExplainer` is preferred over CatBoost's own API for consistency |

Construction call: `shap.TreeExplainer(model_object)`

No `feature_perturbation` or `data` argument at construction for tree models. The
`check_additivity=False` flag may need to be passed to `.shap_values()` for CatBoost to
avoid a sum-check error — document as a known per-family configuration parameter in
`model_type_detection.json` (new section `tree_explainer_kwargs_by_family`).

---

## 5. LinearExplainer Mapping

`shap.LinearExplainer` requires background data at construction time. This resolves
`[OPEN A-5]` from `architecture.md`.

**Adopted decision:**
```
shap.LinearExplainer(model_object, shap.maskers.Independent(feature_dataframe))
```

The feature DataFrame (already cleaned and target-excluded by SchemaValidator) is used
as the background population. This is appropriate for global explainability because:
- The entire inference population is the distribution of interest.
- No separate training data reference is available to the SHAP module (spec.md Sec 2:
  module SHALL NOT perform training, and Epic 3 artifact is the model only).
- `shap.maskers.Independent` models feature independence, which is the standard masker
  for linear models.

**Memory consideration:** If `num_samples` is very large, `shap.sample(feature_dataframe, N)`
should be used as the background instead. Add a configurable `linear_explainer_background_samples`
parameter to `config.ini` with a sensible default (e.g. 200). If `num_samples <= default`,
use the full DataFrame; otherwise sample.

**This assumption must be reviewed during integration** when dataset sizes from Epic 2
are known.

---

## 6. Binary Classification Handling

**Prediction type detection:**
- Config category for detected class name: `"classification"`
- `model.classes_` or `model.n_classes_` introspection: `n_classes == 2` → BINARY

**SHAP value shape from TreeExplainer (binary):**
Different libraries return different shapes for binary classification. The `_normalize_shap_values`
method must handle all of:

| Library | Raw shape returned by `.shap_values()` | Normalized to |
|---|---|---|
| XGBoost | `(n_samples, n_features)` — class 1 values directly | Use as-is |
| RandomForest | `[array(n_samples, n_features), array(n_samples, n_features)]` — one per class | Take index `[1]` (positive class) |
| LightGBM | `(n_samples, n_features)` | Use as-is |
| CatBoost | `(n_samples, n_features)` | Use as-is |
| LogisticRegression (Linear) | `(n_samples, n_features)` | Use as-is |

**Normalized canonical shape:** `ndarray (n_samples, n_features)` — represents SHAP
contributions toward the positive class (class index 1).

**Global importance:** `mean(|shap_values|, axis=0)` → `(n_features,)` vector.

**Mapping schema (Sec 17.2):** `record_id, feature_name, feature_value, shap_value`
— no `class_name` column for binary.

---

## 7. Multiclass Handling

**Prediction type detection:**
- Config category: `"classification"`
- `n_classes_ > 2` → MULTICLASS

**SHAP value shape from TreeExplainer (multiclass):**

| Library | Raw shape | Normalized to |
|---|---|---|
| XGBoost | 3D `(n_samples, n_features, n_classes)` — newer versions | List of K arrays: `[arr[:, :, k] for k in range(n_classes)]` |
| XGBoost (older) | List of K `(n_samples, n_features)` arrays | Use as-is |
| RandomForest | List of K `(n_samples, n_features)` arrays | Use as-is |
| LightGBM | List of K `(n_samples, n_features)` arrays | Use as-is |
| CatBoost | `(n_samples, n_features, n_classes)` | Slice into list |

**Normalized canonical shape:** `list[ndarray(n_samples, n_features)]` — one array per
class, in class-index order.

**Class names:** Extracted from `model.classes_` if available. If `model.classes_` contains
integers (common), format as `class_0, class_1, ...` for CSV column values. If it contains
string labels (e.g. `["cat", "dog", "fish"]`), use those directly.

**Global importance:** Average `mean(|values|, axis=0)` across all K class arrays, then
reduce to `(n_features,)`. This represents model-wide feature importance agnostic to
which class is being predicted.

**Mapping schema (Sec 17.2):** `record_id, class_name, feature_name, feature_value, shap_value`
— one row per record × class × feature combination. For `n_samples=1000`, `n_features=20`,
`n_classes=3`, the output has 60,000 rows. Document this scale in tests.

---

## 8. Regression Handling

**Prediction type detection:**
- Config category: `"regression"` — detected directly from class name containing "Regressor"
- No `n_classes_` check needed (regressors do not have this attribute)

**SHAP value shape from TreeExplainer (regression):** `(n_samples, n_features)` — single
output, no class dimension. All tested libraries return the same shape for regression.

**LinearExplainer for regression:** Not currently in the supported model list (Logistic
Regression is classification-only). If a future linear model type is added, no shape
difference exists — same `(n_samples, n_features)` shape.

**Global importance:** `mean(|shap_values|, axis=0)` — identical to binary.

**Mapping schema (Sec 17.2):** `record_id, feature_name, feature_value, shap_value`
— same schema as binary classification, no `class_name` column.

---

## 9. SHAP Output Data Structures

### 9.1 BuiltExplainer (frozen dataclass, lives in `explainer_factory.py`)

```
BuiltExplainer
    explainer_object: Any          # shap.TreeExplainer or shap.LinearExplainer
    explainer_name: str            # "TreeExplainer" / "LinearExplainer"
    model_family: str              # e.g. "XGBoost"
```

### 9.2 SHAPResult (frozen dataclass, lives in `shap_service.py`)

```
SHAPResult
    prediction_type: str
        Values: "binary_classification" | "multiclass_classification" | "regression"
    shap_values_array: Any
        Binary/Regression: ndarray shape (n_samples, n_features)
        Multiclass:        list of K ndarrays, each shape (n_samples, n_features)
    feature_names: tuple[str, ...]
        Length n_features; same ordering as the cleaned feature DataFrame columns
    class_names: Optional[tuple[str, ...]]
        Populated for multiclass only; None for binary and regression
        Values: ("class_0", "class_1", ...) or actual model.classes_ string labels
    global_importance_dataframe: pd.DataFrame
        Columns: feature_name (str), mean_absolute_shap_value (float)
        Rows: n_features, sorted descending by mean_absolute_shap_value
    mapping_dataframe: pd.DataFrame
        Binary/Regression columns: record_id (int), feature_name (str),
                                   feature_value (float), shap_value (float)
        Multiclass columns:        record_id (int), class_name (str),
                                   feature_name (str), feature_value (float),
                                   shap_value (float)
```

### 9.3 New exception class required

```
SHAPExecutionError(SHAPModuleError)
    Raised when: explainer construction fails, shap_values() call fails,
                 normalization cannot produce a canonical shape, or the
                 model family has no explainer mapping entry.
```

Must be added to `errors.py` before Phase 5 implementation begins.

### 9.4 SessionContext fields required from Phase 5

Currently absent from SessionContext — must be added before Phase 5:

| Field | Type | Set by | Consumed by |
|---|---|---|---|
| `prediction_type` | `Optional[str]` | SHAPService | MetadataExporter (Phase 7), PlotGenerator (Phase 6) |
| `feature_dataframe` | `Optional[pd.DataFrame]` | SchemaValidator (already exists in SchemaValidationResult, not yet on SessionContext) | ExplainerFactory, SHAPService, PlotGenerator |

**Options:**
- **Option A (recommended):** Pipeline orchestrator (SHAPPipeline, Phase 8) carries
  `feature_dataframe` as a local variable and passes it explicitly to ExplainerFactory
  and SHAPService as arguments. SessionContext stores `prediction_type` only.
  Avoids storing a potentially large DataFrame on the context object permanently.
- **Option B:** Add `feature_dataframe: Optional[pd.DataFrame]` to SessionContext.
  Simpler star-shaped flow but risks memory overhead for large datasets.

For Phase 5 unit tests, pass `feature_dataframe` directly — no SessionContext change
needed for Phase 5 itself. Phase 8 (pipeline) will resolve the carrying strategy.

---

## 10. Dependencies on Phase 1-4 Components

| Phase 5 component | Depends on | What it needs |
|---|---|---|
| `ExplainerFactory.__init__` | `config/model_type_detection.json` | Reads new `model_family_to_explainer` section |
| `ExplainerFactory.create` | `LoadedModel.model_object` (Phase 3) | Raw model object passed as argument |
| `ExplainerFactory._build_linear_explainer` | `SchemaValidationResult.feature_dataframe` (Phase 4) | Background data for masker |
| `SHAPService.compute` | `BuiltExplainer` (Phase 5 ExplainerFactory) | Explainer instance |
| `SHAPService.detect_prediction_type` | `SessionContext.detected_model_type` (Phase 3) | Class name for config lookup |
| `SHAPService.detect_prediction_type` | `LoadedModel.model_object` (Phase 3) | `n_classes_` / `classes_` introspection |
| `SHAPService._build_mapping_dataframe` | `SchemaValidationResult.feature_dataframe` (Phase 4) | Original feature values for `feature_value` column |
| `SHAPService.compute` | `SessionContext.feature_names` (Phase 4) | Ordered feature name list |
| Both | `ExecutionLogger` (Phase 1) | Structured event logging |
| Both | `SessionContext` (Phase 1) | `explainer_name`, `shap_values`, `global_feature_importance`, `feature_shap_mapping`, `prediction_type` written here |

**No Phase 5 code imports loaders or validators** — it receives their outputs as
arguments from the pipeline orchestrator.

---

## 11. Integration Points with Future CSV Exporters (Phase 6)

`CSVExporter` (Phase 6) will receive:

| What | From | Column contract |
|---|---|---|
| `SHAPResult.global_importance_dataframe` | SHAPService | `feature_name`, `mean_absolute_shap_value` (Sec 17.1) |
| `SHAPResult.mapping_dataframe` | SHAPService | Binary/Regression: `record_id, feature_name, feature_value, shap_value`; Multiclass: `+class_name` (Sec 17.2) |

**Contract established here:** The DataFrames produced by SHAPService must have exactly
these column names before CSVExporter writes them. No column renaming in CSVExporter.
CSVExporter is pure I/O — it calls `dataframe.to_csv(path, index=False)` only.

`SHAPResult` can be passed directly to `CSVExporter` or its two DataFrames can be passed
individually. The former is preferred (single result object, typed).

`prediction_type` is also needed by CSVExporter to decide which schema to validate
(for test assertions — the actual write is always `to_csv`). CSVExporter can read it
from `SHAPResult.prediction_type`.

---

## 12. Integration Points with Future Visualization Layer (Phase 6)

`PlotGenerator` (Phase 6) will need:

| What | From | Used for |
|---|---|---|
| `SHAPResult.shap_values_array` | SHAPService | Summary plot, beeswarm plot (raw values) |
| `SHAPResult.feature_names` | SHAPService | All three plots (axis labels) |
| `feature_dataframe` | SchemaValidationResult or SessionContext | `shap.summary_plot(shap_values, features=feature_dataframe)` — original feature values required |
| `SHAPResult.prediction_type` | SHAPService | Summary plot type parameter (`plot_type="bar"` vs `"dot"`) |
| `SHAPResult.class_names` | SHAPService | Multiclass: class label axis annotation |
| Output paths | OutputManager (Phase 1) | Three PNG output paths |

**Key design constraint for PlotGenerator:** `shap.summary_plot` requires the original
feature values (not just SHAP values) to render color gradients. This means `feature_dataframe`
must be available to PlotGenerator. Phase 5 must ensure `feature_dataframe` is either
on `SessionContext` or explicitly passed through the pipeline chain to Phase 6.
See Section 9.4 for the recommended resolution.

For multiclass: SHAP's plotting functions accept lists of arrays natively. The
normalized `list[ndarray]` from `SHAPResult.shap_values_array` can be passed directly
to `shap.summary_plot`.

---

## 13. Risks and Assumptions

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-01 | SHAP value shape variability across library versions (list vs 3D array) | HIGH | `_normalize_shap_values` centralizes all shape handling; test each of the five model families independently; pin `shap` version in `requirements.txt` |
| R-02 | CatBoost `check_additivity` error during `shap_values()` | MEDIUM | Pass `check_additivity=False` for CatBoost; configure in `tree_explainer_shap_kwargs_by_family` section of JSON config |
| R-03 | LinearExplainer background data policy (A-5) | LOW-MEDIUM | Resolved as: use feature DataFrame with `shap.maskers.Independent`; add `linear_explainer_background_samples` to `config.ini`; document assumption explicitly |
| R-04 | `n_classes_` not present on unfitted or custom model objects | LOW | Guard with `hasattr` check; fall back to BINARY assumption and add warning to SessionContext |
| R-05 | SHAP execution memory: large datasets produce large SHAP matrices | MEDIUM | `SHAPService` is stateless; caller (pipeline) controls dataset size; no internal mitigation in Phase 5 |
| R-06 | `model.classes_` contains non-string labels (integers, numpy types) | LOW | `_get_class_names` converts to `str`; formats as `class_{i}` if integers or unknown type |
| R-07 | XGBoost `.shap_values()` deprecated in favor of `Explanation` API | LOW | Use `.shap_values(feature_dataframe)` for all explainers; isolate in `_run_explainer`; easy to update |
| R-08 | `prediction_type` not yet on SessionContext | LOW | No SessionContext change needed for Phase 5 itself; `SHAPResult.prediction_type` carries it; Phase 8 pipeline writes it to context |

### Assumptions adopted for Phase 5

| ID | Assumption | Spec traceability |
|---|---|---|
| A-P5-01 | LinearExplainer uses feature DataFrame as background via `shap.maskers.Independent` | Resolves architecture.md [OPEN A-5] |
| A-P5-02 | For binary classification, SHAP values represent contribution toward the positive class (class index 1) | Consistent with standard SHAP practice; not explicitly stated in spec.md |
| A-P5-03 | Multiclass global importance = mean across all class importance vectors | Implicit from Sec 17.1 single-table schema for all prediction types |
| A-P5-04 | Class names default to `"class_{i}"` strings when `model.classes_` contains integers | Sec 17.2 example uses `class_0`, `class_1` which confirms this convention |
| A-P5-05 | `shap` library version 0.41+ is installed (modern Explanation API available) | Checked against `requirements.txt` during implementation |

---

## 14. Readiness Assessment

### Confirmed inputs from Phase 1-4

- `SessionContext.feature_names` — populated and tested (42 tests passing)
- `SessionContext.detected_model_type` — populated and tested
- `SchemaValidationResult.feature_dataframe` — cleaned, target-excluded, tested
- `LoadedModel.model_object` — the fitted sklearn/xgb/lgbm/catboost object
- `config/model_type_detection.json` — extensible; only a new section `model_family_to_explainer` needed
- `ExecutionLogger` — log_explainer_selection and log_shap_generation event methods already implemented
- `errors.py` — needs one new exception class (`SHAPExecutionError`)

### Unresolved blockers

None. The single previously-open item (A-5, LinearExplainer background data) has a
documented resolution (Section 5 above). The `prediction_type` gap in SessionContext
does not block Phase 5 implementation — `SHAPResult` carries it, and the pipeline
(Phase 8) will write it to the context.

### What must be done before implementation begins

1. Add `SHAPExecutionError` to `errors.py`
2. Add `model_family_to_explainer` section to `model_type_detection.json`
3. Add `class_name_to_prediction_category` section to `model_type_detection.json`
4. Add `tree_explainer_shap_kwargs_by_family` section to `model_type_detection.json`
   (for CatBoost `check_additivity` configuration)
5. Add `linear_explainer_background_samples` key to `config.ini [model]` section

---

## READY FOR IMPLEMENTATION

**Justification:**

All Phase 1-4 inputs are confirmed and tested (126/126 tests passing). The SHAP
explainer selection contract (Spec Sec 14) is unambiguous. All SHAP output schemas
(Sec 17.1, 17.2) are fully specified for all three prediction types. The one previously
open architectural item (LinearExplainer background data, A-5) has a documented,
defensible resolution that does not require external confirmation. Config-driven
mapping for the JSON config extension is straightforward. No Phase 5 logic depends on
Phase 6, 7, or 8.

The normalization risk (R-01) is real but manageable: it is confined to `_normalize_shap_values`
and is fully exercised by the per-family unit tests. The implementation scope is two
files (plus pre-conditions in `errors.py` and JSON config), with well-defined
result types and clear integration contracts for the downstream visualization and
exporter phases.
