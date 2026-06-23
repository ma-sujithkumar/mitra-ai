# Phase 6 Pre-Implementation Review
**Date:** 2026-06-17
**Branch:** epic4-shap
**Preceding phase:** Phase 5 complete -- 187 tests passing

---

## 1. Scope

Phase 6 implements the three output exporters and promotes `SHAPResult` to its own
`models/` sub-package. All inputs (DataFrames, SessionContext) are already produced by
Phase 5's `SHAPService.compute()`. Phase 6 is pure I/O -- no SHAP computation logic.

Do NOT implement:
- Visualizations / plots (Phase 7)
- Pipeline orchestration (Phase 8)
- Summary/beeswarm plots

---

## 2. Files to Create

| File | Class | Purpose |
|---|---|---|
| `src/shap_explainability/models/__init__.py` | -- | Package marker |
| `src/shap_explainability/models/shap_result.py` | `SHAPResult` | Frozen result dataclass (moved from shap_service.py) |
| `src/shap_explainability/exporters/__init__.py` | -- | Package marker |
| `src/shap_explainability/exporters/global_importance_exporter.py` | `GlobalImportanceExporter` | Writes `global_feature_importance.csv` |
| `src/shap_explainability/exporters/feature_shap_mapping_exporter.py` | `FeatureSHAPMappingExporter` | Writes `feature_shap_mapping.csv` |
| `src/shap_explainability/exporters/metadata_exporter.py` | `MetadataExporter` | Writes `metadata.json` |
| `tests/unit/test_global_importance_exporter.py` | -- | Unit tests |
| `tests/unit/test_feature_shap_mapping_exporter.py` | -- | Unit tests |
| `tests/unit/test_metadata_exporter.py` | -- | Unit tests |

## 3. Files to Modify

| File | Change |
|---|---|
| `src/shap_explainability/errors.py` | Add `ExportError` |
| `src/shap_explainability/explainers/shap_service.py` | Replace local `SHAPResult` definition with import from `models.shap_result` |

---

## 4. Design Decisions

### 4.1 SHAPResult relocation

`SHAPResult` lives in `shap_service.py` today. Moving it to `models/shap_result.py`
separates the computation concern (SHAPService) from the result-schema concern (SHAPResult).
`shap_service.py` will import it from there. All other consumers (tests, future Phase 8
pipeline) will also import from `models.shap_result`.

`BuiltExplainer` stays in `explainer_factory.py` -- it is output of that specific module
and no Phase 6 component needs it.

### 4.2 CSV exporters: minimal surface

Each exporter exposes a single `export(dataframe, output_path) -> Path` method.
The DataFrame passed in is already fully formed by SHAPService (`global_importance_dataframe`
and `mapping_dataframe` on `SHAPResult`) -- no re-computation inside the exporters.

Float precision: `float_format="%.17g"` to satisfy spec Sec 17 "full available numerical
precision" requirement.

### 4.3 MetadataExporter: assembles from SessionContext + Optional[SHAPResult]

`MetadataExporter.export(session_context, shap_result, output_path)` accepts
`shap_result: Optional[SHAPResult]` so it can be called on both success paths
(shap_result populated) and early-failure paths (shap_result=None, per architecture.md
Section 5 failure path guarantee).

Fields assembled into metadata.json (spec Sec 18):

| JSON Key | Source |
|---|---|
| `session_id` | `session_context.session_id` |
| `provided_model_name` | `session_context.supplied_model_name` |
| `detected_model_type` | `session_context.detected_model_type` |
| `validation_status` | `session_context.execution_status.value` |
| `model_name_validation_status` | `session_context.model_name_validation_status.value` (or null) |
| `model_name_validation_message` | `session_context.model_name_validation_message` (or null) |
| `explainer` | `session_context.explainer_name` |
| `prediction_type` | `shap_result.prediction_type` (or null when shap_result is None) |
| `num_samples` | `session_context.num_samples` |
| `num_features` | `session_context.num_features` |
| `execution_timestamp` | `session_context.created_at.isoformat()` |
| `warnings` | `session_context.warnings` (list, may be empty) |
| `error_message` | `session_context.error_message` (or null) |

### 4.4 ExportError (new exception class)

Wraps `OSError` / `IOError` from file I/O in all three exporters. Inherits from
`SHAPModuleError` so the pipeline failure path can catch it uniformly.

### 4.5 Directory creation

Each exporter calls `output_path.parent.mkdir(parents=True, exist_ok=True)` before
writing, consistent with CLAUDE.md rule 17 (always mkdir -p for output dirs).

### 4.6 Star-shaped dependency rule

Exporters import from:
- `shap_explainability.models.shap_result` (SHAPResult)
- `shap_explainability.session_context` (SessionContext, for MetadataExporter)
- `shap_explainability.errors` (ExportError)
- `shap_explainability.utils.logger` (ExecutionLogger)

Exporters do NOT import from `explainers/` or `validators/`.

---

## 5. Risk Analysis

| Risk | Mitigation |
|---|---|
| R-01: Circular import after SHAPResult move | shap_service.py imports SHAPResult from models; models does NOT import from explainers -- no cycle |
| R-02: MetadataExporter called before SHAPService (early failure path) | shap_result parameter is Optional[SHAPResult]; all fields sourced from SHAPResult are null-safe |
| R-03: Output path parent directory does not exist | Every exporter calls parent.mkdir(parents=True, exist_ok=True) before write |
| R-04: Float precision loss in CSV | float_format="%.17g" on to_csv() gives full IEEE754 double precision |
| R-05: JSON serialization of datetime / enum values | json.dumps uses default=str to handle datetime.isoformat() and enum values |

---

## 6. Test Plan

### 6.1 test_global_importance_exporter.py (~10 tests)

- File created at expected path
- Return value is the output path
- Columns are feature_name, mean_absolute_shap_value (exact)
- Row count equals n_features
- Feature names preserved exactly
- Values preserved numerically (round-trip)
- Rows sorted descending by value (already sorted by SHAPService, exporter preserves order)
- Parent directory created if missing
- ExportError raised on I/O failure

### 6.2 test_feature_shap_mapping_exporter.py (~10 tests)

- File created at expected path
- Return value is the output path
- Binary/Regression: 4 columns (record_id, feature_name, feature_value, shap_value)
- Multiclass: 5 columns (+ class_name)
- Row count binary: n_samples * n_features
- Row count multiclass: n_samples * n_features * n_classes
- record_id values are sequential integers
- Parent directory created if missing
- ExportError raised on I/O failure

### 6.3 test_metadata_exporter.py (~15 tests)

- File created at expected path
- Return value is the output path
- Required fields present: session_id, provided_model_name, detected_model_type,
  validation_status, explainer, execution_timestamp
- session_id matches
- provided_model_name matches
- validation_status is "running"/"success"/"warning"/"failed"
- prediction_type set when shap_result provided
- prediction_type is null when shap_result=None
- warnings list included
- error_message present on failure context
- execution_timestamp is valid ISO-8601 string
- Parent directory created if missing
- ExportError raised on I/O failure

---

## 7. Acceptance Criteria Traceability

| AC | Description | Phase 6 Coverage |
|---|---|---|
| AC-12 | global_feature_importance.csv generated with required columns | `GlobalImportanceExporter.export()` |
| AC-13 | feature_shap_mapping.csv generated with required columns (per prediction type) | `FeatureSHAPMappingExporter.export()` |
| AC-14 | metadata.json generated with required fields | `MetadataExporter.export()` |
| AC-15 | Meaningful errors and logs on failure (partial metadata written) | `MetadataExporter` accepts Optional[SHAPResult] |

AC-16 (one row per record-feature) is guaranteed by SHAPService (Phase 5) building the
mapping DataFrame; Phase 6 exporters verify column counts in tests.
AC-17 (metadata fields) tested by `test_metadata_exporter.py`.

---

## 8. What Phase 7 Can Assume After Phase 6

- `SHAPResult` importable from `shap_explainability.models.shap_result`
- `GlobalImportanceExporter`, `FeatureSHAPMappingExporter`, `MetadataExporter`
  importable from their respective `exporters/` modules
- `ExportError` importable from `shap_explainability.errors`
- All three exporters accept `output_path: Path` from `OutputManager.csv_path()` or
  `OutputManager.metadata_path()`
- `MetadataExporter.export(session_context, shap_result, output_path)` handles None
  shap_result for early-failure invocations
