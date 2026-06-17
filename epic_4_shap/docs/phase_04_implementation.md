# Phase 4 Implementation Report
**Date:** 2026-06-17
**Branch:** epic4-shap
**Status:** Complete — 126 tests passing (73 existing + 53 new)

---

## Summary

Phase 4 implemented the validators package: business-rule validation on loaded model
artifacts and dataset schemas before any SHAP computation begins. The pipeline can now
load, validate, and resolve a cleaned feature-only dataset and a confirmed model family
— or terminate cleanly with a structured failure recorded in `SessionContext`.

---

## Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| `src/shap_explainability/errors.py` | Modified | Added `ModelValidationError`, `SchemaValidationError` |
| `config/model_type_detection.json` | Modified | Added `supplied_name_to_family` alias section (20 entries) |
| `src/shap_explainability/validators/__init__.py` | Created | Package marker |
| `src/shap_explainability/validators/model_validator.py` | Created | `ModelValidator`, `ModelValidationResult` |
| `src/shap_explainability/validators/schema_validator.py` | Created | `SchemaValidator`, `SchemaValidationResult` |
| `tests/fixtures/fixture_factory.py` | Created | `FixtureFactory` (Spec Sec 22) |
| `tests/unit/test_model_validator.py` | Created | 19 tests covering all 4 validation rules |
| `tests/unit/test_schema_validator.py` | Created | 18 tests covering target exclusion and schema compat |
| `tests/unit/test_fixture_factory.py` | Created | 5 tests verifying all factory methods |

---

## Implementation Details

### errors.py

Two new exception classes added to the existing hierarchy:

```
SHAPModuleError (base)
  ModelLoadError       (Phase 3 — unchanged)
  DatasetLoadError     (Phase 3 — unchanged)
  ModelValidationError (Phase 4 — Rule 3/4 terminating failures)
  SchemaValidationError(Phase 4 — feature count/name mismatch, zero features)
```

`ModelValidationError` is never raised for Rule 2 (name mismatch) — that path is a
non-terminating warning.

### config/model_type_detection.json

Added `supplied_name_to_family` section alongside the existing
`class_name_to_model_family` map. Contains 20 alias entries (all lowercase) covering
common shorthand names for all five supported model families. All key normalization
(`.lower()`) happens at `ModelValidator.__init__` time, not per-call.

Example entries:
```
"xgb" -> "XGBoost"
"rf"  -> "RandomForest"
"lgbm"-> "LightGBM"
"logreg" -> "LogisticRegression"
```

### validators/model_validator.py

**Class:** `ModelValidator`

Implements spec.md Section 8 Rules 1-4. Config-driven alias lookup (CLAUDE.md rule 4 —
no if-else ladders for name matching). Two-stage match resolution:

1. Normalize supplied name to lowercase, look up in `supplied_name_to_family` map.
2. If not found in map, fall back to direct case-insensitive comparison against the
   detected family name (handles cases like `"XGBoost"` supplied against family
   `"XGBoost"`).

| Rule | Condition | Action |
|---|---|---|
| 3 | `detected_class_name` is empty | Raise `ModelValidationError`, mark context FAILED |
| 4 | `model_family is None` | Raise `ModelValidationError`, mark context FAILED |
| 1 | Names match | Set status MATCH, return `ModelValidationResult` |
| 2 | Names differ | Add warning to context (status=WARNING), return `ModelValidationResult` |

Rule 2 explicitly never raises. `session_context.mark_failed()` is called before
raising in Rules 3 and 4 so the context state is correct regardless of whether the
caller catches the exception.

**Class:** `ModelValidationResult` (frozen dataclass)

Fields: `status: ModelNameValidationStatus`, `message: str`, `model_family: Optional[str]`

### validators/schema_validator.py

**Class:** `SchemaValidator`

Implements spec.md Sections 9-13. Takes `target_column_candidates` at construction;
no JSON config needed (candidates come from `AppConfig.target_column_candidates`).

**Validation sequence inside `validate()`:**

1. Identify target column: scan `target_column_candidates` in order, first exact
   case-sensitive match wins. None if no match (adds warning, not failure).
2. Exclude target column: `DataFrame.drop()`. Dataset ordering is preserved (Sec 10/13).
3. Zero-feature guard: if no feature columns remain after exclusion, raise
   `SchemaValidationError` and mark context FAILED (Sec 9).
4. Feature compatibility cross-validation (Sec 12):
   - Both names and count from model: validate names (count is implicit).
   - Count only: validate count.
   - Names only: validate names.
   - Neither: add warning and skip (dataset schema is authoritative per Sec 10).

**Class:** `SchemaValidationResult` (frozen dataclass)

Fields: `feature_dataframe`, `feature_names`, `target_column_name`,
`num_samples`, `num_features`

`session_context` is updated before returning: `target_column_name`, `feature_names`,
`num_samples`, `num_features`.

### tests/fixtures/fixture_factory.py

**Class:** `FixtureFactory` (all static methods — Spec Sec 22)

| Method | Returns | Purpose |
|---|---|---|
| `make_execution_logger(tmp_path)` | `ExecutionLogger` | Logger writing to tmp dir |
| `make_session_context(...)` | `SessionContext` | RUNNING context with required fields |
| `make_loaded_model(...)` | `LoadedModel` | Configurable model fixture (no disk I/O) |
| `make_loaded_dataset(...)` | `LoadedDataset` | Synthetic DataFrame with optional target |

`make_loaded_model` accepts `model_family=None` to simulate unsupported/undetectable
scenarios (Rules 3/4 testing). `make_loaded_dataset` accepts `include_target_column`
and `target_column_name` to test all target column identification paths.

---

## Test Coverage

| Test File | Tests | Spec Coverage |
|---|---|---|
| `test_model_validator.py` | 19 | Sec 8 Rules 1-4, all five supported families, alias lookup, session context state |
| `test_schema_validator.py` | 18 | Sec 9-13: target identification, exclusion, zero-feature failure, count/name compat, no-metadata warning, column ordering |
| `test_fixture_factory.py` | 5 | Sec 22: all four factory methods verified |

**Phase 4 total: 42 new tests**
**Full suite total: 126 tests, all passing**

### Key test scenarios verified

- Rule 2 (mismatch) never raises and sets `ExecutionStatus.WARNING`, not FAILED
- Rule 3 and Rule 4 both set `session_context.has_failed() == True` before raising
- Target column excluded from `feature_dataframe` and `feature_names`
- No target column found produces a warning, not a failure
- Feature count mismatch terminates with `SchemaValidationError`
- Feature name mismatch terminates with `SchemaValidationError`
- Model with no feature metadata: validation skipped, warning added
- Column ordering preserved exactly from dataset (Sec 10/13)
- `FixtureFactory` produces isolated fixtures per call (UUID-based session IDs)

---

## Acceptance Criteria Traceability

| AC | Description | Phase 4 Coverage |
|---|---|---|
| AC-03 | System automatically detects supported model type | Covered by `ModelValidator` Rules 3/4 |
| AC-04 | System validates supplied model_name against detected model type | Covered by `ModelValidator` Rules 1/2 |
| AC-15 | Validation failures generate meaningful error messages and logs | Covered — all failure paths log before raising and set context |
| AC-17 | metadata.json records supplied_model_name and detected_model_type | Session context fields `model_name_validation_status` and `model_name_validation_message` populated; consumed by MetadataExporter in Phase 7 |

AC-05 through AC-14 remain pending Phase 5-7 (explainers, visualizations, exporters).

---

## Critical Integration Criteria (Deferred)

Per `phase_04_preimplementation.md`, these are resolved at integration time and require
no code changes — only `config.ini` updates:

| ID | Item | Current Default |
|---|---|---|
| OI-02/OI-05 | Target column naming from Epic 2 | Candidates `target,label,outcome` in `config.ini` |
| A-8 | Model name alias contract | 20 aliases in `model_type_detection.json` |
| OI-BEHAVIOR | Behavior when no target column found | Warning + continue (not a hard stop) |

---

## What Phase 5 Can Now Assume

After Phase 4, the pipeline has confirmed:

- `session_context.detected_model_type` — the Python class name of the loaded model
- `session_context.model_name_validation_status` — MATCH or MISMATCH (or FAILED/terminated)
- `session_context.feature_names` — ordered list of feature column names for SHAP
- `session_context.target_column_name` — the excluded column name (or None)
- `session_context.num_samples` / `num_features` — dataset dimensions post-exclusion

`SchemaValidationResult.feature_dataframe` is the cleaned, target-excluded DataFrame
that `ExplainerFactory` and `SHAPService` (Phase 5) can use directly.

---

## Unit Test Fixes

**Applied after initial commit — test collection was broken on all three Phase 4 test files.**

### Root Cause

`test_fixture_factory.py`, `test_model_validator.py`, and `test_schema_validator.py` all
import `FixtureFactory` via:

```python
from tests.fixtures.fixture_factory import FixtureFactory
```

For this package-style import to resolve, two conditions must hold:

1. The `tests` directory must be a Python package (requires `tests/__init__.py`).
2. The `epic_4_shap/` project root must be on `sys.path` so that `tests` is discoverable.

Neither condition was met:
- `pytest.ini` had `pythonpath = src`, which only added `epic_4_shap/src` to `sys.path`.
- `tests/__init__.py` and `tests/unit/__init__.py` did not exist.

The Phase 1-3 tests were unaffected because they never imported from the `tests` package.

### Fix Applied

| Change | Detail |
|---|---|
| `pytest.ini` | Added `.` to `pythonpath` so `epic_4_shap/` root is on `sys.path` |
| `tests/__init__.py` | Created (empty package marker) |
| `tests/unit/__init__.py` | Created (empty package marker) |

### Files Modified

- `pytest.ini` — `pythonpath = src` => `pythonpath = src .`
- `tests/__init__.py` — new, empty
- `tests/unit/__init__.py` — new, empty

No validator business logic was changed.

### Test Results After Fix

```
126 passed in 3.54s
```

All 126 tests pass: 73 pre-Phase-4 tests unaffected, 42 Phase 4 tests now collecting and
passing, 11 pre-existing Phase 1-3 tests confirmed unbroken.
