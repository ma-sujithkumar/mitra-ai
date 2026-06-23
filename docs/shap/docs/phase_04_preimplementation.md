# Phase 4 Pre-Implementation Review
**Date:** 2026-06-17
**Branch:** epic4-shap
**Status:** Ready to implement — critical integration criteria deferred to integration phase

---

## Phase 3 Summary

Phases 1-3 are complete with **73 unit tests, all passing**. The foundation, loaders,
configuration, session context, logging, and output management are implemented and solid.
The pipeline can now load model artifacts and datasets but cannot yet validate, compute,
or export anything.

---

## Recommended Next Phase: Phase 4 — Validators

### 1. Phase Objective

Implement business-rule validation on the loaded artifacts before any SHAP computation
begins. This phase enforces the Spec Sec 8-13 requirements: validate the supplied model
name against the detected model type, identify and exclude the target column from
features, and verify schema compatibility between model metadata and dataset columns.

The output is a validated, narrowed `SessionContext` carrying confirmed feature names,
target column, and model name validation status — or a clean failure recorded in session
context.

---

### 2. Files to Create

```
src/shap_explainability/validators/
    __init__.py
    model_validator.py        (Spec Sec 8, Rules 1-4: name-vs-type cross-check)
    schema_validator.py       (Spec Sec 9-13: target column detection, feature schema, compatibility)

tests/unit/
    test_model_validator.py
    test_schema_validator.py

tests/fixtures/
    fixture_factory.py        (Spec Sec 22: synthetic models, datasets, configs for test reuse)
```

`fixture_factory.py` belongs here rather than Phase 9 because Phase 4 tests will be the
first to need realistic fake models and DataFrames. Deferring it creates test duplication
across test files.

---

### 3. Dependencies

**Internal (already implemented):**
- `src/shap_explainability/session_context.py` — `ModelNameValidationStatus` enum,
  `SessionContext.mark_failed()`, `add_warning()`
- `src/shap_explainability/errors.py` — Will need two new exception classes:
  `ModelValidationError`, `SchemaValidationError`
- `src/shap_explainability/loaders/model_loader.py` — `LoadedModel` (input to
  `ModelValidator`)
- `src/shap_explainability/loaders/dataset_loader.py` — `LoadedDataset` (input to
  `SchemaValidator`)
- `src/shap_explainability/config_loader.py` — `AppConfig.target_column_candidates`
  (drives `SchemaValidator`)
- `config/model_type_detection.json` — Reused by `ModelValidator` for name-to-family
  lookup

**External:**
- No new pip packages; validators operate on strings and DataFrames only

**Open items deferred to integration (OI-02 / OI-05):**

These are not blocking Phase 4 implementation. Defaults are baked into `config.ini` and
the implementation is designed to be configurable at integration time when Epic 2 holders
can confirm the contract. See Critical Integration Criteria section at the bottom of this
document.

Phase 4 implementation assumptions (overridable via `config.ini`):
- Target column candidate names: `target,label,outcome` — first match in dataset wins
- If no candidate found: log a warning and continue (not a hard stop)
- Model name matching: case-insensitive lookup against detected family name

---

### 4. Risks

| Risk | Severity | Notes |
|---|---|---|
| OI-02/OI-05 unresolved | High | `SchemaValidator` cannot be finalized without knowing the target column contract from Epic 2. Block Phase 5 if the wrong assumption ships. Coordinate before implementation starts. |
| Model name matching ambiguity (A-8) | Medium | Supplied name `"xgboost"` vs detected `"XGBClassifier"` requires a mapping strategy. The existing `model_type_detection.json` maps class names to families. Recommended approach: normalize supplied name to lowercase, look up the detected class name's family, check if the supplied name is a case-insensitive prefix or alias of the family name. The alias map should live in a new JSON section — not in code. |
| Spec Rule 2 non-blocking mismatch | Medium | Spec Sec 8 Rule 2 says a name mismatch is a warning, not a hard stop. `ModelValidator` must record the mismatch in `SessionContext` and continue — not raise an exception. Easy to get wrong if the team defaults to raise-on-error patterns. |
| Feature-count vs feature-name disagreement | Low-Medium | Model metadata may have `num_features_from_model = 50` but no `feature_names_from_model`. In that case, `SchemaValidator` can only validate count, not names. The validator must handle all four combinations: both present, count only, names only, neither. |
| Test fixture sprawl | Low | Without `fixture_factory.py`, each test file will create its own fake models and DataFrames. Consolidate from the start. |

---

### 5. Validation Strategy

**Unit tests for `ModelValidator` (~15 tests):**
- Exact match: supplied name matches detected family (e.g., `"XGBoost"` vs
  `XGBClassifier`)
- Case-insensitive match passes
- Mismatch records warning in `SessionContext`, does not raise
- Unknown supplied name records warning
- Unknown detected class name records warning
- `ModelNameValidationStatus` enum values are set correctly in all paths

**Unit tests for `SchemaValidator` (~15 tests):**
- Target column identified correctly from candidate list (first match wins)
- Target column excluded from `feature_names` in `SessionContext`
- No target column found: configurable behavior (warning vs error) — must match spec
  decision
- Feature count matches model metadata: passes
- Feature count mismatch: raises `SchemaValidationError`
- Feature names match model metadata: passes
- Feature names mismatch: raises `SchemaValidationError`
- Model has no feature metadata (`feature_names_from_model = None`,
  `num_features_from_model = None`): schema validation skipped with a logged warning,
  not an error

**Unit tests for `fixture_factory.py` (~5 tests):**
- Confirms factory produces valid `LoadedModel` and `LoadedDataset` instances
- Confirms synthetic datasets include/exclude target column as requested

**Acceptance criteria gate before Phase 5:**

After Phase 4, the pipeline should be able to: load model + dataset, validate names and
schema, and populate `SessionContext.feature_names`, `target_column_name`, and
`model_name_validation_status` — or cleanly record a failure. Run the 73 existing tests
plus ~35 new ones; all must pass before Phase 5 begins.

---

## Implementation Checklist

- [ ] Add `ModelValidationError` and `SchemaValidationError` to `errors.py`
- [ ] Implement `validators/model_validator.py`
- [ ] Implement `validators/schema_validator.py`
- [ ] Implement `tests/fixtures/fixture_factory.py`
- [ ] Write `tests/unit/test_model_validator.py` (~15 tests)
- [ ] Write `tests/unit/test_schema_validator.py` (~15 tests)
- [ ] Write `tests/unit/test_fixture_factory.py` (~5 tests)
- [ ] All 108 tests passing (73 existing + ~35 new)
- [ ] Update `SessionContext` fields populated by validators confirmed in tests

---

## Critical Integration Criteria

These items are not blocking Phase 4 implementation. They must be resolved during the
integration phase when information is available from other epic holders. The implementation
uses configurable defaults that can be updated in `config.ini` without code changes.

| ID | Criteria | Owner | Current Default | Impact if Changed |
|---|---|---|---|---|
| OI-02 | Confirm whether Epic 2 engineered dataset always includes a target column | Epic 2 holder | Candidate scan: `target,label,outcome` in `config.ini` | If Epic 2 guarantees a fixed name, simplify `SchemaValidator` to exact match instead of candidate scan |
| OI-05 | Confirm canonical target column naming convention from Epic 2 | Epic 2 holder | First candidate found in dataset wins | If multiple candidates could exist, update matching behavior in `SchemaValidator._identify_target_column()` |
| A-8 | Confirm model name alias contract (e.g., does `"xgboost"` match `XGBClassifier`?) | Epic 3 / integration spec | Case-insensitive family name prefix match via `model_type_detection.json` | If stricter or looser matching is required, update `model_type_detection.json` alias entries |
| OI-BEHAVIOR | Confirm behavior when no target column candidate is found in dataset | Product / spec | Log warning, continue without excluding target column | If hard stop is required, update `config.ini [target_column] ON_NOT_FOUND=error` |
