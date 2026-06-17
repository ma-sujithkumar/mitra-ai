# Phase 03 Implementation: Loaders

## 1. Phase Objective

Implement the loader layer of the Epic 4 SHAP Explainability Module. This phase
delivers two production-ready classes that load and structurally validate the two
pipeline inputs — the trained model artifact (Epic 3 output) and the engineered
dataset CSV (Epic 2 output) — before any validation, SHAP computation, or export
stage runs. It also establishes the shared error hierarchy used by all downstream
stages.

Phase 3 is strictly an I/O and deserialization layer. It does not validate model
names, assess schema compatibility, compute SHAP values, or produce any outputs.

---

## 2. Files Created

| File | Role |
|------|------|
| `src/shap_explainability/errors.py` | Shared structured exception hierarchy for all pipeline stages |
| `src/shap_explainability/loaders/__init__.py` | Package marker for the loaders sub-package |
| `src/shap_explainability/loaders/model_loader.py` | ModelLoader class and LoadedModel dataclass |
| `src/shap_explainability/loaders/dataset_loader.py` | DatasetLoader class and LoadedDataset dataclass |
| `config/model_type_detection.json` | Config-driven class-to-family detection map (9 entries) |
| `tests/fixtures/__init__.py` | Package marker scaffolding tests/fixtures/ for Phase 4+ |
| `tests/unit/test_model_loader.py` | 16 unit tests for ModelLoader |
| `tests/unit/test_dataset_loader.py` | 13 unit tests for DatasetLoader |

---

## 3. Files Modified

None. Phase 3 created new files only. No existing source, config, or test file was
modified.

---

## 4. Classes Implemented

### `SHAPModuleError` (errors.py)
Base exception inheriting from `RuntimeError`. All domain failures in the pipeline
inherit from this class so callers can catch either the base type for blanket
handling or a specific subtype for fine-grained recovery.

### `ModelLoadError(SHAPModuleError)` (errors.py)
Raised by `ModelLoader` when the model artifact cannot be found, read, or
deserialized by either pickle or joblib.

### `DatasetLoadError(SHAPModuleError)` (errors.py)
Raised by `DatasetLoader` when the dataset file is missing, unreadable, empty, or
structurally invalid.

### `LoadedModel` (loaders/model_loader.py)
Frozen dataclass. Typed container returned by `ModelLoader.load()`. Holds the
deserialized model object together with all introspected metadata extracted from the
artifact at load time.

Fields:
- `model_object: Any` — Deserialized estimator (e.g. XGBClassifier instance)
- `detected_class_name: str` — Python class name from `type().__name__`
- `model_family: Optional[str]` — Supported family string from config map; None if unsupported
- `serialization_format: str` — Which path succeeded: "pickle" or "joblib"
- `feature_names_from_model: Optional[tuple[str, ...]]` — Feature names from model metadata if available
- `num_features_from_model: Optional[int]` — Feature count from model metadata if available

### `ModelLoader` (loaders/model_loader.py)
Loads a trained model artifact and detects its concrete model type.

### `LoadedDataset` (loaders/dataset_loader.py)
Frozen dataclass. Typed container returned by `DatasetLoader.load()`. Holds the
full DataFrame and structural metadata derived from the CSV header.

Fields:
- `dataframe: pd.DataFrame` — Full DataFrame with all columns (including any target column)
- `column_names: tuple[str, ...]` — Ordered column names as they appear in the CSV header
- `num_rows: int` — Data row count (excluding header)
- `num_columns: int` — Total column count

### `DatasetLoader` (loaders/dataset_loader.py)
Loads and structurally validates the Epic 2 engineered dataset CSV.

---

## 5. Public Methods

### ModelLoader

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(execution_logger: ExecutionLogger, model_type_config_path: Optional[Path] = None) -> None` | Accepts the session logger and an optional override path to model_type_detection.json. Loads and caches the detection map at construction time. |
| `load` | `(pickle_file_path: str \| Path) -> LoadedModel` | Validates the file exists, deserializes using the extension-driven primary/fallback strategy, detects the class name and model family, extracts feature metadata, and returns a LoadedModel. |

### DatasetLoader

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(execution_logger: ExecutionLogger) -> None` | Accepts the session logger. |
| `load` | `(dataset_path: str \| Path) -> LoadedDataset` | Validates the file exists, reads the CSV with BOM-safe encoding, validates the DataFrame is non-empty and has at least one column, extracts ordered column names, and returns a LoadedDataset. |

All private methods (`_validate_file_exists`, `_deserialize_model`, `_load_with_format`,
`_extract_feature_names`, `_extract_num_features`, `_load_type_detection_config`,
`_default_model_type_config_path`, `_read_csv`, `_validate_dataframe`) are
implementation details and not part of the public contract.

---

## 6. Design Decisions

### 6.1 Config-driven model type detection (no if-else ladder)

Per CLAUDE.md rule 4, branching on model class names using if-else ladders is
prohibited. The class-name-to-family mapping is stored in
`config/model_type_detection.json` and loaded into a `dict` at `ModelLoader`
construction time. Lookups are O(1) dict `.get()` calls. Adding support for a new
model type requires only a JSON edit, not a code change.

### 6.2 Extension-driven deserialization order with fallback

`ModelLoader._deserialize_model()` decides which format to try first based on file
extension:
- `.joblib` extension: joblib is the primary attempt; pickle is the fallback
- All other extensions (`.pkl`, `.pickle`, etc.): pickle is primary; joblib is fallback

`joblib.load()` is a superset of `pickle.load()` and can transparently deserialize
standard pickle streams. This means that a pickle-serialized file with a `.joblib`
extension loads successfully via the joblib primary path — no fallback is triggered.
Both formats are attempted before `ModelLoadError` is raised.

### 6.3 BOM-safe CSV reading

`DatasetLoader._read_csv()` uses `encoding="utf-8-sig"`. This encoding
automatically strips the UTF-8 BOM byte sequence (`\xef\xbb\xbf`) that Microsoft
Excel prepends to exported CSV files. Without this, the first column name would be
prefixed with the BOM character, causing schema comparisons to fail silently.
Pattern adopted from `epic_3/training/data_loader.py`.

### 6.4 Standalone module boundary

`epic_4_shap` has no imports from `epic_3`, `model_library`, or any other epic
package. All reuse from the existing codebase is pattern-based (error hierarchy
shape, BOM encoding, structured exception chaining). This allows the module to be
deployed independently without the rest of the repository.

### 6.5 Feature metadata extraction strategy

`ModelLoader._extract_feature_names()` checks attributes in priority order:

1. `feature_names_in_` — Set by sklearn >= 1.0, XGBoost >= 1.6, and LightGBM's
   sklearn-compatible API when the model is fitted on a pandas DataFrame.
2. `feature_names_` — CatBoost stores feature names here.

LightGBM's `feature_name()` method is intentionally skipped. When the model was
fitted on a numpy array, `feature_name()` returns auto-generated names such as
`Column_0`, `Column_1`, etc. Using those would cause false-positive schema
validation errors in SchemaValidator, where dataset column names would never match.

### 6.6 Star-shaped architecture boundary: DatasetLoader does not identify the target column

`DatasetLoader.load()` returns the full DataFrame including any target column. It
does not attempt to identify, strip, or reason about the target column. That
responsibility belongs exclusively to `SchemaValidator`, which has the context
(AppConfig.target_column_candidates, model feature count) required to make that
determination. This keeps the loader layer ignorant of schema concerns.

### 6.7 Default config path resolved via __file__

`ModelLoader._default_model_type_config_path()` resolves the JSON config path
relative to the module file itself rather than using a hardcoded absolute path or a
CWD-relative path. This makes the module relocatable and testable without
environment setup. The same pattern is used by `ConfigLoader._default_config_file_path()`.

### 6.8 Frozen dataclasses for return types

Both `LoadedModel` and `LoadedDataset` are `@dataclass(frozen=True)`. Frozen
dataclasses are immutable after construction, preventing any downstream stage from
accidentally mutating loader output. They are also hashable by default.

---

## 7. Reused Repository Components

No direct imports from existing packages were introduced. The following patterns
were reviewed and adopted by convention:

| Source file | Pattern reused |
|-------------|----------------|
| `epic_3/training/data_loader.py` | `encoding="utf-8-sig"` for BOM-safe CSV reading |
| `epic_3/training/errors.py` | Error hierarchy shape: base `RuntimeError` subclass with specific subtypes |
| `epic_3/training/errors.py` | Exception chaining with `raise X from original_exc` |
| `epic_3/training/data_loader.py` | File-existence validation before reading |
| `epic_3/training/artifact_writer.py` | Confirmed that Epic 3 saves models as `.pkl` via `pickle.dump()` |
| `epic_4_shap/src/shap_explainability/config_loader.py` | `__file__`-relative default path resolution pattern |
| `epic_4_shap/src/shap_explainability/utils/logger.py` | `ExecutionLogger` as the injected logger type; named event methods |

---

## 8. Assumptions Made

| ID | Assumption | Impact if wrong |
|----|------------|-----------------|
| A-L1 | Epic 3 model artifacts are serialized with `pickle.dump()` as raw estimator objects (confirmed by `artifact_writer.py`) | Pickle-first deserialization order works correctly for `.pkl` files |
| A-L2 | When LightGBM is fitted with a pandas DataFrame, `feature_names_in_` is set via the sklearn-compatible API | LightGBM feature names would be None; SchemaValidator would skip cross-validation |
| A-L3 | CatBoost stores feature names in `feature_names_` (not `feature_names_in_`) | CatBoost feature names would be None |
| A-L4 | The engineered dataset CSV from Epic 2 uses UTF-8 encoding (with or without BOM) | `DatasetLoadError` raised on files with non-UTF-8 encodings such as Latin-1 |
| A-L5 | `DatasetLoader` receives the full dataset path; path resolution (relative vs absolute) is the caller's responsibility | Callers must pass resolved paths or accept resolve() behavior |
| A-L6 | The engineered dataset always has a header row as the first line | A headerless CSV would be loaded with the first data row treated as column names |

---

## 9. Validation Performed

### ModelLoader validation
1. File existence check (`Path.is_file()`) before any deserialization attempt
2. Model type config existence check at construction time
3. Dual-format deserialization: `ModelLoadError` is raised only after both pickle
   and joblib have been attempted and both have failed
4. Feature name and count extraction is defensive (`getattr` with None default,
   length guard); never raises

### DatasetLoader validation
1. File existence check before reading
2. `EmptyDataError` caught and re-raised as `DatasetLoadError` for empty or
   zero-byte files
3. `ParserError` caught for malformed CSV content
4. `UnicodeDecodeError` caught for binary or non-UTF-8 files
5. Post-read check: zero columns raises `DatasetLoadError`
6. Post-read check: zero data rows raises `DatasetLoadError`

---

## 10. Unit Tests Added

### test_model_loader.py (16 tests)

| Test name | What it verifies |
|-----------|-----------------|
| `test_load_random_forest_pickle_detects_random_forest_family` | RF family detection and class name |
| `test_load_logistic_regression_pickle_detects_logistic_regression_family` | LR family detection |
| `test_load_xgboost_pickle_detects_xgboost_family` | XGBoost family detection |
| `test_load_lightgbm_pickle_detects_lightgbm_family` | LightGBM family detection |
| `test_load_catboost_pickle_detects_catboost_family` | CatBoost family detection |
| `test_load_pkl_extension_records_pickle_format` | `.pkl` extension => serialization_format == "pickle" |
| `test_load_joblib_file_records_joblib_format` | `.joblib` extension with joblib content => "joblib" |
| `test_joblib_extension_prefers_joblib_over_pickle_format` | Joblib content + .joblib ext => joblib primary path succeeds |
| `test_pickle_content_with_joblib_extension_loads_successfully` | joblib.load() reads standard pickle; load succeeds with correct model |
| `test_missing_file_raises_model_load_error` | Non-existent path raises ModelLoadError("does not exist") |
| `test_corrupted_file_raises_model_load_error` | Garbage bytes raise ModelLoadError after both formats fail |
| `test_unsupported_model_class_sets_family_to_none` | Unknown class => model_family is None, detected_class_name is set |
| `test_feature_names_extracted_when_fitted_with_dataframe` | feature_names_in_ populated => tuple of column names |
| `test_feature_names_none_when_fitted_with_numpy_array` | numpy fit => feature_names_from_model is None |
| `test_num_features_extracted_after_fit` | n_features_in_ => num_features_from_model == 2 |
| `test_detected_class_name_is_always_set` | class name is always a non-empty string |
| `test_model_object_is_the_fitted_estimator` | model_object is the deserialized sklearn estimator instance |

### test_dataset_loader.py (13 tests)

| Test name | What it verifies |
|-----------|-----------------|
| `test_load_valid_csv_returns_loaded_dataset_instance` | Returns a LoadedDataset instance |
| `test_column_names_preserve_original_ordering` | column_names tuple matches CSV header order |
| `test_num_rows_matches_data_row_count` | num_rows == 5 for a 5-row CSV |
| `test_num_columns_matches_header_column_count` | num_columns == 3 for a 3-column CSV |
| `test_column_names_is_a_tuple_not_a_list` | column_names is a tuple, not a list |
| `test_dataframe_column_ordering_matches_column_names` | dataframe.columns order == column_names |
| `test_dataframe_row_count_matches_num_rows` | len(dataframe) == num_rows |
| `test_dataframe_values_are_preserved` | Cell values are correctly loaded (not coerced or dropped) |
| `test_bom_encoded_csv_column_names_contain_no_bom_character` | BOM stripped; first column name is "col_a" not "﻿col_a" |
| `test_missing_file_raises_dataset_load_error` | Non-existent path raises DatasetLoadError("does not exist") |
| `test_csv_with_header_only_and_no_data_rows_raises_dataset_load_error` | Header-only CSV raises DatasetLoadError("no data rows") |
| `test_completely_empty_file_raises_dataset_load_error` | Zero-byte file raises DatasetLoadError |
| `test_binary_file_content_raises_dataset_load_error` | Non-UTF-8 bytes raise DatasetLoadError |

---

## 11. Test Results

All 84 tests in the epic_4_shap test suite pass.

```
84 passed in 3.60s
```

Breakdown by module:
- Phase 2 foundation tests (55): all passing
- Phase 3 model loader tests (16): all passing
- Phase 3 dataset loader tests (13): all passing

One test was corrected during this phase:
- Originally named `test_pickle_content_with_joblib_extension_falls_back_to_pickle`;
  asserted `serialization_format == "pickle"`.
- Root cause: `joblib.load()` is a superset of `pickle.load()` and can read standard
  pickle streams. No fallback is triggered; joblib succeeds as the primary attempt.
- Fix: renamed test; assertion changed to `model_family == "RandomForest"` and
  `model_object is not None`, verifying correct behavior rather than an incorrect
  internal assumption about which format "won".

---

## 12. Known Limitations

| ID | Limitation | Affects |
|----|------------|---------|
| L-01 | Only UTF-8 and UTF-8-BOM encodings are supported for dataset CSV files. Latin-1, UTF-16, and other encodings will raise DatasetLoadError. | DatasetLoader |
| L-02 | `feature_names_from_model` is None for any model fitted on numpy arrays (except those with `n_features_in_` alone). SchemaValidator must treat None as "feature names unavailable from model" and rely solely on the dataset. | ModelLoader / SchemaValidator interface |
| L-03 | LightGBM's `feature_name()` method is intentionally not called. If a future user supplies a LightGBM model fitted on numpy and expects named features to be surfaced from the model object, they will not be. | ModelLoader |
| L-04 | The model type detection config covers 9 class names for 5 model families. Any class not in the JSON map results in `model_family=None`; ModelValidator must apply Sec 8 Rule 4 for that case. | ModelLoader |
| L-05 | A headerless CSV (no first-row header) is not detected as malformed. The first data row is treated as column names by pandas. DatasetLoader provides no way to detect or reject this. | DatasetLoader |
| L-06 | There is no file-size guard. An extremely large CSV will be loaded fully into memory before any row-count validation occurs. | DatasetLoader |

---

## 13. Readiness for Next Phase

Phase 3 is complete and stable. All loader contracts are finalized and tested.

### Blocked open items before Phase 4 can be finalized

| Item | Blocks |
|------|--------|
| `[OPEN OI-05 / A-3]` Target column naming convention: whether `engineered_dataset.csv` includes a target column and if so, its column name | `SchemaValidator._identify_target_column()` |
| `[OPEN A-5]` LinearExplainer background-data policy for Logistic Regression (full dataset vs. subset vs. KMeans summary) | `ExplainerFactory` Logistic Regression branch |

### Interfaces ready for Phase 4 consumption

| Produced by Phase 3 | Consumed by Phase 4 |
|---------------------|---------------------|
| `LoadedModel.model_object` | ModelValidator (class name check), ExplainerFactory |
| `LoadedModel.detected_class_name` | ModelValidator (Sec 8 Rules 1-4) |
| `LoadedModel.model_family` | ExplainerFactory (explainer selection) |
| `LoadedModel.feature_names_from_model` | SchemaValidator (cross-validation against dataset columns) |
| `LoadedModel.num_features_from_model` | SchemaValidator (feature count cross-check) |
| `LoadedDataset.dataframe` | SchemaValidator, SHAPService |
| `LoadedDataset.column_names` | SchemaValidator (authoritative feature source per Sec 10) |
| `DatasetLoadError`, `ModelLoadError` | Pipeline orchestrator error handling (Sec 20) |

### Phase 4 scope (validators)

- `src/shap_explainability/validators/__init__.py`
- `src/shap_explainability/validators/model_validator.py` — Sec 8 Rules 1-4
- `src/shap_explainability/validators/schema_validator.py` — Sec 9-13 (blocked on OI-05)
