# Epic 4 - SHAP Explainability Module: Architecture

Source inputs: `epic_4_shap/spec.md` (v1.0), `epic_4_shap/docs/architecture_review.md`
Status: Draft architecture for implementation planning. Items marked `[OPEN]` trace back to blocking ambiguities/open items raised in `architecture_review.md` and must be confirmed before the corresponding piece is implemented.

---

## 1. Folder Structure

The architecture review (Section "Ambiguities", item A-2) found that spec.md Section 21 conflates the *source package layout* with the *output artifact layout* under one tree. This architecture splits them explicitly.

### 1.1 Source / repository layout

```
epic_4_shap/
├── CLAUDE.md
├── README.md
├── spec.md
├── requirements.txt
├── config/
│   └── config.ini                  # single config file for this module (CFG-01..04)
├── docs/
│   ├── architecture_review.md
│   └── architecture.md
├── src/
│   └── shap_explainability/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── session_context.py
│       ├── config_loader.py
│       ├── loaders/
│       │   ├── __init__.py
│       │   ├── model_loader.py
│       │   └── dataset_loader.py
│       ├── validators/
│       │   ├── __init__.py
│       │   ├── model_validator.py
│       │   └── schema_validator.py
│       ├── explainers/
│       │   ├── __init__.py
│       │   ├── explainer_factory.py
│       │   └── shap_service.py
│       ├── visualizations/
│       │   ├── __init__.py
│       │   └── plot_generator.py
│       ├── exporters/
│       │   ├── __init__.py
│       │   ├── csv_exporter.py
│       │   └── metadata_exporter.py
│       └── utils/
│           ├── __init__.py
│           ├── logger.py
│           └── output_manager.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── fixture_factory.py      # Sec 22 standalone test utility
│   ├── unit/
│   │   ├── test_model_loader.py
│   │   ├── test_dataset_loader.py
│   │   ├── test_model_validator.py
│   │   ├── test_schema_validator.py
│   │   ├── test_explainer_factory.py
│   │   ├── test_shap_service.py
│   │   ├── test_plot_generator.py
│   │   ├── test_csv_exporter.py
│   │   ├── test_metadata_exporter.py
│   │   ├── test_logger.py
│   │   └── test_output_manager.py
│   └── integration/
│       ├── test_pipeline_happy_path.py
│       ├── test_pipeline_failures.py
│       ├── test_pipeline_model_name_mismatch.py
│       └── test_pipeline_reproducibility.py
└── outputs/                         # gitignored; runtime-generated only
```

`outputs/` is the local default sink only until `[OPEN OI-01]` (configured output root) is finalized; the real output root is resolved through `config.ini` / `OutputManager`, not hardcoded to this folder.

### 1.2 Output artifact layout (runtime-generated, per session)

This is the structure spec.md Section 21 was actually describing under the literal heading "Output Directory Structure," distinct from 1.1 above:

```
<configured_output_root>/
└── <session_id>/
    ├── plots/
    │   ├── summary_plot.png
    │   ├── feature_importance_bar.png
    │   └── beeswarm_plot.png
    │
    ├── csv/
    │   ├── global_feature_importance.csv
    │   └── feature_shap_mapping.csv
    │
    ├── metadata/
    │   └── metadata.json
    │
    └── logs/
        └── execution.log
```

Subfolder names (`plots/`, `csv/`) are an architectural default, not yet spec-confirmed `[OPEN A-2]`; `OutputManager` is the single place this convention would change if revised.

---

## 2. Package Structure

- Top-level installable/importable package: `shap_explainability` (under `epic_4_shap/src/`), matching the module's standalone nature stated in spec.md Section 1 ("standalone SHAP Explainability Module").
- Sub-packages mirror the five workflow concerns from spec.md Section 15 plus two cross-cutting concerns:

| Sub-package | Concern | Spec Traceability |
|---|---|---|
| `loaders` | Acquire raw inputs (model artifact, dataset) | Sec 4, Sec 7 steps 1-2 |
| `validators` | Enforce Sections 6-13 rules before computation | Sec 6-13 |
| `explainers` | Explainer selection + SHAP computation | Sec 14-15 (steps 7-9) |
| `visualizations` | Plot generation | Sec 16 |
| `exporters` | CSV + metadata persistence | Sec 17-18 |
| `utils` | Logging and output-path resolution (cross-cutting, used by every other sub-package) | Sec 19, Sec 5 |
| (root) `pipeline.py`, `session_context.py`, `config_loader.py` | Orchestration, shared state, configuration - not isolated into a sub-package since each is a single cohesive module, not a family of related modules | Sec 15 (overall), Sec 28 |

- No sub-package imports another sub-package's internals directly except through the orchestrator (`pipeline.py`) and the shared `SessionContext` object - this keeps the dependency graph a strict star/hub shape (orchestrator at the center), which directly supports the Maintainability NFR (Sec 24: "modular architecture with clear separation").
- `config_loader.py` reads `config/config.ini` once at pipeline start and produces a single immutable configuration object passed into `OutputManager` and `ExecutionLogger` - no module re-reads the file independently, avoiding config drift mid-run.
- Per project-wide convention (single config.ini, sections per concern), `config.ini` holds one section per CFG item group, e.g. `[output]`, `[logging]`, `[target_column]`, `[plot]`, rather than four flat top-level keys, so it can grow without restructuring the file.

---

## 3. Module Responsibilities

| Module | Responsibility | Inputs | Outputs |
|---|---|---|---|
| `pipeline.py` | Sequence the 13-step workflow (Sec 15); own the stop-on-failure control flow (Sec 7, Sec 20); guarantee metadata/log flush on both success and failure paths | Input JSON contract (Sec 4) | Pipeline run result (success/failure + populated session output folder) |
| `session_context.py` | Define the typed, mutable-by-stage data carrier threaded through the pipeline (session_id, paths, loaded artifacts, detected type, feature list, SHAP results, metadata-in-progress) | - | - |
| `config_loader.py` | Parse `config.ini`, expose typed accessors for CFG-01..04 | `config/config.ini` | Immutable config object |
| `loaders/model_loader.py` | Validate `pickle_file_path` exists, load via joblib/pickle (A11), return model handle + detected class | `pickle_file_path` | Loaded model object, detected type string |
| `loaders/dataset_loader.py` | Validate `engineered_dataset_path` exists/readable, load CSV, basic emptiness check | `engineered_dataset_path` | Loaded dataframe |
| `validators/model_validator.py` | Apply Sec 8 Rules 1-4 (name-vs-detected-type matching, unsupported/undetectable termination) | Supplied `model_name`, detected type | Validation result (status + message), never raises for Rule 2 (warning-only) |
| `validators/schema_validator.py` | Identify/exclude target column (Sec 11), validate feature count/columns/order against model (Sec 12), enforce no-hardcoded-feature-name invariant (Sec 13) | Dataframe, model metadata (if any) | Cleaned feature-only dataframe, resolved feature name list, validation result |
| `explainers/explainer_factory.py` | Map detected model type to the correct SHAP explainer class via a config-driven lookup table (Sec 14), not nested conditionals | Detected model type | Constructed explainer instance |
| `explainers/shap_service.py` | Run the explainer to produce raw SHAP values; compute aggregated global importance and the long-form per-record/per-feature table | Explainer, cleaned dataframe | SHAP values matrix, importance table, long-form mapping table |
| `visualizations/plot_generator.py` | Render summary, bar, and beeswarm plots (Sec 16) to PNG | SHAP values, feature names, output paths | Three PNG files |
| `exporters/csv_exporter.py` | Write `global_feature_importance.csv` and `feature_shap_mapping.csv` with required columns (Sec 17) | Importance table, mapping table, output paths | Two CSV files |
| `exporters/metadata_exporter.py` | Assemble and write `metadata.json` (Sec 18), merging results from every prior stage including warning/discrepancy details | SessionContext | `metadata.json` |
| `utils/logger.py` | Session-scoped execution logger; one declarative method per Sec 19 event category; respects CFG-02 level | Config, session_id | `execution.log`, in-process log stream |
| `utils/output_manager.py` | Resolve output root (CFG-01), create session + artifact subfolders (Sec 5, Section 1.2 above), expose path-getter methods to every exporter/visualization module so no path is hardcoded elsewhere | Config, session_id | Filesystem folders, path strings |
| `tests/fixtures/fixture_factory.py` | Synthesize tiny datasets, train one instance per Phase 1 model type, serialize, and emit a matching Sec 4 JSON payload, independent of real Epic 2/3 artifacts (Sec 22) | - | Fixture model files, fixture datasets, fixture JSON payloads |

---

## 4. Class Responsibilities

One primary class per module above (names indicative, structure only - no implementation):

- **`SHAPPipeline`** (`pipeline.py`) - single public entry point accepting the Sec 4 JSON contract; owns one instance of every stage class below; implements stop-on-validation-failure control flow; ensures partial metadata/log artifacts are still written on failure (AC-15).
- **`SessionContext`** (`session_context.py`) - pure data holder, no behavior; carries session_id, resolved paths, loaded model/dataset handles, detected model type, resolved feature list, SHAP outputs, and the accumulating metadata dict across stages without global state.
- **`ConfigLoader`** (`config_loader.py`) - reads `config.ini` once; exposes typed getters (`output_root()`, `log_level()`, `target_column_strategy()`, `plot_format()`); no other class touches the raw config file.
- **`ModelLoader`** (`model_loader.py`) - validates file existence, loads artifact, detects concrete model class; has no knowledge of `model_name` validation (kept separate per single-responsibility split agreed in the review).
- **`DatasetLoader`** (`dataset_loader.py`) - validates path/readability, loads dataframe, checks non-empty/feature-count > 0; has no knowledge of target-column or model-compatibility logic.
- **`ModelValidator`** (`model_validator.py`) - encodes Sec 8 Rules 1-4 via a supported-model-type lookup map (not if-else ladders, per project convention); returns a structured result distinguishing terminating failures (Rules 3-4) from non-terminating warnings (Rule 2).
- **`SchemaValidator`** (`schema_validator.py`) - identifies/excludes the target column using the configured strategy; validates feature presence/count/order against model metadata when introspectable; returns the cleaned feature dataframe consumed by every later stage.
- **`ExplainerFactory`** (`explainer_factory.py`) - single method resolving detected model type to a SHAP explainer instance via the Sec 14 mapping table; encapsulates explainer-specific construction (e.g., background-data needs for `LinearExplainer`, pending `[OPEN A-5]`).
- **`SHAPService`** (`shap_service.py`) - computes raw SHAP values from the selected explainer; computes aggregated global importance (Sec 17.1) and the long-form per-record/per-feature mapping (Sec 17.2); kept distinct from `ExplainerFactory` so selection and computation remain independently testable.
- **`PlotGenerator`** (`plot_generator.py`) - one method per plot type (summary/bar/beeswarm); takes SHAP values + feature names + an injected output path from `OutputManager`; has no path-resolution logic of its own.
- **`CSVExporter`** (`csv_exporter.py`) - two methods, one per CSV file (Sec 17.1/17.2); pure formatting + I/O over data already computed by `SHAPService`.
- **`MetadataExporter`** (`metadata_exporter.py`) - single assemble-and-write method; merges results from `ModelLoader`, `ModelValidator`, `SchemaValidator`, `ExplainerFactory`, and `SHAPService` into the Sec 18 schema, including the warning/discrepancy fields `[OPEN A-4]`.
- **`ExecutionLogger`** (`logger.py`) - wraps the standard `logging` module, scoped to one log file per session; exposes named methods per Sec 19 event category so call sites stay declarative.
- **`OutputManager`** (`output_manager.py`) - resolves the configured output root, creates the session folder and artifact subfolders (Section 1.2), exposes `plot_path()`, `csv_path()`, `metadata_path()`, `log_path()` consumed by every exporter/visualization class.
- **`FixtureFactory`** (`fixture_factory.py`, test-only) - builds synthetic datasets/models/payloads for each Phase 1 model type, enabling Sec 22's "executable independently of upstream epics" requirement.

---

## 5. Data Flow

End-to-end flow for one pipeline invocation, following spec.md Section 15's 13 steps, expressed against the classes in Section 4:

1. `SHAPPipeline.run(input_json)` receives and structurally validates the Sec 4 contract (session_id, model_name, pickle_file_path, engineered_dataset_path); creates an empty `SessionContext`.
2. `ConfigLoader` resolves output root, log level, target column strategy, plot format from `config.ini`; `OutputManager` creates `<output_root>/<session_id>/` and its subfolders; `ExecutionLogger` opens the session log file. Logged: execution start (Sec 19).
3. `ModelLoader.load(pickle_file_path)` -> model handle + detected model type, stored on `SessionContext`. Logged: model loading, model type detection.
4. `DatasetLoader.load(engineered_dataset_path)` -> raw dataframe, stored on `SessionContext`. Logged: dataset validation.
5. `ModelValidator.validate(model_name, detected_type)` -> validation result; on Rule 3/4 (undetectable/unsupported) the pipeline stops here (see Section 6); on Rule 2 (mismatch) a warning is recorded on `SessionContext` and flow continues. Logged: model-name validation.
6. `SchemaValidator.clean(raw_dataframe, model_handle, target_column_strategy)` -> identifies/excludes target column, validates feature count/order/names against model metadata where available, returns cleaned feature-only dataframe + resolved feature name list, both stored on `SessionContext`. Logged: schema validation.
7. `ExplainerFactory.create(detected_type)` -> explainer instance, stored on `SessionContext`. Logged: explainer selection.
8. `SHAPService.compute(explainer, cleaned_dataframe)` -> SHAP value matrix, stored on `SessionContext`. Logged: SHAP generation.
9. `SHAPService.aggregate(shap_values, feature_names)` -> global importance table and long-form per-record/per-feature mapping table, both stored on `SessionContext`.
10. `PlotGenerator.render_all(shap_values, feature_names, output_manager)` -> three PNG files written under `plots/`. Logged: plot generation.
11. `CSVExporter.export_all(importance_table, mapping_table, output_manager)` -> two CSV files written under `csv/`. Logged: CSV export.
12. `MetadataExporter.export(session_context)` -> `metadata.json` written, merging every stage's recorded facts and any warning from step 5. Logged: metadata generation.
13. `SHAPPipeline` logs execution completion and returns a success result referencing the session output folder.

On any terminating failure (steps 3-6, or an unexpected exception in 7-12), the pipeline jumps directly to a failure path that still invokes `ExecutionLogger` (execution failure event) and `MetadataExporter` with whatever partial `SessionContext` state exists, then returns/raises a failure result - this guarantees AC-15 (meaningful errors + logs) even on early termination.

Data flows strictly forward through `SessionContext`; no later stage mutates an earlier stage's already-recorded fields, only appends its own - this keeps the metadata trail (Sec 18, Traceability NFR in Sec 24) accurate regardless of where the pipeline stopped.

---

## 6. Validation Flow

Nine ordered layers, matching the strategy proposed in `architecture_review.md` Section 7, each producing a structured pass/fail/warn result rather than ad hoc exceptions, so `MetadataExporter` can render every outcome uniformly:

```
[1] Input contract validation        (Sec 4)            -> TERMINATE on missing/malformed field
        |
[2] Path existence validation        (Sec 4.3/4.4, 9)   -> TERMINATE on missing file
        |
[3] Model load validation            (Sec 7 step 2)      -> TERMINATE on deserialization failure
        |
[4] Model type detection & support   (Sec 8 Rule 3/4)    -> TERMINATE if undetectable or unsupported
        |
[5] Model-name consistency check     (Sec 8 Rule 1/2)    -> WARN ONLY, never terminates; continues
        |
[6] Dataset structural validation    (Sec 9)              -> TERMINATE if empty / zero features
        |
[7] Target column resolution         (Sec 11)             -> must run before [8]; feeds cleaned dataframe forward
        |
[8] Schema/compatibility validation  (Sec 10, 12)          -> TERMINATE on feature count/name mismatch
        |
[9] Dynamic-feature-name invariant   (Sec 13)              -> enforced as a code/test-time invariant, not a runtime branch
        |
   PROCEED to SHAP computation (Section 5, steps 7-13)
```

Rules of the flow:
- Layers 1-4 and 6-8 are **terminating**: any failure stops the pipeline immediately, logs the failure, and routes to the failure path described in Section 5.
- Layer 5 is the **only non-terminating** validation rule in the entire spec; it must be implemented so a mismatch cannot accidentally escalate to a terminating error - this is called out explicitly because Sec 8's prose otherwise reads similarly to the terminating rules around it.
- Layer 7 must execute before layer 8, since schema/compatibility validation needs the *feature-only* dataframe (target column already excluded), not the raw loaded dataframe - this ordering is implied but not stated outright in spec.md and is made explicit here.
- Layer 9 (no hardcoded feature names) is structurally guaranteed by the module design (Section 2/3 - all feature names flow from `SchemaValidator`'s resolved list, never literals) and is verified by tests rather than a runtime check.
- Every terminating failure still produces a `metadata.json` with whatever fields were known (Section 5's failure path) and a log entry for "execution failures" (Sec 19), satisfying AC-15 even though the run did not complete.

---

## 7. Testing Strategy

Directly reflecting Section 1's `tests/` layout, mirroring `architecture_review.md` Section 8, with explicit acceptance-criteria traceability:

### 7.1 Unit tests (`tests/unit/`)
One file per class in Section 4, exercised against fixtures from `FixtureFactory` rather than real Epic 2/3 artifacts:
- `model_loader`, `dataset_loader` - valid/missing/corrupted/empty input variants.
- `model_validator` - all four Sec 8 rules independently, confirming Rule 2 never raises.
- `schema_validator` - target column present/absent/multiple-candidates `[OPEN A-3]`, feature count/name mismatches, graceful skip when model lacks feature metadata.
- `explainer_factory` - correct explainer per Phase 1 model type (Sec 6); unsupported type behavior.
- `shap_service` - SHAP value matrix shape correctness, deterministic/reproducible aggregation given fixed inputs.
- `plot_generator` - each of the three plot types produces a non-empty file at the expected path.
- `csv_exporter` - exact required columns (Sec 17.1/17.2) and exactly one row per record-feature pair (AC-16).
- `metadata_exporter` - required fields present (AC-17); warning-path schema populated once `[OPEN A-4]` is resolved.
- `logger` - all Sec 19 event categories logged at least once; CFG-02 level respected.
- `output_manager` - folder/subfolder creation when missing; no hardcoded path components.

### 7.2 Integration tests (`tests/integration/`)
Full-pipeline runs against `FixtureFactory`-generated artifacts:
- Happy path per Phase 1 model type -> asserts every artifact in AC-07 through AC-13 exists and is non-trivial.
- Failure paths (missing model, missing dataset, unsupported model type, empty dataset, schema mismatch) -> asserts graceful termination, logged error, and a `metadata.json` reflecting the failure (AC-15).
- Model-name mismatch -> asserts execution *continues* and the warning is both logged and present in `metadata.json` (Sec 8 Rule 2, Sec 18).
- Reproducibility -> two identical runs produce value-identical CSV/metadata output aside from timestamp fields (Sec 24 NFR).
- Session isolation -> two distinct `session_id` runs do not collide in the output tree.

### 7.3 Standalone test utility (`tests/fixtures/fixture_factory.py`)
Directly satisfies Sec 22: synthesizes a tiny labeled dataset, trains one instance of each Phase 1 model type, serializes it, and emits a matching Sec 4 JSON payload - usable for manual/ad hoc runs before Epic 2/3 integration is ready, and as the shared fixture source for 7.1/7.2.

### 7.4 Acceptance criteria traceability
A lightweight matrix mapping AC-01 through AC-20 (spec.md Section 27) to the specific unit/integration test(s) covering each should be maintained alongside the test suite once tests exist, so acceptance coverage is demonstrable rather than assumed - recommended as a `tests/AC_TRACEABILITY.md` once test files are in place.

---

## Open Items Carried Forward

These remain unresolved from `architecture_review.md` and constrain the items below until confirmed:

- `[OPEN OI-05 / A-3]` - target column presence/naming convention -> blocks final `SchemaValidator` logic.
- `[OPEN OI-01]` - output root default/config mechanism -> blocks final `OutputManager` default behavior.
- `[OPEN A-2]` - output artifact subfolder naming (`plots/`, `csv/` proposed here as a default, not yet spec-confirmed).
- `[OPEN A-5]` - `LinearExplainer` background-data policy -> blocks final `ExplainerFactory` construction logic for Logistic Regression.
