# Epic 4 - SHAP Explainability Module: Architecture Review

Source reviewed: `epic_4_shap/spec.md` (v1.0)
Reviewer: Claude Code
Date: 2026-06-17

---

## 1. Ambiguities

| # | Spec Reference | Ambiguity |
|---|---|---|
| A-1 | Sec 5, OI-01 | "Configured output directory" has no defined source (config.ini key name, env var, or CLI arg) and no confirmed default. |
| A-2 | Sec 21 | Output directory tree mixes the *module source layout* (`loaders/`, `validators/`, `explainers/`, ...) with the *output directory structure* under a literal placeholder `<palceholder name>/` (typo for "placeholder"). It is unclear whether Sec 21 describes the **source code package layout** or the **per-session output folder layout** - the heading says "Output Directory Structure" but the contents are Python module names, not output artifact folders (plots/, csv/, metadata/, logs/). This needs to be split into two separate structures. |
| A-3 | Sec 11, OI-02, OI-05, CFG-03 | Target column identification strategy is "configurable" but the actual strategy (exact name match, regex, list of candidate names, schema flag from Epic 2) is never specified. CFG-03 gives example values (`target`, `label`, `outcome`) but does not say whether matching is case-sensitive, whether multiple candidates can collide, or what happens if more than one candidate column is present in the same dataset. |
| A-4 | Sec 8 Rule 2 | "record discrepancy in metadata" - the exact metadata schema field(s) for a mismatch are partially shown in Sec 18 (`validation_status: warning`) but no field carries *both* the supplied and detected names simultaneously alongside a warning status in the same example. The two metadata.json examples in Sec 18 appear to be alternates rather than a single merged schema, and it's ambiguous whether `validation_status: warning` replaces or supplements `"success"`. |
| A-5 | Sec 14 | "Logistic Regression -> LinearExplainer" assumes a linear, non-probabilistic decision function. SHAP's `LinearExplainer` requires an *independent feature masker* or background data and a `feature_perturbation` setting (`interventional` vs `correlation_dependent`). The spec does not say which mode, nor where background/reference data comes from (full dataset? sample? training subset?). |
| A-6 | Sec 15 step 8 | "Generate SHAP values" does not specify whether this happens on the **full dataset** or a **sample** (relevant for large datasets - see Risk R-3). Sec 17.2 implies "every record" is exported, which suggests full-dataset SHAP computation, but this conflicts with practical runtime/memory limits for TreeExplainer on large N or KernelExplainer-class models. |
| A-7 | Sec 6 / Sec 14 | Multiclass and regression handling is explicitly listed as an open item (OI-06), yet Sec 16-17 visualization/CSV specs assume a single SHAP value per (record, feature) - this only holds for binary classification or regression. For multiclass, SHAP returns a value per class, and the output schemas (CSV columns, plots) are undefined for that case. |
| A-8 | Sec 4.2 / Sec 8 | "model_name" validation compares supplied name to "detected model type" - but detected type will be a Python class name (e.g., `XGBClassifier`), while supplied name is a logical name (e.g., `"xgboost"`). The spec never defines the matching/normalization rule (exact string equality vs. a lookup table mapping logical names to one-or-more class names). The Sec 18 example silently assumes equivalence between `"xgboost"` and `"XGBClassifier"` without stating the comparison mechanism. |
| A-9 | Sec 9 / Sec 12 | "dataset is readable" and "schema is usable by the loaded model" are validation rules without concrete pass/fail criteria (e.g., what constitutes "usable" - dtype compatibility? NaN tolerance? column order?). |
| A-10 | Sec 19 | Logging destination/format is unspecified: file name (`execution.log` appears only in AC-13, not Sec 19), rotation policy, structured vs plain text, and whether logs are per-session or global. |
| A-11 | Sec 28 CFG-01..04 | Configuration mechanism (config.ini vs config.yaml vs env vars) is not stated. The repository already has a precedent (`model_library/config/config.ini`) per project-wide CLAUDE.md conventions, so this should be reconciled, but spec.md is silent on it. |

---

## 2. Risks

| # | Risk | Impact | Notes |
|---|---|---|---|
| R-1 | **Per-record, per-feature SHAP export (Sec 17.2) scales as O(records x features).** | High | For 1M records x 50 features, the mapping CSV is 50M rows. This is a real disk/memory/runtime risk not addressed anywhere in the spec (no sampling, batching, or row-count ceiling mentioned). |
| R-2 | **KernelExplainer / non-tree, non-linear model growth path.** | Medium | Phase 1 models all map cleanly to TreeExplainer/LinearExplainer, but Sec 24 "Extensibility" promises future model types without redesign. If a future model needs `KernelExplainer` or `Permutation` explainer, those require background datasets and are vastly slower - the current explainer-factory contract (Sec 14) has no extensibility hook for explainer-specific configuration (e.g., background sample size, link function). |
| R-3 | **Runtime/memory blow-up for large datasets with TreeExplainer interaction effects disabled by default but SHAP value computation on full dataset.** | Medium-High | No batching/chunking strategy specified; large datasets could exceed memory when SHAP value matrices are held entirely in memory before CSV export. |
| R-4 | **Silent correctness risk from model/dataset feature-order mismatch.** | High | Sec 11.4 says "feature ordering shall remain consistent with model expectations," but for sklearn-style models without `feature_names_in_`, there's no way to verify order except by trusting Epic 2's column order (Sec 10 explicitly says the model shall not be relied upon for feature names). If Epic 2's dataset column order ever silently changes, SHAP values will be computed against the wrong feature semantics with no error raised - this is a correctness risk, not just an availability one. |
| R-5 | **model.pkl trust boundary.** | Medium | Loading arbitrary pickle files (Sec 4.3, A11) without sandboxing is an arbitrary-code-execution risk if `pickle_file_path` ever originates from an untrusted source. The spec assumes internal trusted artifacts (A3), but this should be explicitly called out as an accepted risk rather than left implicit. |
| R-6 | **Validation/termination ordering creates partial-output risk.** | Low-Medium | Sec 7 says "execution shall stop on validation failure," but if failure happens after some artifacts (e.g., output folder, partial logs) are already written, downstream consumers (e.g., a UI polling the session folder) could see an incomplete/inconsistent output set without a clear "do not consume yet" signal (no explicit `status.json` / lock-file / `IN_PROGRESS` marker is specified). |
| R-7 | **CatBoost categorical-feature SHAP nuance.** | Low-Medium | CatBoost's TreeExplainer path through SHAP has known version-sensitivity (especially with native categorical feature handling) - A8 acknowledges SHAP/model-version coupling generally but doesn't flag CatBoost specifically as the highest-risk integration of the five Phase 1 models. |
| R-8 | **Concurrent sessions / output overwrite.** | Low | Sec 5 organizes by `session_id` but doesn't state whether `session_id` collisions (re-running the same session) overwrite, append, or error. No idempotency rule given. |

---

## 3. Missing Requirements

1. **Output folder layout for artifacts** (distinct from source code layout) - which subfolders (if any) hold plots vs CSVs vs metadata vs logs under `<output_root>/<session_id>/`. Sec 21 does not actually specify this (see A-2).
2. **Classification task type / label handling** - binary vs multiclass vs regression output schema differences (OI-06 flags this as open, but it is in fact a hard *blocking* requirement for Sec 16/17 to be implementable, not a nice-to-have).
3. **Target column confirmation from Epic 2** (OI-05) - blocking for Sec 11 implementation; "configurable" identification strategy needs a concrete default algorithm even if Epic 2 confirmation is pending.
4. **Explainer background/reference data policy** for `LinearExplainer` (and any future non-tree explainer) - what data is used as the background distribution.
5. **Sampling/row-limit policy** for SHAP value generation and CSV export on large datasets (ties to R-1/R-3) - e.g., a `CFG` parameter for max rows processed, with explicit truncation behavior and metadata flagging.
6. **Explicit `execution.log` filename/location requirement** in the body text (Sec 19) - currently only appears in AC-13, creating a gap between narrative spec and acceptance criteria.
7. **Schema for "discrepancy" metadata field** (Sec 8 Rule 2) - exact JSON key names for storing both supplied and detected model names together with the warning.
8. **Idempotency / re-run behavior** for a previously used `session_id` (overwrite vs error vs version suffix).
9. **In-progress vs complete status signal** for consumers polling the output directory (e.g., a `status: "running"|"completed"|"failed"` marker file), referenced indirectly by R-6.
10. **Numeric precision / rounding rule** for CSV outputs (Sec 17.1/17.2 examples show 3 decimal places, but no rule is stated - is this a formatting requirement or just illustrative?).
11. **Multiclass/regression-aware plot and CSV schema variants** - currently only single-column SHAP value layouts are defined.
12. **Resource limits / timeouts** - no non-functional requirement bounds maximum execution time or memory ceiling, despite R-1/R-3.
13. **Versioning of output schema** - no `schema_version` field in `metadata.json`, which would aid long-term extensibility (Sec 24) when CSV/metadata formats evolve.
14. **CLI / programmatic entry point contract** - Sec 22 describes a *test* utility invoking "the SHAP pipeline," implying a callable pipeline entry point, but the production entry-point contract (how Epic 3 actually invokes Epic 4: CLI args, Python function call, message queue) is never defined.

---

## 4. Proposed Architecture

The spec's Sec 21 structure (loaders / validators / explainers / exporters / visualizations / utils) is sound and maps directly onto the explicit workflow steps in Sec 15. The recommended architecture is a **single-direction pipeline orchestrator** invoking five cohesive stages, each independently testable per Sec 22, with a typed contract object passed between stages (consistent with the typed-contract pattern already used in `epic_3/training_orchestrator/contracts.py`).

```
                 +-------------------------------------------------+
                 |              SHAPPipelineOrchestrator           |
                 |  (epic_4_shap/src/pipeline.py)                  |
                 +-------------------------------------------------+
                          |        |        |        |       |
                          v        v        v        v       v
                    +--------+ +--------+ +-------+ +-----+ +--------+
                    |Loaders | |Validate| |Explain| |Export| |Visualize|
                    +--------+ +--------+ +-------+ +-----+ +--------+
                          \________________|________________/
                                           v
                                  SessionContext (typed)
                                  + ExecutionLogger
```

Stage responsibilities (mirrors Sec 15 step numbering):
- **Loaders** -> Sec 15 steps 1-2 (engineered dataset, model artifact)
- **Validators** -> Sec 15 steps 3-6 (model type detection, model-name check, dataset checks, schema compatibility)
- **Explainers** -> Sec 15 steps 7-9 (explainer selection, SHAP value generation, importance metrics)
- **Visualizations** -> Sec 15 step 10
- **Exporters** -> Sec 15 steps 11-13 (CSV export, metadata export; log export handled by a cross-cutting logger rather than an "exporter" since logging spans every stage, not just the end)

A `SessionContext` (typed data object, not a code class per se at this stage - just an architectural unit) flows through every stage carrying: session_id, resolved paths, loaded dataset/model handles, detected model type, selected explainer name, computed SHAP values, and the accumulating metadata dict. This avoids re-deriving the same facts (e.g., feature list) in multiple stages, directly supporting Sec 13 (dynamic feature handling) and Sec 24 (maintainability via separation of concerns).

Configuration resolution (output root, log level, target column strategy, plot format) is centralized in one config-loader component reused from the existing `model_library/core/config_loader.py` + `model_library/config/config.ini` pattern already present in this repository, per the project-wide convention of a single global config.ini. This should be evaluated for direct reuse before writing a new loader (per CLAUDE.md reuse-first guidance) - **this is a concrete reuse candidate that should be checked against `dev`/`model_library` before implementation starts.**

---

## 5. Module Breakdown

Mapping Sec 21's package layout to the workflow in Sec 15, with one clarified addition (a top-level pipeline/orchestrator module, which Sec 21 omits):

| Module | Responsibility | Spec Traceability |
|---|---|---|
| `pipeline.py` (new, not in Sec 21 but required) | Orchestrates the 13-step workflow, owns SessionContext lifecycle, top-level error boundary | Sec 15 |
| `loaders/model_loader.py` | Resolve `pickle_file_path`, validate existence, load via joblib/pickle, detect model class | Sec 4.3, Sec 7 (steps 1-2), Sec 8 |
| `loaders/dataset_loader.py` | Resolve `engineered_dataset_path`, load CSV, basic structural checks | Sec 4.4, Sec 7 (step 2), Sec 9 |
| `validators/model_validator.py` | Model-name vs detected-type rules (Rule 1-4), unsupported-model termination | Sec 6, Sec 8 |
| `validators/schema_validator.py` | Dataset emptiness/feature-count checks, target column exclusion, dataset/model compatibility | Sec 9, Sec 10, Sec 11, Sec 12, Sec 13 |
| `explainers/explainer_factory.py` | Model-type -> explainer-class lookup table (config-driven map per project CLAUDE.md guidance against if-else ladders) | Sec 14 |
| `explainers/shap_service.py` | Invoke chosen explainer, compute SHAP values, compute aggregated importance metrics | Sec 15 steps 8-9 |
| `visualizations/plot_generator.py` | Summary plot, bar plot, beeswarm plot generation and file I/O | Sec 16 |
| `exporters/csv_exporter.py` | `global_feature_importance.csv`, `feature_shap_mapping.csv` | Sec 17 |
| `exporters/metadata_exporter.py` | `metadata.json` construction and write | Sec 18 |
| `utils/logger.py` | Session-scoped execution logger (file + level), used by every stage | Sec 19, CFG-02 |
| `utils/output_manager.py` (new, not in Sec 21 but required) | Output-root resolution, session folder creation, path layout for plots/csv/metadata/logs subfolders | Sec 5, Sec 21 (output side), resolves A-2 |
| `config/` (reuse candidate) | Single config.ini + loader for CFG-01..04, reusing `model_library` pattern if confirmed compatible | Sec 28, project CLAUDE.md global rule |
| `tests/` | Standalone test utility per Sec 22, fixture datasets/models, per-stage and end-to-end tests | Sec 22 |

Two structural gaps versus Sec 21 are flagged above (`pipeline.py` and `output_manager.py`) since Sec 21's tree, read literally, has no module that actually sequences the steps or owns the output-folder-creation responsibility described in Sec 5 - every other module is a leaf-level worker.

---

## 6. Class Design

(Conceptual responsibilities only - no code, per instructions. Each class below corresponds 1:1 with a module from Section 5.)

- **`SHAPPipeline`** (in `pipeline.py`)
  Owns the end-to-end run. Single public entry point taking the JSON input contract (Sec 4) and returning/raising based on terminal status. Holds references to one instance of each stage class below. Responsible for the stop-on-failure control flow (Sec 7, Sec 20) and for ensuring metadata/log artifacts are flushed even on failure paths (Sec 20: "errors shall be ... recorded in metadata").

- **`ModelLoader`** (in `model_loader.py`)
  Methods: validate path exists, load artifact (joblib/pickle per A11), detect concrete model class. Returns a loaded-model handle plus detected type string. Does not know about `model_name` validation - that's the validator's job (single responsibility, matches Sec 21's loaders/validators split).

- **`DatasetLoader`** (in `dataset_loader.py`)
  Methods: validate path/readability, load CSV into a dataframe, basic emptiness/row-count check. Does not know about target-column logic (owned by schema validator) or model compatibility (also schema validator) - loader stays purely about getting bytes into memory.

- **`ModelValidator`** (in `model_validator.py`)
  Encodes Rules 1-4 from Sec 8 as discrete methods/lookup rather than nested if-else (per project coding convention to favor hash maps/config over if-else ladders): a supported-model-type set/map drives Rule 4, and a simple equality/alias check (resolving ambiguity A-8 once clarified) drives Rules 1-2. Emits a structured validation result (status + message) rather than directly writing metadata, keeping it decoupled from the exporter.

- **`SchemaValidator`** (in `schema_validator.py`)
  Methods: identify and strip target column (Sec 11, using the configurable strategy from CFG-03), validate feature-count and column presence against model metadata when available (Sec 12), validate dynamic feature names without hardcoding (Sec 13). Returns a cleaned, feature-only dataframe plus the resolved feature-name list used by every downstream stage.

- **`ExplainerFactory`** (in `explainer_factory.py`)
  A single method: given detected model type, return the correct SHAP explainer instance, sourced from the Sec 14 mapping table (implemented as a config-driven map, not nested if/elif, per project convention). Encapsulates explainer-specific construction details (e.g., background data requirement for `LinearExplainer`, once A-5 is resolved) so callers stay model-agnostic.

- **`SHAPService`** (in `shap_service.py`)
  Methods: compute raw SHAP values via the supplied explainer, compute aggregated global importance (mean absolute SHAP per feature, Sec 17.1), and produce the per-record/per-feature long-form table (Sec 17.2). Kept distinct from `ExplainerFactory` so explainer *selection* and SHAP *computation* remain separately testable (Sec 22 lists "explainer selection" and "SHAP value generation" as separate validation checks).

- **`PlotGenerator`** (in `plot_generator.py`)
  One method per plot type (summary, bar, beeswarm) per Sec 16, each taking SHAP values + feature names and writing a PNG to a path supplied by `OutputManager`. No knowledge of session/output-root resolution - paths are injected.

- **`CSVExporter`** (in `csv_exporter.py`)
  Two methods: write global importance CSV (Sec 17.1) and write the feature-SHAP mapping CSV (Sec 17.2). Pure I/O + formatting; takes already-computed data structures from `SHAPService`.

- **`MetadataExporter`** (in `metadata_exporter.py`)
  Builds the `metadata.json` structure (Sec 18) by merging inputs from every prior stage's results (session id, supplied/detected model name, validation status, explainer name, sample/feature counts, timestamp, and - once A-4 is resolved - the discrepancy details). Single method: assemble-and-write.

- **`ExecutionLogger`** (in `logger.py`)
  Thin wrapper around the standard `logging` module configured per CFG-02, scoped to a session-specific log file. Exposes named logging methods/constants for each Sec 19 event category so call sites stay declarative (e.g., `log_event(stage, status, detail)`) rather than free-text log calls scattered inconsistently.

- **`OutputManager`** (in `output_manager.py`)
  Resolves the configured output root (CFG-01), creates `<output_root>/<session_id>/` and its artifact subfolders (resolving A-2), and exposes path-getter methods (`plot_path(name)`, `csv_path(name)`, `metadata_path()`, `log_path()`) consumed by every other exporter/visualization class - centralizing path construction so no other class hardcodes a path, consistent with the project-wide "no hardcoded paths" rule.

- **`SessionContext`** (data holder, possibly in `pipeline.py` or a `models.py`)
  Not a behavior-owning class - a typed structure (akin to the `contracts.py` pattern already used in `epic_3/training_orchestrator`) threading session_id, resolved paths, loaded artifacts, detected type, feature list, SHAP results, and the in-progress metadata dict through the pipeline stages without global state.

---

## 7. Validation Strategy

Validation should be layered to match Sec 7-12, with each layer short-circuiting the pipeline on failure (Sec 7, Sec 20) and every layer's outcome recorded for `metadata.json` (Sec 18) regardless of pass/fail:

1. **Input contract validation** (Sec 4) - all four required fields present and correctly typed before any I/O occurs. Fail fast with a clear "missing field" error; this is the cheapest check and should run first.
2. **Path-existence validation** (Sec 4.3/4.4, Sec 7 step 1, Sec 9) - model file and dataset file exist and are readable, independent of content.
3. **Model load validation** (Sec 7 step 2) - artifact deserializes without exception; corrupted-file handling (Sec 20) wraps the joblib/pickle load in a single well-defined failure path.
4. **Model type detection & support validation** (Sec 7 step 3, Sec 8 Rules 3-4) - detected type must resolve to a known class; unknown/unsupported types terminate with a descriptive message naming the unsupported type (Sec 6).
5. **Model-name consistency validation** (Sec 8 Rules 1-2) - non-terminating; produces a warning + metadata annotation only, never blocks execution. This must be implemented as *non-fatal* explicitly, since it is the one validation rule in the spec that does not terminate on mismatch.
6. **Dataset structural validation** (Sec 9) - non-empty, feature count > 0, readable as a dataframe.
7. **Target column resolution** (Sec 11) - identify and exclude target column using the configured strategy; this step's output (the cleaned feature dataframe) feeds every later validation and computation step, so it must run before schema compatibility validation, not after.
8. **Schema/compatibility validation** (Sec 12, Sec 10) - feature columns exist, feature counts match model expectations where introspectable (e.g., `n_features_in_` on sklearn-compatible models), feature names compared only when model metadata exposes them (graceful skip otherwise, per Sec 10's explicit allowance that the module must not depend on the model carrying feature names).
9. **Dynamic-feature-name verification** (Sec 13) - confirms no part of the pipeline references a hardcoded feature name; this is more of a code-review/test-time invariant than a runtime validation, and should be enforced via the testing strategy (Section 8 below) rather than a runtime check.

Each validation layer should produce a small structured result (pass/fail/warning + message) rather than raising ad-hoc exceptions with inconsistent shapes, so the orchestrator and metadata exporter can treat all validation outcomes uniformly. Terminal failures (Sec 20) should always: log the failure (Sec 19), write a partial `metadata.json` capturing what was known at failure time, and raise/return a clear, user-facing error message - satisfying AC-15 ("validation failures generate meaningful error messages and logs").

---

## 8. Testing Strategy

Sec 22 mandates a standalone test utility decoupled from Epic 2/Epic 3, which aligns well with a layered pytest strategy:

**Unit tests (one per class in Section 6), independent of real Epic 2/3 artifacts:**
- `ModelLoader` - valid load, missing file, corrupted file, each Phase 1 model type (XGBoost, RF, LightGBM, CatBoost, LogisticRegression) trained on a tiny synthetic dataset and pickled/joblib-dumped as fixtures.
- `DatasetLoader` - valid CSV, missing file, empty CSV, unreadable/malformed CSV.
- `ModelValidator` - all four rules from Sec 8 exercised explicitly (match, mismatch-warns-and-continues, undetectable-type-terminates, unsupported-type-terminates).
- `SchemaValidator` - target column present/absent, multiple candidate target columns (once A-3 resolved), feature count mismatch against model, feature name mismatch against model metadata (and graceful skip when model has no feature-name metadata).
- `ExplainerFactory` - correct explainer class returned for each of the five Phase 1 model types; unsupported type raises/handled per Rule 4.
- `SHAPService` - SHAP values shape matches (n_records x n_features); aggregated importance values are deterministic/reproducible given a fixed model+dataset (Sec 24 "Reproducibility" NFR is directly testable here).
- `PlotGenerator` - each of the three plot types produces a non-empty PNG file at the expected path without raising.
- `CSVExporter` - both CSVs have the exact required columns (Sec 17.1/17.2) and exactly one row per record-feature pair for the mapping file (AC-16).
- `MetadataExporter` - required fields present (AC-17), warning-path schema populated correctly once A-4 is resolved.
- `ExecutionLogger` - all Sec 19 event categories are logged at least once across a full successful run; log level configuration (CFG-02) is respected.
- `OutputManager` - session folder + subfolders created when missing, paths returned are well-formed, no path component is hardcoded outside config.

**Integration tests (full pipeline, synthetic but realistic Epic 2/3-shaped fixtures):**
- End-to-end happy path per model type in Sec 6 -> asserts every artifact in AC-07 through AC-13 exists and is non-trivial (non-zero size, parseable JSON/CSV).
- End-to-end failure paths: missing model file, missing dataset file, unsupported model type, empty dataset, schema mismatch - each asserting graceful termination, a logged error, and a metadata.json reflecting the failure (Sec 20, AC-15).
- Model-name mismatch end-to-end - asserts execution *continues* (not terminated) and the warning is both logged and present in metadata (Sec 8 Rule 2, Sec 18).
- Reproducibility test - run the full pipeline twice on identical inputs, assert byte-identical (or value-identical, allowing for timestamp fields) CSV/metadata outputs (Sec 24 NFR).
- Session isolation test - two different `session_id`s in the same run produce non-colliding output folders.

**Test utility (Sec 22's explicit deliverable):**
A small fixture-generation helper (separate from pytest test files, likely under `tests/fixtures/` or `claude_scripts/` per project convention for standalone scripts) that can synthesize a tiny labeled dataset, train one instance of each Phase 1 model type, serialize it, and emit a matching Sec 4 JSON payload - enabling both the integration tests above and ad hoc manual runs before Epic 2/Epic 3 are integration-ready, exactly as Sec 22 requires ("executable independently of upstream epics").

**Non-functional / acceptance traceability:**
A final pass mapping every AC-01..AC-17 to a specific test function (one test, or a small group of tests, per AC) is recommended as a lightweight traceability matrix once tests exist, so the acceptance criteria in Sec 27 are demonstrably covered rather than informally assumed.

---

## Summary of Blocking Items Before Implementation

The following open items should be resolved with stakeholders before implementation begins, as they affect class/module contracts directly:

1. OI-05 / target column presence and naming convention (blocks `SchemaValidator` design).
2. OI-06 / binary vs multiclass vs regression support (blocks `SHAPService`, `PlotGenerator`, and CSV schema design).
3. OI-01 / output root default and configuration mechanism (blocks `OutputManager`).
4. A-2 / output artifact folder structure (currently undefined in spec, must be decided rather than inferred).
5. A-5 / LinearExplainer background-data policy.
