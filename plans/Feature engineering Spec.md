# SPEC: Feature Engineering Agent

---

## 1. GOAL

A pipeline that takes a raw tabular dataset and produces ML ready features. It handles missing values, encodes categoricals, creates new features, selects the best ones, and hands back a transformed dataset along with a JSON file that records every transform applied. That JSON is what gets used when the pipeline runs on new data given a raw row and the file, you can reproduce the exact output without rerunning the pipeline.

The model is only called where a human looking at column names and sample values would need to stop and think. Everything else is code.

---

## 2. PACKAGES TO BE USED

- google-adk (LLM transport — `OpenAICompatibleLlm` wraps the provider client; tool dispatch is plain Python)
- openai (HTTP client used by `OpenAICompatibleLlm` for any OpenAI-compatible endpoint)
- pandas
- numpy
- scikit-learn
- scipy
- ray
- mrmr-selection
- pyyaml
- pydantic
- dataclasses (stdlib)
- logging (stdlib)
- ast (stdlib)
**Bring-your-own model.** The pipeline is model-provider-agnostic. The caller supplies a model string (e.g., `gemini/gemini-2.0-flash`, `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`) and sets the API key in `config.yaml` under `llm.api_key`. The orchestrator copies the key into `os.environ[llm.api_key_env_var]` at the very start of startup, before any ADK or provider-client import; if the configured env var name is the OpenAI default but the model string targets a different provider, the orchestrator derives the correct env var name from the model prefix and logs the override. ADK's `LlmRequest` carries the prompt; the actual provider call is the OpenAI-compatible HTTP path inside `pipeline/openai_llm.py`. No tool-dispatch loop is delegated to ADK. `config/config.yaml` must be added to `.gitignore` since it holds the real key.

---

## 3. LIST OF ALGORITHMS FOR EACH MODALITY

*[code] = no model call. [agent] = model is called to make the decision.*

The pipeline is deterministic except at the single decision point where reasoning over precomputed statistics changes the outcome: feature selection. Reporting calls the model only as a styling layer over an already-finalised summary, with a template fallback for offline runs.

### Imputation (4) — [code]
1. Median fill
2. Mode fill
3. KNNImputer
4. IterativeImputer (MICE)

Strategy per column is rule-based: numeric columns default to median, categorical/binary to mode, and any column whose null rate exceeds `imputation.null_drop_threshold` (config) is dropped before any other transform runs. KNN / IterativeImputer are available as drop-in strategies and are selected via deterministic predicates on the column profile (null-rate band, dtype, correlation evidence).

### Outlier Handling — [code]

Detectors:
5. IQR
6. Z-score
7. IsolationForest (detector only)

Actions:
- scale     → apply RobustScaler to the column (default for numeric)
- flag      → add binary column `<col>_is_outlier`, keep row intact
- drop_row  → drop row and corresponding target row atomically (explicit opt-in only)

The default `(detector, action)` pair is `(iqr, scale)`. Detector and action are configurable per column via `config.outlier.*`; no model call is made.

### Categorical Encoding (1) — [code]
8. LabelEncoder

### Datetime Decomposition — [code]
Detection: SemanticTypeInfer assigns type `datetime`; that column is routed here.
9. Year
10. Month
11. Day of month
12. Quarter
Original datetime column is dropped after decomposition.

### Feature Creation Operations (15) — [code]
13. ratio
14. difference
15. product
16. sum_group
17. square
18. sqrt
19. log1p
20. row_mean
21. row_max
22. row_count_positive
23. days_since
24. is_recent
25. equal_width_bins
26. quantile_bins
27. cross_categorical

Operation selection runs through `FeatureCreator.create_deterministic`: a rule-based generator emits proposal specs from column types and proxy MI, the validator enforces `temporal_class` and provenance, and code executes the surviving specs. `cross_categorical` carries `temporal_class: pre_encoding` and runs before the Encoder; every other operation is `post_encoding`.

### Scaling (4) — [code]
28. StandardScaler
29. RobustScaler
30. MinMaxScaler
31. PowerTransformer (Yeo-Johnson)

Scaler per numeric column is picked from the distribution profile (skewness, kurtosis, outlier rate, boundedness). Default is `standard`.

### Feature Selection — single LLM call over precomputed stat artifacts [agent]

A deterministic stat phase runs after Scaler and writes the artifacts in `pipeline/feature_stats.py` to `.mitra/<run_id>/stats/`: `mutual_info.json`, `information_gain.json`, `mrmr_ranking.json`, `variance.json`, `correlation_pearson.json`, `correlation_spearman.json`, `clusters.json`, `linear_baseline.json`, `pca.json`. All math (MI via k-NN estimator, entropy-based information gain via `KBinsDiscretizer` + `mutual_info_classif(discrete_features=True)`, mRMR, variance, correlation pairs, linear baseline, PCA explained-variance) is computed once here and shared with `FeatureSelector` via static helpers (`_mi_scores`, `_information_gain`, `_mrmr`, `_lasso`, `_laplacian_score`, `_top_mi`, `_pca`) so the same primitives back both the artifacts and any in-tool fallback.

`FeatureSelector` reads those artifacts and makes **one** LLM call. The prompt presents the precomputed stats — never the raw data — and asks the model to (a) pick which columns to keep up to `top_k_features`, (b) decide whether to compress the kept columns via PCA, and (c) pick `pca_n_components` when PCA is on. The response is parsed against `FeatureSelectionResponse` (`pipeline/responses.py`): `{keep, drop, use_pca, pca_n_components, rationale}`. Code then applies the result deterministically and records `state.selection_method` as one of:

- `"llm_select"` — LLM-picked columns, no PCA.
- `"llm_select+pca(N)"` — LLM-picked columns compressed into N components.
- `"fallback:mrmr_all"` — LLM call failed (parse, empty response, transport error) or no model is configured; fallback runs mRMR over all features at `top_k_features`.
- `"none"` — no feature columns existed.

There is no Judge sub-agent on this path: a single structured call over already-decomposed statistics is sufficient for the keep/drop + PCA decision.

### Type Inference — [code]
`SemanticTypeInfer` is heuristic-only. Each column is assigned a type from its pandas dtype, null rate, nunique, regex signature (UUID / email / ISO-date / phone / numeric-string), and sample values. No model call is made.

Valid types: `numeric`, `categorical`, `datetime`, `id`, `text`, `binary`, `target`. Columns assigned type `id` are dropped automatically before any transformation runs.

### Reporting — [agent] iff `config.report.use_llm` is true
The model is called once at the end to write the report from the structured pipeline summary. When `config.report.use_llm` is false (the default) or no model client is provided, `reporter.py` writes a deterministic template that fills the same Markdown headings from the same summary dict — no separate file, no schema change.

---

## 4. APPLICATION

**How it gets called.** The caller passes a dataset path, task type, target column name, and a model string. The corresponding API key must be set as an environment variable before the pipeline starts. The pipeline runs end to end and returns the engineered dataset and the artifact file. No other configuration is needed.

**Null detection in categoricals.** For columns SemanticTypeInfer assigns the type `categorical` or `binary`, the string tokens `"None"`, `"NA"`, `"N/A"`, `"na"`, `"n/a"`, `"none"`, and `"NaN"` are preserved as a literal category value — they are *not* treated as missing data. Many real datasets use these strings as meaningful labels (e.g., `Alley="NA"` means "no alley access" in the House Prices dataset), and converting them to nulls erases that signal. Concretely: pandas' default `read_csv` NaN coercion is disabled (`keep_default_na=False`, explicit `na_values=[""]`) so the strings reach SemanticTypeInfer intact. After typing, only columns assigned `numeric`, `datetime`, or `target` participate in null detection by `MissingValueHandler`; for categorical and binary columns the imputer never sees these tokens as nulls. The token list lives in `config.yaml/imputation.categorical_null_literals` so a caller can extend or shrink it without code changes.

**Numeric placeholder normalization.** The same `keep_default_na=False` setting that protects categorical columns also lets `"NA"`-style tokens reach SemanticTypeInfer in *numeric* columns — `LotFrontage`, `MasVnrArea`, `GarageYrBlt` in House Prices all use `"NA"` for missing measurements. Those columns arrive as `dtype=object` mixing numeric strings and `"NA"`, and SemanticTypeInfer correctly assigns them type `numeric` from the regex signature and sample values. Immediately after SemanticTypeInfer returns, the orchestrator runs a single normalization pass: every column whose assigned type is `numeric` and whose dtype is not already numeric is coerced via `pd.to_numeric(col, errors="coerce")`. The `"NA"` tokens become real `NaN`s at this point and reach `MissingValueHandler` as actual nulls. The categorical-null-literals rule above still applies because only columns typed `numeric` are coerced — a `"NA"` in a categorical column stays a category label. The pass runs once, before any other tool sees the dataframe.

**Imputation decisions.** Strategy per column is rule-based on the column profile. Numeric columns default to median fill; categorical / binary to mode fill. A column whose null rate exceeds `imputation.null_drop_threshold` (default 0.50) is dropped entirely. KNN and IterativeImputer are selectable strategies for columns whose null-rate band and correlation evidence warrant them; the selecting predicate lives in `imputer.py` and is configurable through `config.imputation.*`. No model call is made.

**Outlier decisions.** Default behaviour is `(detector=iqr, action=scale)` for every numeric column. The three valid actions are scale, flag, and drop_row as defined in §3. The action is configurable per column via `config.outlier.*`; the detector is meaningful only when the action is `flag` or `drop_row` (the detector mask is what gets flagged or dropped) and is ignored when the action is `scale` (the whole column is RobustScaled). The `OutlierDecision` Pydantic schema declares `detector: Literal["iqr","zscore","isolation_forest"] | None` so the schema can describe both the rule path and a future agentic path without divergence.

**Feature selection decisions.** Profiling runs first, then `feature_stats.py` writes every artifact the selector needs to `.mitra/<run_id>/stats/` (mutual information, information gain, mRMR ranking, variance, Pearson + Spearman correlation pairs, clusters, linear baseline, PCA explained variance). `FeatureSelector` then makes a single LLM call, passing all of those artifacts in the prompt — the model never sees raw data, only the precomputed statistics. The model returns `{keep, drop, use_pca, pca_n_components, rationale}`; code de-duplicates, caps at `top_k_features`, and either keeps the chosen raw columns or compresses them into `pca_n_components` (defaulting to the smallest component count that retains `pca_variance_retained`). When the LLM call fails the selector falls back to deterministic mRMR over all features at `top_k_features`. The rationale for keeping this single-call shape: every method-level decision (mRMR vs PCA vs lasso vs rf-importance) reduces to which precomputed ranking to trust and whether the data is wide-enough to benefit from PCA — both readable directly from the artifact set.

**Scaling decisions.** Scaler per numeric column is picked deterministically from the distribution profile: near-normal columns get StandardScaler, skewed columns or those with high outlier rates get RobustScaler, bounded columns get MinMaxScaler, heavily skewed columns get PowerTransformer (Yeo-Johnson). Default is `standard`. Skewness, kurtosis, outlier rate, and bounded flag are already on the profile; the selecting predicate is the only thing the scaler reads.

**Target column handling.** The target is separated from features at ingestion. It goes through imputation if it has missing values. If the target is categorical (classification with string labels) it is label encoded. It is never scaled or included in feature selection. Feature creation uses it as a reference for MI scoring only. It is always the last column in the output.

**Task type.** Valid values are classification and regression. If --task is not supplied, the pipeline infers it from the target column at startup: if the target is numeric and has more than task_infer_nunique_threshold unique values it is treated as regression, otherwise classification. The threshold lives in config.yaml. If --task is supplied, it is validated against the two valid values and a ValueError is raised on anything else. Mutual information and other task-sensitive methods switch variant accordingly based on the resolved task type.

**Parallel execution.** Univariate analysis (per-column null rates, distribution shape, skewness, top categories) runs as a `@ray.remote` map across columns inside `DataProfiler`. Multivariate analysis (pairwise correlations, mutual information) is computed alongside it. The `feature_stats.py` phase computes its expensive artifacts (MI, RF importance, mRMR ranking, correlation clusters, linear baseline, PCA) once and persists them to `.mitra/<run_id>/stats/` so the selector reads JSON rather than recomputing; expensive primitives within that phase use the same Ray `num_cpus` cap.

---

## 5. CONSTRAINTS

**Orchestrator: deterministic Python sequence.** `FeatureEngineerOrchestrator` (`pipeline/orchestrator.py`) runs the tools in a fixed order — profile → infer types → handle missing → handle outliers → encode → create → scale → compute feature stats → select features → validate → write report. There is no ADK agent loop; the orchestrator dispatches each step directly and logs `ok` / `error` plus elapsed time to `execution_log.txt`. Precondition / postcondition errors are logged and the pipeline continues to the next step.

**LLM transport: ADK-native, model-provider-agnostic.** The caller supplies a model string (e.g., `openai/gpt-4o`, `gemini/gemini-2.0-flash`, `anthropic/claude-3.5-sonnet`). `_make_model_call` builds a `pipeline/openai_llm.py::OpenAICompatibleLlm` and routes prompts through ADK's `LlmRequest` pipeline. A model string is required — the pipeline refuses to start without one. The API key is read from `config.yaml/llm.api_key` and injected into `os.environ[llm.api_key_env_var]` as the first step of orchestrator startup, before any ADK / provider-client import. If the configured env var is the OpenAI default but the model string targets a different provider, the orchestrator derives the correct env var from the model prefix and logs the override. The pipeline refuses to start if `llm.api_key` is empty, missing, or still the placeholder.

Pipeline tools are plain `BaseTool` subclasses called directly from the orchestrator's step sequence (`Tool(...)(state)`). They receive `PipelineState` as an argument and mutate it in place. `pipeline/tools/adk_tools.py` provides ADK-wrapper shims and a `set_pipeline_state(...)` injection point that remain in the codebase for future agentic-mode orchestration; the deterministic orchestrator does not call them.

**Evidence Packets.** Every model call receives a typed `EvidencePacket` constructed by the calling tool. Packets are defined as dataclasses in `pipeline/evidence.py` and rendered to prompts by a single serializer. Tools must not build prompt strings from ad-hoc f-strings over the profile dict — the dataclass is the contract.

The required schema per decision point:

- `SemanticTypeInferEvidence` — per column: `dtype`, `null_rate`, `nunique`, `top_values` (up to 5), `random_samples` (5 string-cast values), `regex_signature` (hits for UUID / email / ISO date / phone / numeric-string patterns).

- `MissingValueHandlerEvidence` — per column with nulls: `null_rate`, `null_run_lengths` (histogram of consecutive-null streak lengths; distinguishes MCAR from MNAR), `null_mask_corr_top5` (other columns whose values correlate with this column's null mask), `target_rate_when_null` vs `target_rate_when_present`, `random_present_values` (10), `dtype`, `semantic_type`.

- `OutlierHandlerEvidence` — per numeric column: 10-bin histogram, `extreme_values_top5` and `extreme_values_bottom5` with their aligned target values, `mi_with_target`, `target_corr`, `downstream_model_hint` (`linear` or `tree`, supplied by the caller in `config.yaml`).

- `ScalerEvidence` — per numeric column: 20-bin histogram, `skewness`, `kurtosis`, `outlier_rate`, `bounded` (boolean + range if true), `monotonic_with_target` (Spearman rank correlation).

- `FeatureCreatorEvidence` — per column: `semantic_type`, `mi_with_target`, `nunique`, `correlated_with_top3`, `decomposed_from` (provenance for datetime children), and a global `co_occurring_pairs` list (top column pairs by joint MI).

- `FeatureSelectorEvidence` (single-call, stat-artifact-backed) — global `n_rows`, `n_features`, `features_per_row`, `task`, `linear_baseline_score` (CV-AUC or CV-R² of a logistic/linear baseline on the top-K MI features), plus the top-N entries from each precomputed artifact: `mutual_info` scores, `information_gain` scores, `mrmr_ranking`, `low_variance` columns, `correlation_pearson` high-pairs, and the `pca` summary (`n_components_for_threshold`, `variance_retained`). The selector reads these directly from `.mitra/<run_id>/stats/` and renders them into the prompt; FeatureSelector never recomputes them in the per-run path.

EvidencePacket fields are the only data the model sees. Adding a field is a code change in `evidence.py` plus a config entry, never a prompt edit.

**`evidence_cited` form.** The renderer emits dotted paths in the sent-field whitelist (`columns.dtype`, `columns.null_run_lengths`). Models cite fields naturally in either dotted form (`columns.dtype`) or indexed form (`columns[3].dtype`) since the serialized JSON they see contains a list under `columns`. Both forms are accepted: the validator strips `[<int>]` brackets from each `evidence_cited` entry before whitelist membership (`re.sub(r"\[\d+\]", "", cited)`). The whitelist itself stays dotted; the equivalence is enforced at the matcher, not at the renderer. Prompts do not advertise either form — the model picks whichever is natural for it.

The orchestrator runs a fixed step sequence. The model is consulted at exactly two decision points: **FeatureSelector** (one call, always; per §3), and **FeatureReporter** (one call iff `config.report.use_llm` is true). Both follow the Evidence-Packet → Pydantic-response → revise-once → fall-through contract below — FeatureReporter's `EvidencePacket` is just the final pipeline-summary dict and its "rev" path is the deterministic template.

**Prompts describe, do not prescribe.** Prompt templates may state what each strategy does mechanically (e.g. "RobustScaler centres by median and scales by IQR") but may not enumerate when to apply it (e.g. "use RobustScaler when the column is skewed"). The latter is the deterministic policy written in English — it converts the LLM into a lookup table and defeats the reason for calling it. If a strategy can be selected by a fixed rule over scalar features, the call is replaced with code; if it cannot, the model must derive the mapping from the EvidencePacket without a hint sheet in the prompt.

To make this rule auditable, each tool with a strategy choice owns a module-level `STRATEGY_DEFINITIONS: dict[str, str]` constant — one entry per allowed strategy, value is its mechanical description only. The prompt template injects this dict verbatim. A `when-to-use` mapping must not appear in the same file as `STRATEGY_DEFINITIONS` or anywhere in the prompt template; review of any change to either is bound to this rule.

Nothing the model outputs is applied to data directly. Every model response is parsed against a Pydantic model `<ToolName>Response` defined in `pipeline/responses.py`. Each per-item decision must carry four fields: `strategy` (or `type` / `operation` depending on the call), `rationale` (free text, minimum length from `config.yaml`), `evidence_cited` (list of EvidencePacket field names the rationale references), and `alternatives_considered` (list of other strategies the model weighed before picking).

A response is rejected — and the call is retried exactly once with `prior_response_was_uninformative=true` plus a delta-evidence pack contrasting the columns that received the same answer — if any of the following hold:

1. `evidence_cited` is empty or references field names absent from the prompt.
2. `rationale` is shorter than the configured minimum or matches a denylist of boilerplate phrases stored in `config.yaml/validation.boilerplate_denylist`.
3. `alternatives_considered` is empty.
4. The batched response is degenerate: more than `validation.lazy_response_threshold` (default 0.8) of items share the same strategy *tuple* — the joint of every `Literal` field on the response item (e.g. `(detector, action)` for OutlierDecision, `(operation, temporal_class)` for CreatorSpec) — across a heterogeneous EvidencePacket. Single-field degeneracy checks miss "all 50 columns got `(iqr, scale)`"; the joint check catches it. The degeneracy check fires only when the batch has at least `validation.lazy_min_batch_size` items (default 3); smaller batches skip the check because the "share-one-strategy" signal is uninformative at that scale.

If the retry also fails the four checks, the tool falls through to its deterministic default and the failure is recorded in `state.warnings` with the rejected response attached. The retry budget is one per tool call, not per item.

On schema parse failure the pipeline attempts coercion in order: cast to float, then parse as timestamp, then apply LabelEncoder. If all three fail the operation is skipped and a warning is logged. Coercion handles malformed types; the four checks above handle lazy content. The two paths are distinct and run in this order: parse → coerce → content-check → revise-once → fall through.

**FeatureValidator coercion is stricter.** The validator runs after FeatureSelector and is the last gate before the dataframe is written. Its `_try_coerce` path is float → datetime only — LabelEncoding inside the validator is **forbidden**. Encoding belongs to the Encoder; if a column reaches the validator still non-numeric and non-datetime, it is dropped from `state.selected_columns` and a warning is logged. Silent LabelEncoding at this stage hides upstream typing or normalization bugs (e.g. an object-dtype numeric column that bypassed the §4 normalization pass), so the validator refuses to paper over them.

`cross_categorical` operations carry `temporal_class: pre_encoding` and execute before the Encoder runs. Any FeatureCreator operation spec missing a `temporal_class` is rejected at plan-validation time.

Parallelism uses Ray. Ray is initialised once at orchestrator startup via `ray.init(num_cpus=8)` which enforces the global cap of eight workers. Ray uses shared memory for numpy arrays and pandas DataFrames (zero-copy via Apache Arrow) so there is no serialisation overhead for sklearn objects. All parallel work is expressed as `@ray.remote` functions in `pipeline/parallel.py`. No tool initialises Ray or sets worker counts — that is the orchestrator's responsibility at startup.

Every stochastic operation takes `random_state=42`. Same inputs must produce the same outputs on every run. KNNImputer is deterministic by algorithm and does not accept a seed. IterativeImputer is the only imputer that requires `random_state=42`. The seed is read from `config.pipeline.random_state` — no `random_state=42` literal is allowed in tool code.

**Deterministic naming.** Tools that synthesize column names (FeatureSelector PCA expansion, any future tool that derives names from a set of source columns) must produce the same names on every run for the same inputs. Python's built-in `hash()` is salted per process and is forbidden anywhere a name lands in `state.df` or in `feature_artifact.json`. Use `hashlib.md5("|".join(sources).encode()).hexdigest()[:8]` or a state-carried counter for any generated identifier.

**Tool idempotency.** Each tool's postcondition is the gate that the next tool reads through its precondition. The orchestrator runs each step exactly once, so re-entry is not a steady-state concern; the idempotency predicates (`state.column_types is not None` for SemanticTypeInfer, `state.row_count_after_outlier is not None` for OutlierHandler, `state.selected_columns is not None` for FeatureSelector, etc.) live in `adk_tools.py` as no-ops on the deterministic path and as gate checks for the future agentic-mode orchestrator. Append-only fields (`state.transformers`, `state.dropped_columns`, `state.created_columns`, `state.warnings`) must not double-write if a tool is invoked a second time.

**Startup smoke test.** Before the ADK runner starts, the orchestrator sends one structured prompt — a minimal `EvidencePacket` plus the matching `## RESPONSE SHAPE` header — through the same `_make_model_call` path the tools use. The response is parsed via `validate_response` with thresholds relaxed (`min_rationale_chars=1`, `min_alternatives=0`, no boilerplate denylist) so the smoke check probes only the JSON-shape and field-citation pathway, not response quality. If `validate_response` returns `failures=['parse']` the pipeline aborts at startup with the raw response attached to the error. Content failures other than `parse` are logged but do not abort — the smoke prompt is not a content benchmark, only a transport + parse + whitelist test.

No hardcoding in any .py file. All default values, thresholds, algorithm parameters, and model hyperparameters live in `config/config.yaml`. The schema is a Pydantic model `ConfigSchema` defined in `pipeline/config.py`. At startup `config.yaml` is loaded with pyyaml and validated via `ConfigSchema(**raw_dict)` — a missing or wrong-typed key raises at import time, not mid run.

---

## 6. OUTPUTS & PROJECT STRUCTURE

```
config/
    config.yaml                 all thresholds, algorithm parameters, strategy rules

pipeline/
    __init__.py
    state.py                    PipelineState dataclass and all sub dataclasses
    config.py                   ConfigSchema Pydantic model; loads and validates config.yaml at import time
    base.py                     BaseTool ABC; precondition checks inputs not None, postcondition checks output not None, __call__ chains both around run()
    parallel.py                 Ray remote function definitions; ray.init(num_cpus=8) called once at orchestrator startup
    evidence.py                 EvidencePacket dataclasses + serializer; sole contract for what each tool sends to the model
    responses.py                Pydantic response models + validate_response(...) helper (parse + content checks + degeneracy check)
    feature_stats.py            FeatureStatsComputer — Phase B; computes MI / information gain / mRMR / variance / correlation / clusters / linear baseline / PCA artifacts and writes them to .mitra/<run_id>/stats/
    judge_agent.py              JudgeAgent — isolated ADK sub-agent; defined for future agentic-mode orchestration (FeatureCreator ranking, SelectionPlanResponse). Not invoked from the deterministic orchestrator.
    openai_llm.py               OpenAICompatibleLlm — ADK-native LLM client used by _make_model_call for OpenAI-compatible endpoints
    orchestrator.py             FeatureEngineerOrchestrator; runs the deterministic step sequence, owns startup smoke test and feature_artifact.json
    tools/
        __init__.py
        adk_tools.py            ADK tool wrappers + set_pipeline_state() injection point. Retained for future agentic-mode orchestration; the deterministic orchestrator does not call them.
        profiler.py             DataProfiler
        infer.py                SemanticTypeInfer (heuristic-only)
        imputer.py              MissingValueHandler
        outlier.py              OutlierHandler
        encoder.py              Encoder
        creator.py              FeatureCreator (create_deterministic on the orchestrator path)
        scaler.py               Scaler
        selector.py             FeatureSelector (single LLM call over .mitra stats; mRMR fallback)
        validator.py            FeatureValidator (float → datetime coercion only; no LabelEncoding)
        reporter.py             FeatureReporter (LLM iff config.report.use_llm, else template)

schema.md                       full input/output contract, all args documented
README.md                       entry point for any caller, links schema.md
main.py                         CLI: `python main.py run data.csv --task classification --target churn --model <model_string>`
```

**What a run produces.**

`pipeline_output/<run_id>/engineered_dataset.csv` — the transformed dataset, all features numeric, scaled, and selected, ready to pass to a model.

`pipeline_output/<run_id>/feature_artifact.json` — full record of every transform applied, written by the orchestrator (not the reporter). Schema:

```json
{
  "run_id": "20260609T143022_a3f1b2c4",
  "task": "classification",
  "target_column": "churn",
  "dropped_columns": ["id", "col_high_nulls"],
  "created_columns": [{"name": "col1_div_col2", "operation": "ratio", "sources": ["col1", "col2"]}],
  "transformers": [
    {"step": "imputation", "column": "col1", "strategy": "median", "fill_value": 3.5},
    {"step": "encoding",   "column": "col2", "strategy": "label",  "classes": ["a","b","c"]},
    {"step": "scaling",    "column": "col3", "strategy": "standard","mean": 0.5, "std": 1.2}
  ],
  "selected_columns": ["col1", "col3", "col1_div_col2"],
  "selection_method": "llm_select",
  "warnings": ["col_x had 52% nulls and was dropped"]
}
```

`selection_method` is one of `llm_select`, `llm_select+pca(N)`, `fallback:mrmr_all`, or `none` (defined in §3 Feature Selection). Use this file to replay the same transforms on new data without re-running the pipeline.

`pipeline_output/<run_id>/report.md` — a report covering data quality findings, encoding decisions, features created, selection results, and recommendations. Written by the model from a structured summary when `config.report.use_llm` is true. Falls back to a string constant template in `reporter.py` that dumps the structured summary dict under fixed section headings (`Data Quality`, `Encoding`, `Features Created`, `Feature Selection`, `Warnings`).

`pipeline_output/<run_id>/execution_log.txt` — append-only log of every step that ran, status (`ok` / `error`), elapsed seconds, and, when a step made an LLM call, the call's source tag (`ok`, `ok:revised`, `fallback`).

`pipeline_output/<run_id>/raw_responses.txt` — one entry per LLM attempt (first and revision) with `caller`, `attempt`, `status`, failure reasons, and the raw response body truncated to `validation.raw_log_max_chars`.

`.mitra/<run_id>/stats/*.json` — the precomputed deterministic stat artifacts read by `FeatureSelector`: `mutual_info.json`, `information_gain.json`, `mrmr_ranking.json`, `variance.json`, `correlation_pearson.json`, `correlation_spearman.json`, `clusters.json`, `linear_baseline.json`, `pca.json`. Persisted under `.mitra/` rather than `pipeline_output/` so they survive across re-runs of the same dataset and can be inspected separately from the engineered output. The directory is `state.stats_dir`; the workspace root is `config.paths.workspace_root`.

`run_id` is generated as `YYYYMMDDTHHMMSS_<8-char uuid>` (e.g. `20260609T143022_a3f1b2c4`) at orchestrator init, stored in `PipelineState`, and returned to the caller alongside the output paths.

**Observability detail.**
- `execution_log.txt` is the canonical per-tool record. One line per tool with: ISO timestamp, tool name, status (`ok` / `error`), elapsed seconds, and a short detail string. For tools that make a model call the detail string includes the LLM call source tag from `call_with_revision` (`ok`, `ok:revised`, or `fallback`) so a reader can see at a glance whether the deterministic default was used.
- `raw_responses.txt` records every LLM attempt (first and revision) for every tool and the Judge sub-agent. One entry per attempt with `caller`, `attempt`, `status` (`ok`, `ok:revised`, `rejected`, `fallback`), failure reasons, and the raw response body. The body is capped per attempt at `validation.raw_log_max_chars` (default 60000); longer bodies are truncated with a tail marker `... [truncated, total N chars]`. The cap is configurable because long batched responses (a 100-column SemanticTypeInfer batch can run past 20kB) need to land in the log intact for post-hoc debugging.

---

## 7. OPEN AMBIGUITIES

### A. Semantic Type Taxonomy
SemanticTypeInfer assigns a "type" to every column, but the valid type vocabulary is never defined. What are the possible types — numeric, categorical, datetime, id, text, binary, target? How does each type govern downstream behavior (e.g., does an "id" column get dropped automatically)?
Solution: Yes if the the column is primary key or a unique key the it does not contribute for anything so can be dropped.

### B. Imputation Drop Threshold
"A column with 70% nulls might get dropped entirely." Is 70% a hard threshold or an example? If it is a threshold, it should live in config.yaml with a named key. The word "might" leaves the drop decision undefined.
Solution: Yes if the columns has more then 50% null values drop.
### C. Outlier Removal Scope
"Remove" appears in both IsolationForest flag and remove and in the application section describing outlier behavior, but it is never clarified whether removal means dropping the affected rows or dropping the column. For row removal: what happens to the corresponding target values?
Solution: 
outlier_action:
  - "scale"       → apply RobustScaler to the flagged column (default for numeric)
  - "flag"        → add binary column `<col>_is_outlier`, keep row intact
  - "drop_row"    → drop row AND corresponding y[i] atomically (explicit opt-in only)

### D. Feature Selection Ensemble
"When it is unclear, the ensemble runs." The ensemble is never defined. Which of the eight selection methods are included? How are their scores aggregated — average rank, vote, union of top k? Is there a configurable k?
Solution: ensemble is not needed, If features are collectively redundant and individually weak (high inter-correlation, low MI), apply PCA. If features are individually significant (high MI, low inter-correlation), apply mRMR. If both conditions partially hold, apply mRMR on significant features first, then PCA on the residual redundant block. Default fallback: mRMR.

### E. Orchestrator order: fixed sequential
The orchestrator runs the tools in the fixed order listed in §6, inserting a `compute_feature_stats` step between scaler and selector:

`profile → infer types → handle missing → handle outliers → encode → create → scale → compute_feature_stats → select → validate → report.`

Each step is invoked exactly once and `PreconditionError` / `PostconditionError` is logged before continuing to the next step (the pipeline does not abort on a single tool failure — the state may already carry enough work to be useful, and the artifact records the warnings). Postcondition failures do not trigger an in-place retry; the next step's precondition will block if its inputs are missing, and the warning is what the operator inspects to decide whether to rerun the whole pipeline. Per-call LLM revision (§5 / §7-G) is the only retry loop in the pipeline.

### F. FeatureCreator Output Cap
No upper bound is defined on the number of operations the model may propose. Twenty source columns can produce hundreds of pairwise ratios, products, and differences. Without a cap, feature count can explode before selection runs. Should config.yaml hold a max_created_features key?

Solution: cap the number of created columns at `feature_creation.cap` (config). The cap applies to `FeatureCreator.create_deterministic` — the deterministic generator emits operation specs ordered by proxy-MI of source columns, redundancy against already-emitted specs, and operation-type diversity, then truncates at `cap`. Specs are validated for `temporal_class` (`pre_encoding` for `cross_categorical`, `post_encoding` for everything else) before execution; specs missing `temporal_class` are rejected.

A Judge sub-agent (`pipeline/judge_agent.py`) is defined for future agentic-mode orchestration. Its declared schemas — `FeatureCreatorResponse` (ranking) and `SelectionPlanResponse` (per-cluster `selection_plan` with action vocabulary `mrmr`, `pca`, `mrmr_then_pca`, `drop`, `lasso`, `rf_importance`) — remain in `responses.py` so the contract is stable. The deterministic orchestrator does not call Judge: FeatureCreator runs `create_deterministic`, and FeatureSelector makes the single-call decision described in §3 / §4.

### G. Validation and Revision
"Type assignments and operation specs both go through a validation step before anything touches the dataframe." What happens on failure — is the operation silently skipped, is an exception raised, or is the model retried with an error message? The validator.py tool listed in §6 is never described.

Solution: two failure modes are handled separately.

**Type failures (malformed output).** When the model returns a value that does not fit the response schema — wrong type, missing field, bad JSON — the pipeline attempts coercion in order: cast to float, then parse as timestamp, then apply LabelEncoder. If all three fail the operation is skipped and a warning is logged.

**Content failures (lazy or unjustified output).** When the model returns a syntactically valid response that fails the four content checks in §5 (empty `evidence_cited`, boilerplate `rationale`, empty `alternatives_considered`, or a degenerate batch where more than `validation.lazy_response_threshold` of items share a strategy despite heterogeneous evidence), the call is retried exactly once with a delta-evidence pack that highlights the contrasts the model missed. If the retry fails the same checks the tool falls through to its deterministic default with the rejected response attached to `state.warnings`.

These are the only revision paths. Coercion is not revision; a coerced response still has to pass the content checks.
### H. Target Column Handling
The target column is never mentioned in the context of transformations. Is it excluded from imputation, encoding, scaling, and selection automatically? Is it passed through unchanged? Does its presence affect feature creation (e.g., should features be created relative to the target)?
Solution: Target is separated from features at ingestion. It goes through imputation if it has missing values. If the target is categorical (classification with string labels) it must be label encoded. It is never scaled or included in feature selection. Feature creation uses it as a reference for MI scoring only. It is always passed through to the output as the last column.

### I. LabelEncoder Only / Categorical Coverage
Only LabelEncoder is listed. There is no handling defined for high cardinality categoricals, ordinal vs. nominal distinction, or binary columns. Is OrdinalEncoder, TargetEncoder, or frequency encoding out of scope? What is the maximum cardinality before a column is treated differently?
Solution: only label encoder is in scope

### J. Per-call defaults / offline behaviour
There is no global offline mode — the smoke test runs at startup and aborts the pipeline on `failures=['parse']`. The deterministic orchestrator's tools each have a fixed default they always use: median / mode for imputation, `(iqr, scale)` for outliers, `standard` for scaling, the rule-based `create_deterministic` generator for feature creation, and dtype + regex-signature for SemanticTypeInfer. Those defaults are the primary execution path, not fallbacks. The two tools that consult the LLM (`FeatureSelector`, and `FeatureReporter` when `config.report.use_llm` is true) wrap their call in `call_with_revision`: a parse or content failure triggers exactly one revision retry, and a second failure falls through to the tool-level deterministic default — `FeatureSelector` runs mRMR over all features at `top_k_features` and records `selection_method = "fallback:mrmr_all"`; `FeatureReporter` writes the template report. The rejected response is recorded to `raw_responses.txt` and a summary line is appended to `state.warnings`.

### K. Cross-Categorical Timing
cross_categorical (operation #27) is listed under Feature Creation, which runs after Encoding per the tool order in §6. By that point, categorical columns have already been label encoded to integers. Does cross_categorical operate on the encoded numeric values or is it expected to run before encoding? The intended pipeline order is not explicitly stated.
Solution: cross_categorical must execute before Encoding (temporal_class: pre_encoding). Operations missing temporal_class are rejected at plan-validation time.

### L. Datetime Detection and Post-Decomposition Handling
Datetime decomposition is listed but the detection mechanism is not specified is it driven by SemanticTypeInfer, pandas dtype, or a regex pass on values? After decomposition into year, month_sin/cos, dow_sin/cos, and is_weekend, is the original datetime column dropped or retained?
Solution: Detection is driven by SemanticTypeInfer — columns assigned type `datetime` are routed to decomposition. Decompose into year, month, day, quarter. Original datetime column is dropped after decomposition.

### M. Task Type Vocabulary
The CLI example uses "classification" and "regression." Mutual information has separate classif/regression variants. Are multilabel, multi class, or time series tasks in scope? What happens when an unrecognized task type string is passed?
Solution: Only `classification` and `regression` are valid. An unrecognized value raises `ValueError` at startup before any tool runs.

### N. Eight-Worker Global Cap Enforcement
"Hard cap: eight workers across all thread pools at any time." Multiple ThreadPoolExecutors are instantiated independently (profiler, scaler, selector). How is the global cap enforced — a shared semaphore passed into each executor, a single top level executor, or a per pool limit of eight?
Solution: Use Ray. `ray.init(num_cpus=8)` is called once at orchestrator startup and enforces the global cap. All parallel work is defined as `@ray.remote` functions in `pipeline/parallel.py`. Tools call `ray.get([fn.remote(item) for item in items])` — no executor object, no semaphore, no per-tool configuration needed.

### O. config.yaml Schema Location
"The config is validated against a schema at startup." Where is the schema defined — inline as a Pydantic model in Python, a separate YAML/JSON schema file, or a manual key existence check? A missing schema file or class is as dangerous as a missing config key.
Solution: Define a Pydantic model `ConfigSchema` in `pipeline/config.py`. At startup, `config.yaml` is loaded with pyyaml and passed as `ConfigSchema(**raw_dict)`. A `ValidationError` at import time surfaces missing or wrong-typed keys immediately, before any tool runs. The Pydantic model is the single source of truth — no separate schema file.

### P. BaseTool Precondition / Postcondition Contract
BaseTool is described as an ABC with precondition and postcondition checking but its interface is never specified. What methods must subclasses implement? What do conditions check — presence of required state keys, absence of NaNs, row count bounds?
Solution: BaseTool has two abstract methods: `precondition(state)` checks that required state fields are not None before the tool runs, and `postcondition(state)` checks the tool's own output is not None after it runs. The base `__call__` calls precondition → run → postcondition in order.

### Q. report.md Template Fallback
"Falls back to a template if the model call fails." Is the template hardcoded in reporter.py, stored in config.yaml, or a separate file? What sections does it include — headings only, placeholder text, or the raw structured summary the model would have received?
Solution: Template is a string constant in `reporter.py`. It dumps the structured pipeline summary dict as-is under fixed section headings. No separate file needed.

### R. feature_artifact.json Schema
The artifact contents are described in prose but no JSON schema, Pydantic model, or example object is provided. Downstream consumers (transform replay, the serving pipeline) cannot be implemented without a concrete field level contract.
Solution:
```json
{
  "run_id": "20260609T143022_a3f1b2c4",
  "task": "classification",
  "target_column": "churn",
  "dropped_columns": ["id", "col_with_high_nulls"],
  "created_columns": [{"name": "col1_div_col2", "operation": "ratio", "sources": ["col1", "col2"]}],
  "transformers": [
    {"step": "imputation", "column": "col1", "strategy": "median", "fill_value": 3.5},
    {"step": "encoding",   "column": "col2", "strategy": "label",  "classes": ["a","b","c"]},
    {"step": "scaling",    "column": "col3", "strategy": "standard","mean": 0.5, "std": 1.2}
  ],
  "selected_columns": ["col1", "col3", "col1_div_col2"],
  "selection_method": "llm_select",
  "warnings": ["col_x had 52% nulls and was dropped"]
}
```

`selection_method` is one of `llm_select`, `llm_select+pca(N)`, `fallback:mrmr_all`, or `none` — see §3 Feature Selection.

### S. KNNImputer Determinism
"Every stochastic operation takes random_state=42." KNN Imputer (sklearn) does not accept a random_state parameter it is deterministic by algorithm but this fact is not documented. IterativeImputer does accept random_state. The spec should clarify which imputers are stochastic and how determinism is guaranteed for each.
Solution: KNNImputer is deterministic by algorithm — no seed needed. IterativeImputer is stochastic and gets `random_state=42`. Only IterativeImputer requires the seed; all others are inherently deterministic.

### T. IsolationForest Flag and Remove vs. Flag Only
The outlier method list names "IsolationForest flag and remove" as a single action. The application section separately describes a "flag and keep" outcome for outliers that carry signal. Is "flag only without removal" a valid agent action distinct from "flag and remove"? If so, it is missing from the algorithm list in §3.
Solution: IsolationForest is a detector only. After detection the agent picks one of the three actions already defined in C (scale, flag, drop_row). Rename the entry in §3 to "IsolationForest (detector)" — the action is always separate from the detection method.

### U. run_id Generation
Output paths use `<run_id>` but the generation strategy is not defined timestamp, UUID4, hash of inputs, or caller supplied. Callers cannot predict or reference run output paths without knowing this.
Solution: `run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]`. Example: `20260609T143022_a3f1b2c4`. Generated once at orchestrator init, stored in `PipelineState`, and returned to the caller alongside the output paths.

### V. `evidence_cited` form: dotted vs indexed
The validator membership-checks `evidence_cited` against the EvidencePacket field names returned by `render()`. The renderer emits dotted paths (`columns.dtype`). When the EvidencePacket contains a list-of-dataclasses field (`columns: list[ColumnTypeEvidence]`), the serialized JSON the model sees is an array, and models naturally cite array entries by index (`columns[3].dtype`). The whitelist contains only `columns.dtype`, so `columns[3].dtype` fails the check and every batched response gets rejected with `failures=['evidence']`.

Solution: both forms are accepted. The validator normalizes each `evidence_cited` entry by stripping `[<int>]` brackets before whitelist membership (`re.sub(r"\[\d+\]", "", cited)`). The whitelist itself stays dotted; the equivalence is enforced at the matcher in `responses.py::_field_known`, not at the renderer. Prompts must not advertise either form — the model picks whichever is natural.

### W. Numeric columns arriving as object dtype
Many real datasets encode missing-numeric values as the string `"NA"` in object-dtype columns (House Prices `LotFrontage`, `MasVnrArea`, `GarageYrBlt` all do this). After `read_csv(keep_default_na=False, na_values=[""])` — required by §4 to preserve `"NA"` as a categorical label — those columns reach SemanticTypeInfer as object dtype mixing numeric strings and `"NA"`. SemanticTypeInfer correctly types them `numeric`, but `df[col].isna()` is False everywhere because `"NA"` is a string, so MissingValueHandler skips them and downstream tools see object-typed data they cannot scale.

Solution: see §4 "Numeric placeholder normalization". After SemanticTypeInfer returns, the orchestrator runs a single normalization pass — for every column with assigned semantic type `numeric` whose pandas dtype is not numeric, `pd.to_numeric(col, errors='coerce')` is applied in place. `"NA"` tokens become real `NaN` and reach `MissingValueHandler` as nulls. The pass runs once, before any other tool sees the dataframe, and is gated on the assigned semantic type so categorical `"NA"` is untouched.

### X. Validator coercion path
`FeatureValidator._try_coerce` attempts float → datetime → label-encoding in order. The label-encoding leg silently replaces non-coercible columns with integer codes, hiding upstream typing or normalization bugs.

Solution: see §5 "FeatureValidator coercion is stricter". The validator tries float → datetime only. If both fail the column is dropped from `state.selected_columns` and a warning is logged. LabelEncoding inside the validator is removed — encoding belongs to the Encoder, not the validator's fallback path.

### Y. Smoke-test scope
The startup smoke test checks only that the model returns non-empty text. A model that returns reasoning-channel content only, or that ignores the response shape entirely, would still pass — and the cascading failures would only surface after an hour-long pipeline run.

Solution: see §5 "Startup smoke test". The orchestrator sends one structured prompt (a minimal EvidencePacket plus the matching response-shape header) through the same `_make_model_call` path the tools use, and parses the response via `validate_response` with relaxed thresholds (`min_rationale_chars=1`, `min_alternatives=0`, denylist empty). The smoke check probes JSON-shape, field-citation, and transport — not response quality. `failures=['parse']` aborts startup with the raw response attached; other content failures are logged only.

### Z. Tool idempotency under re-dispatch
A future agentic-mode orchestrator may call any tool more than once if its decision loop re-enters a state. Tool side effects (`state.transformers.append(...)`, `state.dropped_columns.extend(...)`) would duplicate, polluting the artifact.

Solution: see §5 "Tool idempotency". The fixed-sequence orchestrator runs each step exactly once, so re-entry does not occur in practice; the predicates below are still declared so the ADK-wrapper layer can short-circuit safely if it is ever wired up. Per-tool predicates: `SemanticTypeInfer → state.column_types is not None`, `MissingValueHandler → no nulls in non-categorical columns`, `OutlierHandler → state.row_count_after_outlier is not None`, `Encoder → state.pre_encoding_done and no object-dtype columns left`, `FeatureCreator.run_pre → state.pre_encoding_done`, `FeatureCreator.run_post → all post-encoding specs executed`, `Scaler → all numeric feature columns are float and not in outlier_scaled set`, `FeatureSelector → state.selected_columns is not None`, `FeatureValidator → target column is last and df is all float`, `FeatureReporter → report.md exists`. Checks live alongside the wrappers in `adk_tools.py`.

### AA. Deterministic component naming
FeatureSelector's `_pca` names new components via `abs(hash(tuple(X.columns))) % 10_000`. Python's `hash` is salted per process. Re-running produces different column names; `feature_artifact.json` becomes unreplayable across processes.

Solution: see §5 "Deterministic naming". The Python built-in `hash` is forbidden for any name that lands in `state.df` or in the artifact. Use `hashlib.md5("|".join(sources).encode()).hexdigest()[:8]` for hashed identifiers. PCA component names take the form `pca_<md5_8>_<i>` where `md5_8` is the deterministic hash of the source columns and `i` is the component index. Any future tool that synthesizes column names must use the same construction.

### BB. Lazy-batch degeneracy minimum size
`validate_response`'s degeneracy check (joint strategy tuple) currently fires only when the batch has at least 3 items. The floor is hardcoded.

Solution: promote the floor to `validation.lazy_min_batch_size` in config (default 3). `validate_response` reads it. Batches smaller than the floor skip the degeneracy check entirely — the "share-one-strategy" signal is meaningless at that scale and would produce false positives on a 1- or 2-column EvidencePacket.

### CC. Outlier `detector` field when action=scale
The Pydantic `OutlierDecision` schema requires a detector for every decision, but when the action is `scale` the detector mask is computed and immediately discarded — the whole column is RobustScaled regardless. The model is asked to pick a detector with no consequence.

Solution: see §4 "Outlier decisions". `OutlierDecision.detector` becomes `Literal["iqr","zscore","isolation_forest"] | None`. The prompt instructs the model to omit `detector` when picking `scale`. `flag` and `drop_row` still require a detector (the mask is what they act on). The detector-tuple component of the lazy-batch joint-strategy check uses `(detector or "n/a", action)` so a batch of all `(None, scale)` decisions is still caught by the degeneracy check.
