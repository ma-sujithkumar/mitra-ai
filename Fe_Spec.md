# SPEC: Feature Engineering Agent

---

## 1. GOAL

A pipeline that takes a raw tabular dataset and produces ML ready features. It handles missing values, encodes categoricals, creates new features, selects the best ones, and hands back a transformed dataset along with a JSON file that records every transform applied. That JSON is what gets used when the pipeline runs on new data given a raw row and the file, you can reproduce the exact output without rerunning the pipeline.

The model is only called where a human looking at column names and sample values would need to stop and think. Everything else is code.

---

## 2. PACKAGES TO BE USED

- google-adk (agent harness — owns all model calls and tool dispatch)
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

**Bring-your-own model.** The pipeline is model-provider-agnostic. The caller supplies a model string (e.g., `gemini/gemini-2.0-flash`, `openai/gpt-4o`) and sets the corresponding API key as an environment variable. ADK resolves the provider from the model string. No model client library is bundled — the caller installs whatever their provider requires.

---

## 3. LIST OF ALGORITHMS FOR EACH MODALITY

*[code] = no model call. [agent] = model is called to make the decision.*

### Imputation (4) — [code] execution, [agent] picks strategy per column
1. Median fill
2. Mode fill
3. KNNImputer
4. IterativeImputer (MICE)

### Outlier Handling — [code] execution, [agent] picks detector and action per column

Detectors:
5. IQR
6. Z-score
7. IsolationForest (detector only)

Actions (agent picks one per column):
- scale     → apply RobustScaler to the column (default for numeric)
- flag      → add binary column `<col>_is_outlier`, keep row intact
- drop_row  → drop row and corresponding target row atomically (explicit opt-in only)

### Categorical Encoding (1) — [code]
8. LabelEncoder

### Datetime Decomposition — [code]
Detection: SemanticTypeInfer assigns type `datetime`; that column is routed here.
9. Year
10. Month
11. Day of month
12. Quarter
Original datetime column is dropped after decomposition.

### Feature Creation Operations (15) — [agent] proposes which to apply, [code] executes
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

### Scaling (4) — [code] execution, [agent] picks scaler per column
28. StandardScaler
29. RobustScaler
30. MinMaxScaler
31. PowerTransformer (Yeo-Johnson)

### Feature Selection (8) — [code] execution, [agent] picks which method to run
32. VarianceThreshold
33. Mutual information (classification / regression)
34. ANOVA F-test
35. Lasso path
36. RandomForest importance
37. PCA
38. mRMR

### Type Inference — [agent]
The model is called once with all column names, dtypes, null rates, and sample values to assign a type to every column. This is the only step where the model sees the data, and even then only as summary statistics, not raw rows.

Valid types: `numeric`, `categorical`, `datetime`, `id`, `text`, `binary`, `target`. Columns assigned type `id` are dropped automatically before any transformation runs.

### Reporting — [agent]
The model is called once at the end to write the report from the structured pipeline summary. If no model client is provided this step produces a template report instead.

---

## 4. APPLICATION

**How it gets called.** The caller passes a dataset path, task type, target column name, and a model string. The corresponding API key must be set as an environment variable before the pipeline starts. The pipeline runs end to end and returns the engineered dataset and the artifact file. No other configuration is needed.

**Imputation decisions.** The model looks at the null pattern, the column's correlation with other features, and the null rate before picking a strategy. A column with random low rate nulls gets median. A column where nulls correlate with other columns gets KNN because the surrounding structure carries signal. A column with more than 50% nulls is dropped (`null_drop_threshold: 0.50` in config.yaml). Whether missingness is random or tied to other variables (MCAR vs MAR vs MNAR) changes the right answer, and a fixed rule can't detect that — the model reads the profile and decides.

**Outlier decisions.** The agent picks a detector (IQR, Z-score, or IsolationForest) and then picks an action separately. A column with a few values ten standard deviations out is probably a data entry error — use drop_row. A column where outliers follow a pattern relative to the target might carry signal — use flag. A tree model tolerates outliers; a linear model does not — use scale for numeric columns going into linear models. The three valid actions are scale, flag, and drop_row as defined in §3.

**Feature selection decisions.** After profiling, the model looks at task type, number of features, dataset size, and correlation structure and picks the right method. Small linear datasets get Lasso. Non-linear patterns get tree importance. If features are collectively redundant and individually weak (high inter-correlation, low MI), apply PCA. If features are individually significant (high MI, low inter-correlation), apply mRMR. If both conditions partially hold, apply mRMR on significant features first, then PCA on the residual redundant block. Default fallback: mRMR.

**Scaling decisions.** The model picks the scaler per column from the distribution profile. Near normal columns get StandardScaler. Skewed columns or those with outliers get RobustScaler. Bounded columns get MinMaxScaler. Heavily skewed columns get PowerTransformer. Skewness, kurtosis, and outlier rates are already in the profile — the model reads them and decides.

**Target column handling.** The target is separated from features at ingestion. It goes through imputation if it has missing values. If the target is categorical (classification with string labels) it is label encoded. It is never scaled or included in feature selection. Feature creation uses it as a reference for MI scoring only. It is always the last column in the output.

**Task type.** Valid values are `classification` and `regression`. Mutual information and other task-sensitive methods switch variant accordingly. An unrecognized task type raises `ValueError` at startup before any tool runs.

**Parallel execution.** Univariate analysis (per column stats null rates, distribution shape, skewness, top categories) and multivariate analysis (pairwise correlations, mutual information between columns) are independent and run at the same time inside DataProfiler. Feature selection scoring methods also run in parallel across three threads with scores combined after. Scaling runs across four column batches in parallel.

---

## 5. CONSTRAINTS

**Agent harness: Google ADK.** The orchestrator is a `google.adk.agents.Agent`. The caller supplies a model string; the pipeline is model-provider-agnostic. ADK resolves the provider and handles all model calls. There is no `model_fn` callable anywhere in the codebase. A model string is required — the pipeline refuses to start without one. There is no offline fallback mode.

Each pipeline tool (DataProfiler, SemanticTypeInfer, etc.) is wrapped as a plain Python function and registered on the orchestrator agent as an ADK tool. Tool functions are stateless from ADK's perspective — they close over a shared `PipelineState` instance that is set once at startup. ADK tools never receive `PipelineState` as an argument.

The model is invoked by the ADK orchestrator agent in three decision points. Each works differently.

SemanticTypeInfer sends all column names, dtypes, null rates, and sample values in one call and gets back a JSON array of type assignments. One call, one response.

FeatureCreator sends column stats and gets back a JSON list of operation specs. The model makes one pass of suggestions and code handles execution from there.

FeatureReporter sends a structured summary of pipeline decisions and gets back a Markdown report. No JSON, no validation, just text.

The orchestrator agent runs ADK's agent loop — it calls a tool, receives the result, and decides what to call next. ADK manages this loop internally. The number of model calls depends on how many tools return errors; in a clean run it is one call per decision point listed above.

Nothing the model outputs is applied to data directly. Type assignments and operation specs both go through a validation step before anything touches the dataframe. On validation failure the pipeline attempts coercion in order: cast to float, then parse as timestamp, then apply LabelEncoder. If all three fail the operation is skipped and a warning is logged. The model decides what to do; code does it.

`cross_categorical` operations carry `temporal_class: pre_encoding` and execute before the Encoder runs. Any FeatureCreator operation spec missing a `temporal_class` is rejected at plan-validation time.

Parallelism uses Ray. Ray is initialised once at orchestrator startup via `ray.init(num_cpus=8)` which enforces the global cap of eight workers. Ray uses shared memory for numpy arrays and pandas DataFrames (zero-copy via Apache Arrow) so there is no serialisation overhead for sklearn objects. All parallel work is expressed as `@ray.remote` functions in `pipeline/parallel.py`. No tool initialises Ray or sets worker counts — that is the orchestrator's responsibility at startup.

Every stochastic operation takes `random_state=42`. Same inputs must produce the same outputs on every run. KNNImputer is deterministic by algorithm and does not accept a seed. IterativeImputer is the only imputer that requires `random_state=42`.

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
    orchestrator.py             FeatureEngineerOrchestrator; constructs ADK Agent, sets pipeline state, runs ADK Runner
    tools/
        __init__.py
        adk_tools.py            ADK tool functions wrapping each BaseTool; set_pipeline_state() injection point
        profiler.py             DataProfiler
        infer.py                SemanticTypeInfer
        imputer.py              MissingValueHandler
        outlier.py              OutlierHandler
        encoder.py              Encoder
        creator.py              FeatureCreator
        scaler.py               Scaler
        selector.py             FeatureSelector
        validator.py            FeatureValidator
        reporter.py             FeatureReporter

schema.md                       full input/output contract, all args documented
README.md                       entry point for any caller, links schema.md
main.py                         CLI: `python main.py run data.csv --task classification --target churn --model <model_string>`
```

**What a run produces.**

`pipeline_output/<run_id>/engineered_dataset.csv` — the transformed dataset, all features numeric, scaled, and selected, ready to pass to a model.

`pipeline_output/<run_id>/feature_artifact.json` — full record of every transform applied. Schema:

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
  "selection_method": "mRMR",
  "warnings": ["col_x had 52% nulls and was dropped"]
}
```

Use this file to replay the same transforms on new data without re-running the pipeline.

`pipeline_output/<run_id>/report.md` — a report covering data quality findings, encoding decisions, features created, selection results, and recommendations. Written by the model from a structured summary. Falls back to a string constant template in `reporter.py` that dumps the structured summary dict under fixed section headings.

`pipeline_output/<run_id>/execution_log.txt` — append only log of every tool that ran, what it did, and how long it took.

`run_id` is generated as `YYYYMMDDTHHMMSS_<8-char uuid>` (e.g. `20260609T143022_a3f1b2c4`) at orchestrator init, stored in `PipelineState`, and returned to the caller alongside the output paths.

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

### E. Orchestrator Loop vs. Fixed Pipeline Order
The tools list in §6 implies a fixed sequence (profiler → infer → imputer → outlier → encoder → creator → scaler → selector → validator → reporter). The orchestrator description says it "picks a tool, runs it, looks at the result, and decides what to do next." Can tools run out of order? Can any tool re run? What conditions trigger a deviation from the default sequence?
Solution : 1. Orchestrator dispatches based on input satisfaction, not fixed sequence. Default order is a recommendation, not a constraint.
           2. Any tool may rerun up to 5 times if validator identifies a correctable failure in its output. Third failure escalates to Judge Agent.
           3. The orchestrator decides tool execution order at runtime. When multiple tools are queued and their inputs are all satisfied, the orchestrator dispatches them in parallel. Sequential execution applies only when a tool is waiting on another tool's output.

### F. FeatureCreator Output Cap
No upper bound is defined on the number of operations the model may propose. Twenty source columns can produce hundreds of pairwise ratios, products, and differences. Without a cap, feature count can explode before selection runs. Should config.yaml hold a max_created_features key?
Solution: The Judge Agent decides on the best set of features based on ranking for feature selection.

### G. Validation Failure Behavior
"Type assignments and operation specs both go through a validation step before anything touches the dataframe." What happens on failure — is the operation silently skipped, is an exception raised, or is the model retried with an error message? The validator.py tool listed in §6 is never described.
Solution: it should be try eccept block first try to convert to float next try time stamp except label encode.
### H. Target Column Handling
The target column is never mentioned in the context of transformations. Is it excluded from imputation, encoding, scaling, and selection automatically? Is it passed through unchanged? Does its presence affect feature creation (e.g., should features be created relative to the target)?
Solution: Target is separated from features at ingestion. It goes through imputation if it has missing values. If the target is categorical (classification with string labels) it must be label encoded. It is never scaled or included in feature selection. Feature creation uses it as a reference for MI scoring only. It is always passed through to the output as the last column.

### I. LabelEncoder Only / Categorical Coverage
Only LabelEncoder is listed. There is no handling defined for high cardinality categoricals, ordinal vs. nominal distinction, or binary columns. Is OrdinalEncoder, TargetEncoder, or frequency encoding out of scope? What is the maximum cardinality before a column is treated differently?
Solution: only label encoder is in scope

### J. model_fn Offline Fallbacks
"All steps that need a model fall back to their fixed defaults." The concrete fallback for each agent step is not defined. SemanticTypeInfer — infer from pandas dtype only? FeatureCreator skip feature creation entirely or apply a minimal fixed set? Orchestrator — run tools in the fixed sequential order from section 6?
Solution: no fallback needed at this stage the execution is done only if the agents clear the smoke test.

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
  "selection_method": "mRMR",
  "warnings": ["col_x had 52% nulls and was dropped"]
}
```

### S. KNNImputer Determinism
"Every stochastic operation takes random_state=42." KNN Imputer (sklearn) does not accept a random_state parameter it is deterministic by algorithm but this fact is not documented. IterativeImputer does accept random_state. The spec should clarify which imputers are stochastic and how determinism is guaranteed for each.
Solution: KNNImputer is deterministic by algorithm — no seed needed. IterativeImputer is stochastic and gets `random_state=42`. Only IterativeImputer requires the seed; all others are inherently deterministic.

### T. IsolationForest Flag and Remove vs. Flag Only
The outlier method list names "IsolationForest flag and remove" as a single action. The application section separately describes a "flag and keep" outcome for outliers that carry signal. Is "flag only without removal" a valid agent action distinct from "flag and remove"? If so, it is missing from the algorithm list in §3.
Solution: IsolationForest is a detector only. After detection the agent picks one of the three actions already defined in C (scale, flag, drop_row). Rename the entry in §3 to "IsolationForest (detector)" — the action is always separate from the detection method.

### U. run_id Generation
Output paths use `<run_id>` but the generation strategy is not defined timestamp, UUID4, hash of inputs, or caller supplied. Callers cannot predict or reference run output paths without knowing this.
Solution: `run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]`. Example: `20260609T143022_a3f1b2c4`. Generated once at orchestrator init, stored in `PipelineState`, and returned to the caller alongside the output paths.
