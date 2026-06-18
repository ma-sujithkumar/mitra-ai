# Implementation Plan: Feature Engineering Agent

---

## Build Order

Four phases. Each phase only starts when the previous is complete. Later phases depend on earlier ones structurally.

```
Phase 1 — Foundation       config.yaml, config.py, state.py, base.py, parallel.py,
                           evidence.py, responses.py
Phase 2 — Tools            all 10 tools in pipeline/tools/ (each also exposed as an ADK tool function)
                           plus the Judge sub-agent in pipeline/judge_agent.py
Phase 3 — Orchestrator     orchestrator.py — ADK Agent + ADK tool registration + Judge construction
Phase 4 — Entry point      main.py (ADK Runner), schema.md, README.md
```

**Agent harness: Google ADK.** The orchestrator is a `google.adk.agents.Agent`. Each pipeline tool (BaseTool subclass) is wrapped as an ADK tool function and registered on the orchestrator agent. ADK's agent loop handles all model calls and tool dispatch. There is no manual `model_fn` callable anywhere in the codebase.

---

## Phase 1: Foundation

### `config/config.yaml`
All thresholds and parameters. No values hardcoded anywhere else. Keys needed:

```yaml
imputation:
  null_drop_threshold: 0.50
  knn_neighbors: 5
  iterative_max_iter: 10
  categorical_null_literals:           # strings preserved as category values, NOT treated as nulls
    - "None"
    - "NA"
    - "N/A"
    - "na"
    - "n/a"
    - "none"
    - "NaN"

outlier:
  iqr_multiplier: 1.5
  zscore_threshold: 3.0
  isolation_contamination: 0.05
  default_action: scale

feature_creation:
  max_created_features: 50
  equal_width_bins: 5
  quantile_bins: 5

feature_selection:
  correlation_threshold: 0.90
  mi_threshold: 0.01
  variance_threshold: 0.01
  lasso_alpha: 0.01
  rf_n_estimators: 100
  pca_variance_retained: 0.95
  top_k_features: 20
  cluster_cut_threshold: 0.30         # 1 - |corr| distance cut for average-linkage hierarchical clustering
  linear_baseline_k: 10               # number of top-MI features used to fit the linear baseline score

scaling:
  power_transformer_method: yeo-johnson

pipeline:
  max_tool_retries: 3
  random_state: 42
  max_workers: 8
  task_infer_nunique_threshold: 20   # if --task omitted: target with >20 unique values → regression, else classification
  downstream_model_hint: tree        # "linear" | "tree" — supplied to OutlierHandlerEvidence

validation:
  min_rationale_chars: 60            # responses shorter than this are rejected
  min_alternatives: 2                # `alternatives_considered` must have at least this many entries
  lazy_response_threshold: 0.80      # batched response is degenerate if > this fraction share one strategy
  lazy_min_batch_size: 3             # degeneracy check skipped on batches smaller than this
  raw_log_max_chars: 60000           # per-attempt cap in raw_responses.txt; longer bodies are truncated
  boilerplate_denylist:              # case-insensitive substring matches; matching rationales are rejected
    - "based on the data"
    - "because it is appropriate"
    - "it is the best choice"
    - "looks suitable"
    - "as per the guidance"

llm:
  max_tokens: 2048                    # passed to ADK GenerateContentConfig on every model call
  api_key_env_var: GOOGLE_API_KEY     # name of the env var the orchestrator injects the key into before ADK init
  api_key: your_actual_key_here       # actual API key value; orchestrator copies into os.environ[api_key_env_var] at startup
```

---

### `pipeline/config.py`
- Pydantic model `ConfigSchema` mirroring every key above with types and defaults.
- `PipelineSettings` includes `task_infer_nunique_threshold: int` and `downstream_model_hint: Literal["linear","tree"]`.
- `LlmConfig` sub-model: `max_tokens: int`, `api_key_env_var: str`, `api_key: str` (non-empty).
- New `ValidationSettings` sub-model: `min_rationale_chars: int`, `min_alternatives: int`, `lazy_response_threshold: float` (∈ [0,1]), `lazy_min_batch_size: int` (≥1), `raw_log_max_chars: int` (≥1024), `boilerplate_denylist: list[str]`.
- `FeatureSelectionSettings` includes `cluster_cut_threshold: float` and `linear_baseline_k: int`.
- `ImputationConfig` includes `categorical_null_literals: list[str]` — string tokens preserved as category values for categorical/binary columns (spec §4 "Null detection in categoricals").
- Module-level `load_config(path) -> ConfigSchema` — loads yaml, validates, returns.
- Any missing key raises `ValidationError` at import time.

---

### `pipeline/state.py`
Single `PipelineState` dataclass. All tools read from and write to this object. Fields populated progressively as pipeline runs.

```python
@dataclass
class PipelineState:
    # inputs
    df: pd.DataFrame
    target: pd.Series
    task: str                        # "classification" | "regression"
    target_column: str
    run_id: str
    config: ConfigSchema
    # no model_fn — ADK agent manages all model calls

    # populated by profiler
    profile: dict | None = None      # per-column stats

    # populated by infer
    column_types: dict | None = None # col -> type string

    # populated by each tool (append-only lists)
    transformers: list = field(default_factory=list)
    dropped_columns: list = field(default_factory=list)
    created_columns: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # populated by selector
    selected_columns: list | None = None
    selection_method: str | None = None

    # output path
    output_dir: Path | None = None

    # orchestrator sets after creator.run_pre completes
    pre_encoding_done: bool = False

    # orchestrator sets after all drop_row actions complete
    row_count_after_outlier: int | None = None
```

---

### `pipeline/base.py`
Abstract base class all tools inherit from.

```python
class BaseTool(ABC):
    @abstractmethod
    def precondition(self, state: PipelineState) -> None:
        # raise PreconditionError if required state fields are None

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        ...

    @abstractmethod
    def postcondition(self, state: PipelineState) -> None:
        # raise PostconditionError if own output field is None

    def __call__(self, state: PipelineState) -> PipelineState:
        self.precondition(state)
        state = self.run(state)
        self.postcondition(state)
        return state
```

Two exception classes: `PreconditionError`, `PostconditionError`.

---

### `pipeline/parallel.py`
- Defines `@ray.remote` decorated versions of compute-heavy functions used by tools.
- Helper `run_parallel(remote_fn, items) -> list` — calls `ray.get([remote_fn.remote(item) for item in items])`.
- Ray is not initialised here. `ray.init(num_cpus=8)` is called once by the orchestrator at startup.
- No tool calls `ray.init` or sets worker limits.
- Also exposes `compute_mini_profile(df, cols) -> dict` (shared utility used by SemanticTypeInfer's datetime decomposition and FeatureCreator's `run_post`).
- Exposes `compute_correlation_clusters(X, cut_threshold) -> dict[int, list[str]]` (average-linkage hierarchical clustering on `1 - |corr|`, returns `cluster_id -> [columns]`). Used by `FeatureSelectorEvidence`.
- Exposes `compute_linear_baseline(X, y, task, k) -> float` (fits LogisticRegression / LinearRegression on the top-`k` MI features, returns CV-AUC for classification or CV-R² for regression). Used by `FeatureSelectorEvidence`.

---

### `pipeline/evidence.py`
Typed dataclasses for every decision point. One dataclass per LLM call site. Tools must build these — no ad-hoc f-string prompts over the profile dict.

```python
@dataclass
class SemanticTypeInferEvidence:
    columns: list[ColumnTypeEvidence]   # per-column packet

@dataclass
class ColumnTypeEvidence:
    name: str
    dtype: str
    null_rate: float
    nunique: int
    top_values: list[str]               # up to 5
    random_samples: list[str]           # 5 string-cast values
    regex_signature: dict[str, int]     # {"uuid": n, "email": n, "iso_date": n, "phone": n, "numeric_string": n}

@dataclass
class MissingValueEvidence:
    columns: list[NullColumnEvidence]

@dataclass
class NullColumnEvidence:
    name: str
    null_rate: float
    null_run_lengths: list[int]                       # histogram of consecutive-null streaks
    null_mask_corr_top5: dict[str, float]
    target_rate_when_null: float | None
    target_rate_when_present: float | None
    random_present_values: list[str]                  # 10
    dtype: str
    semantic_type: str

@dataclass
class OutlierEvidence:
    columns: list[OutlierColumnEvidence]
    downstream_model_hint: str                        # "linear" | "tree", from config

@dataclass
class OutlierColumnEvidence:
    name: str
    histogram_10bin: list[int]
    extreme_top5: list[tuple[float, float | str]]     # (value, target_at_that_row)
    extreme_bottom5: list[tuple[float, float | str]]
    mi_with_target: float
    target_corr: float

@dataclass
class ScalerEvidence:
    columns: list[ScalerColumnEvidence]

@dataclass
class ScalerColumnEvidence:
    name: str
    histogram_20bin: list[int]
    skewness: float
    kurtosis: float
    outlier_rate: float
    bounded: bool
    bounds: tuple[float, float] | None
    monotonic_with_target: float                      # Spearman

@dataclass
class FeatureCreatorEvidence:
    columns: list[CreatorColumnEvidence]
    co_occurring_pairs: list[tuple[str, str, float]]  # (col_a, col_b, joint_mi)

@dataclass
class CreatorColumnEvidence:
    name: str
    semantic_type: str
    mi_with_target: float
    nunique: int
    correlated_with_top3: dict[str, float]
    decomposed_from: str | None

@dataclass
class FeatureSelectorEvidence:
    n_rows: int
    n_features: int
    task: str
    linear_baseline_score: float                      # CV-AUC or CV-R² on top-k MI
    clusters: list[ClusterEvidence]

@dataclass
class ClusterEvidence:
    cluster_id: int
    members: list[str]
    mean_mi: float
    max_mi: float
    intra_cluster_corr: float
```

A single serializer in `evidence.py` renders each dataclass to a deterministic prompt block. The dataclass *is* the contract: adding a field requires changing the dataclass plus a config entry, never a prompt edit.

---

### `pipeline/responses.py`
Pydantic models for every model response. Each per-item decision carries `strategy`, `rationale`, `evidence_cited`, `alternatives_considered`. Parsing failure → coercion path (§7-G). Content failure → revision path (§5).

```python
class DecisionItem(BaseModel):
    rationale: str = Field(min_length=1)             # actual length checked against config.validation.min_rationale_chars
    evidence_cited: list[str]                        # must be non-empty and a subset of the EvidencePacket field names sent
    alternatives_considered: list[str]               # must have >= config.validation.min_alternatives entries

class TypeAssignment(DecisionItem):
    column: str
    type: Literal["numeric","categorical","datetime","id","text","binary","target"]

class SemanticTypeInferResponse(BaseModel):
    assignments: list[TypeAssignment]

class ImputationDecision(DecisionItem):
    column: str
    strategy: Literal["median","mode","knn","iterative","drop"]

class MissingValueResponse(BaseModel):
    decisions: list[ImputationDecision]

class OutlierDecision(DecisionItem):
    column: str
    detector: Literal["iqr","zscore","isolation_forest"]
    action: Literal["scale","flag","drop_row"]

class OutlierResponse(BaseModel):
    decisions: list[OutlierDecision]

class ScalerDecision(DecisionItem):
    column: str
    scaler: Literal["standard","robust","minmax","power"]

class ScalerResponse(BaseModel):
    decisions: list[ScalerDecision]

class CreatorSpec(DecisionItem):
    operation: Literal[<the 15 VALID_OPS>]
    sources: list[str]
    name: str
    temporal_class: Literal["pre_encoding","post_encoding"]

class FeatureCreatorResponse(BaseModel):
    specs: list[CreatorSpec]

class ClusterAction(DecisionItem):
    cluster_id: int
    action: Literal["mrmr","pca","mrmr_then_pca","drop","lasso","rf_importance"]

class SelectionPlanResponse(BaseModel):
    plan: list[ClusterAction]
```

A single helper `validate_response(model_cls, raw_text, evidence_field_names, cfg) -> tuple[BaseModel | None, list[str]]` performs:
1. JSON extraction + Pydantic parse → on failure, return (None, ["parse"]). Extraction tries non-greedy first (`r"\{.*?\}"` with re.DOTALL), then falls back to greedy if non-greedy gives nothing parseable; this avoids grabbing an embedded JSON example from inside the rationale.
2. Field-level content checks: rationale length, `evidence_cited` non-empty + subset of provided fields, `alternatives_considered` count.
3. Batch-level degeneracy check: only if `len(items) >= cfg.validation.lazy_min_batch_size`. If >`lazy_response_threshold` of items share the same *strategy tuple* (the joint of every `Literal` field on the response item), returns `["lazy"]`.

`_field_known(cited, sent)` normalizes the cited string by stripping `[<int>]` brackets (`re.sub(r"\[\d+\]", "", cited)`) before whitelist membership. Models may cite either `columns.dtype` or `columns[3].dtype`; both pass. The whitelist itself stays in dotted form — the normalization is in the matcher only (spec §7-V).

Returns (parsed_model, failure_reasons). **All LLM-call sites — every tool and the Judge sub-agent — must use this helper; no tool may implement its own JSON-parse or content check.** Callers wire the (one-revision → fall-through) loop around the helper per §5.

---

## Phase 2: Tools

Build in this order — each tool depends on the state fields populated by tools before it.

```
profiler → infer → imputer → outlier → encoder → creator → scaler → selector → validator → reporter
(judge_agent.py is built alongside creator; it has no state dependency and is constructed once at orchestrator startup)
```

`cross_categorical` inside creator must run before encoder. The creator splits its work: pre-encoding operations first, then the rest after encoding. The orchestrator handles this split.

---

### `pipeline/tools/profiler.py` — `DataProfiler`

**Precondition:** `state.df` not None.
**Postcondition:** `state.profile` not None.

Runs univariate and multivariate analysis in parallel via Ray (`run_parallel` from `parallel.py`).

Univariate (per column, parallel):
- null rate, dtype, nunique, top 5 values
- mean, std, skewness, kurtosis (numeric only)
- value_counts for categoricals

Multivariate (single pass after univariate):
- pairwise Pearson correlation matrix → `state.profile["_correlation_matrix"]`
- mutual information between each feature and target → per-column `mi_with_target`
- correlation clusters via `compute_correlation_clusters` → `state.profile["_clusters"]` (used later by `FeatureSelectorEvidence`; selector re-clusters over its actual input dataframe, but the profiler's clusters seed the Creator's co-occurrence ranking)
- linear baseline score via `compute_linear_baseline` on the top-`linear_baseline_k` MI features → `state.profile["_linear_baseline_score"]`
- top column pairs by joint MI → `state.profile["_joint_mi_pairs"]` (consumed by `FeatureCreatorEvidence.co_occurring_pairs`; the profiler computes true joint MI on a sampled set of top-N candidate pairs and stores `(col_a, col_b, joint_mi)` tuples; creator reads from this key when present and falls back to the MI-product proxy only if the key is absent — gaps.txt #1, #5)

Output: populates `state.profile`. Also writes the four `_` -prefixed keys above. The `_`-prefix is reserved for top-level multivariate artifacts; tools must not iterate `state.profile` without filtering them out.

---

### `pipeline/tools/infer.py` — `SemanticTypeInfer`

**Precondition:** `state.profile` not None.
**Postcondition:** `state.column_types` not None.

Builds `SemanticTypeInferEvidence` from `state.profile`. Prompt template describes the seven types mechanically (what each one means) — it does **not** enumerate when to assign them. One model call. Response parsed via `validate_response(SemanticTypeInferResponse, ...)`. Content failures trigger one retry with `prior_response_was_uninformative=True` plus a delta-evidence pack contrasting columns that received the same type. Fall-through: dtype-based assignment per ambiguity #21.

Valid types: `numeric`, `categorical`, `datetime`, `id`, `text`, `binary`, `target`.

Post-call:
- Columns typed `id` → added to `state.dropped_columns`, removed from `state.df`.
- Columns typed `datetime` → routed to decomposition (handled in this tool).
- Columns typed `text` → dropped (out of scope).
- The caller-supplied `target_column` is forced to type `target` regardless of model output.

Datetime decomposition (code, no model): year, month, day, quarter. Original datetime column dropped.

---

### `pipeline/tools/imputer.py` — `MissingValueHandler`

**Precondition:** `state.column_types` not None.
**Postcondition:** zero nulls remain in all non-dropped columns.

Builds `MissingValueEvidence` from `state.profile` (null_rate, null_run_lengths, null_mask_corr_top5, target_rate_when_null/present, random_present_values, dtype, semantic_type). Prompt describes the five strategies mechanically and **omits any when-to-use guidance**. One model call → `validate_response(MissingValueResponse, ...)` → one retry on content failure → fall-through to median (numeric) / mode (other).

**Categorical/binary columns are excluded from null detection entirely** (spec §4 "Null detection in categoricals"). The `cols_with_nulls` scan only considers columns typed `numeric`, `datetime`, or `target`. Any string token from `config.imputation.categorical_null_literals` (`"None"`, `"NA"`, `"N/A"`, etc.) that appears in a categorical column is a legitimate category label, not a missing value. Because the orchestrator loads the CSV with `keep_default_na=False, na_values=[""]`, these tokens reach the imputer as plain strings and `Series.isna()` naturally returns False for them.

Strategies: `median`, `mode`, `knn`, `iterative`, `drop`.

- Columns with null rate > `null_drop_threshold` → `drop` strategy → added to `state.dropped_columns`.
- KNNImputer: deterministic, no seed.
- IterativeImputer: `random_state=42`.
- Each fitted imputer appended to `state.transformers` with fitted params.

Target column: imputed separately with mode (classification) or median (regression) before feature imputation runs.

---

### `pipeline/tools/outlier.py` — `OutlierHandler`

**Precondition:** `state.df` has no nulls (imputer ran).
**Postcondition:** `state.df` updated per chosen action.

Builds `OutlierEvidence` per numeric column (10-bin histogram, top-5 extremes with aligned target values, mi_with_target, target_corr) plus the global `downstream_model_hint` from `pipeline.downstream_model_hint`. One batched model call → `validate_response(OutlierResponse, ...)` → one retry on content failure → fall-through to `(iqr, scale)`.

Detectors: `iqr`, `zscore`, `isolation_forest` (`random_state=42`).
Actions: `scale`, `flag`, `drop_row`.

- `scale` → RobustScaler applied to column (fitted params logged to `state.transformers`).
- `flag` → new binary column `<col>_is_outlier` added to df.
- `drop_row` → row index collected across all `drop_row` columns, rows and corresponding target rows dropped atomically at the end.

---

### `pipeline/tools/encoder.py` — `Encoder`

**Precondition:** outlier handling complete.
**Postcondition:** no string-typed columns remain in `state.df`.

No model call. Code only.

- `cross_categorical` features (from FeatureCreator pre-encoding pass) already exist by this point.
- Columns typed `categorical` or `binary` → LabelEncoder.
- Target column typed `categorical` → LabelEncoder (classification only).
- Fitted classes logged to `state.transformers`.

---

### `pipeline/tools/creator.py` — `FeatureCreator`

**Precondition:** encoding complete (for `run_post`); see ambiguity #13 for `run_pre`.
**Postcondition:** new feature columns added to `state.df`.

Builds `FeatureCreatorEvidence` from `state.profile` (per-column semantic_type, mi_with_target, nunique, correlated_with_top3, decomposed_from, plus global co_occurring_pairs). Prompt describes each of the 15 operations mechanically and **does not enumerate when to use any of them**. One model call → `validate_response(FeatureCreatorResponse, ...)` → one retry on content failure → fall-through: skip creation.

After the parsed specs pass validation, they are sent to the **Judge Agent** for ranking and capping (§7-F). Judge returns a capped, ranked subset; falls back to proxy-MI ranking if Judge LLM is unavailable.

Each surviving spec must have `temporal_class`: `pre_encoding` or `post_encoding`. Specs missing `temporal_class` are rejected at Pydantic parse time.

`cross_categorical` specs → `temporal_class: pre_encoding` (executed before encoder runs in `run_pre`).
All other specs → `temporal_class: post_encoding`, executed in `run_post`.

Each created column appended to `state.created_columns`.

---

### `pipeline/tools/scaler.py` — `Scaler`

**Precondition:** feature creation complete.
**Postcondition:** all numeric feature columns are scaled floats.

Builds `ScalerEvidence` per numeric feature (20-bin histogram, skewness, kurtosis, outlier_rate, bounded + bounds, monotonic_with_target). Prompt describes each scaler mechanically — what it does to the column — and **omits when-to-use guidance**. One batched model call → `validate_response(ScalerResponse, ...)` → one retry on content failure → fall-through to `standard`.

Scalers: `standard`, `robust`, `minmax`, `power` (Yeo-Johnson, `random_state=42`).

- Target column: never scaled.
- Columns are batched and scaled in parallel via Ray (`run_parallel` from `parallel.py`).
- Each fitted scaler appended to `state.transformers`.

---

### `pipeline/tools/selector.py` — `FeatureSelector`

**Precondition:** scaling complete.
**Postcondition:** `state.selected_columns` not None.

Builds `FeatureSelectorEvidence` by re-running `compute_correlation_clusters` and `compute_linear_baseline` over the **current** `state.df` (which includes FeatureCreator-added columns), not from the Profiler's cached `_clusters` / `_linear_baseline_score`. The Profiler's values are stale at this point because the column set has grown. Per-cluster mean/max MI + intra-cluster correlation are computed from the fresh clusters; `n_rows`, `n_features`, `task` from current state.

**Method selection is delegated to the Judge Agent (§7-F).** The selector hands the EvidencePacket to Judge, which returns a `SelectionPlanResponse` — an ordered list of per-cluster actions. Code executes the plan cluster by cluster. No standalone selector model call.

Per-cluster actions: `mrmr`, `pca`, `mrmr_then_pca`, `drop`, `lasso`, `rf_importance`. Cluster execution runs in parallel via Ray. Each per-cluster action is wrapped as a `@ray.remote` function in `parallel.py` that takes the sub-dataframe and returns `(kept_column_names, new_column_arrays)`. `new_column_arrays` is non-empty for PCA (and any future action that synthesises columns). The selector calls `run_parallel(...)` over the cluster list, collects results, then materializes any new columns into `state.df` serially after `ray.get` returns — the central df is never mutated concurrently (gaps.txt #2).

PCA component naming is deterministic: `pca_<md5_8>_<i>` where `md5_8 = hashlib.md5("|".join(sources).encode()).hexdigest()[:8]` and `i` is the component index within the cluster. Python's built-in `hash` is forbidden — see spec §7-AA.

All sklearn seeds (`mutual_info_classif`, `mutual_info_regression`, `Lasso`, `LogisticRegression`, `RandomForest*`, `PCA`) read `cfg.pipeline.random_state` — no `random_state=42` literal.

Judge unavailable → fall-through: mRMR over all features with `top_k_features` from config.

Target column excluded from selection entirely. Populates `state.selected_columns` and `state.selection_method` (where `selection_method` is the serialized plan, e.g. `"plan:[c0:mrmr_then_pca,c1:pca,c2:drop]"`).

---

### `pipeline/tools/validator.py` — `FeatureValidator`

**Precondition:** selection complete.
**Postcondition:** `state.df` contains only `selected_columns` + target, all float64.

No model call. Code only. Checks:
- All selected columns present in df.
- No NaNs in selected columns.
- All selected columns are float64. Coercion is **strict**: float → datetime only. LabelEncoding inside the validator is **forbidden** — see spec §7-X. If both coercions fail the column is dropped from `state.selected_columns` and a warning is logged; the validator never silently relabels a non-numeric column.
- Target column is last column.
- Row count matches post-outlier-removal count.

---

### `pipeline/tools/reporter.py` — `FeatureReporter`

**Precondition:** validation complete.
**Postcondition:** `report.md` written to output dir.

Assembles structured pipeline summary from `state` (dropped columns, created columns, transformers, selected columns, warnings, selection method).

One `model_fn` call: sends summary, gets back Markdown report.

Fallback (if call fails): string constant template in `reporter.py` that serializes the summary dict under fixed headings: Data Quality, Encoding, Features Created, Feature Selection, Warnings.

Also writes `feature_artifact.json` and `execution_log.txt` to output dir.

---

### `pipeline/judge_agent.py` — `JudgeAgent`

Isolated ADK sub-agent (its own `Agent` + `InMemoryRunner`) so its context never enters the orchestrator's. Constructed once at orchestrator startup with the same model string, `api_key`, `base_url`, and `max_tokens`. Injected into FeatureCreator and FeatureSelector via constructor.

Two entry points:

- `rank(specs, profile, target_column, task, cap) -> tuple[list[dict], str]` — used by FeatureCreator. Builds the candidate prompt with proxy-MI per source, calls the LLM, parses through `validate_response(FeatureCreatorResponse, ...)`. Returns the capped, ranked subset plus a source tag (`"judge"` or `"fallback:proxy_mi"`).
- `plan(evidence: FeatureSelectorEvidence, cfg) -> SelectionPlanResponse | None` — used by FeatureSelector. Sends the cluster decomposition, parses through `validate_response(SelectionPlanResponse, ...)`. Returns `None` on hard failure; the selector then falls back to mRMR over all features.

Both entry points use exactly the same content-check + one-revision + fall-through contract as a tool LLM call. The retry budget is one per Judge invocation, not shared with the calling tool.

---

## Phase 3: Orchestrator

### `pipeline/orchestrator.py` — `FeatureEngineerOrchestrator`

The orchestrator is a **Google ADK `Agent`**. It owns ADK agent construction, startup checks, and run invocation. It does not implement a manual loop — ADK's runner handles tool dispatch.

**ADK Agent definition:** `ORCHESTRATOR_INSTRUCTION` is a template formatted at orchestrator init with `config.pipeline.max_tool_retries` so the retry budget the agent sees matches the config value (gaps.txt #3). No hardcoded `"3"` in the instruction string.

```python
from google.adk.agents import Agent

orchestrator_agent = Agent(
    name="feature_engineer_orchestrator",
    model="gemini-2.0-flash-exp",        # or caller-supplied model string
    description="Orchestrates the feature engineering pipeline.",
    instruction=ORCHESTRATOR_INSTRUCTION_TEMPLATE.format(
        max_tool_retries=cfg.pipeline.max_tool_retries
    ),
    tools=[
        profile_data,       # wraps DataProfiler
        infer_types,        # wraps SemanticTypeInfer
        handle_missing,     # wraps MissingValueHandler
        handle_outliers,    # wraps OutlierHandler
        create_features_pre,   # wraps FeatureCreator.run_pre
        encode_features,    # wraps Encoder
        create_features_post,  # wraps FeatureCreator.run_post
        scale_features,     # wraps Scaler
        select_features,    # wraps FeatureSelector
        validate_features,  # wraps FeatureValidator
        write_report,       # wraps FeatureReporter
    ],
)
```

**ADK tool functions** (in `pipeline/tools/adk_tools.py`): each is a plain Python function with a docstring that ADK uses as the tool description. Each function closes over the shared `PipelineState` instance for the current run. ADK tools receive only primitive/JSON-serializable arguments from the agent; they read and mutate `state` directly.

```python
def profile_data() -> dict:
    """Run DataProfiler on the current dataset. Returns summary stats."""
    DataProfiler()(state)
    return {"status": "ok", "columns": list(state.profile.keys())}
```

**Idempotency wrapper.** Every ADK tool wrapper checks a postcondition predicate at entry before running the underlying tool (spec §5 "Tool idempotency"). The check sits in a shared helper `_already_done(predicate_fn, state) -> dict | None`; when the predicate is satisfied, the wrapper returns `{"status": "ok", "detail": "already done"}` and does not invoke the underlying tool. Predicates per wrapper: `infer_types → state.column_types is not None`, `handle_missing → state.df.isna().sum().sum() == 0 and "imputation" present in state.transformers`, `handle_outliers → state.row_count_after_outlier is not None`, `create_features_pre → state.pre_encoding_done`, `encode_features → no non-numeric columns in state.df`, `create_features_post → creator._proposed_post fully executed`, `scale_features → all numeric feature columns in state.df not in outlier_scaled and scaled`, `select_features → state.selected_columns is not None`, `validate_features → state.df.columns[-1] == state.target_column and df.dtype all float64`, `write_report → (state.output_dir / "report.md").exists()`. Append-only state lists (`transformers`, `dropped_columns`, `created_columns`, `warnings`) are protected by the early return — they cannot grow on re-entry.

**Per-tool LLM source tag in execution log.** Tools that make a model call return their `call_with_revision` source tag (`ok`, `ok:revised`, or `fallback`) in their detail string, and the wrapper writes it into `execution_log.txt` alongside the elapsed time. A reader can scan the log and see at a glance whether any tool degenerated to its deterministic default (spec §6 "Observability detail").

**Inter-tool numeric-normalization step.** After `infer_types` succeeds, `adk_tools.handle_missing` first applies the §4 numeric-placeholder-normalization pass: for every column whose `state.column_types[col] == "numeric"` and whose pandas dtype is not numeric, run `state.df[col] = pd.to_numeric(state.df[col], errors="coerce")`. Only after this is `MissingValueHandler` invoked. The pass is idempotent — repeating it on an already-numeric column is a no-op. It lives in the wrapper rather than inside `MissingValueHandler.run` so the imputer's contract ("imputer sees nulls") stays clean.

**Startup (called before ADK runner):**
1. Load and validate config via `ConfigSchema`.
2. Read `config.llm.api_key`. If empty or missing, raise `RuntimeError`. The pipeline refuses to start without it.
3. Inject the key into the process environment: `os.environ[config.llm.api_key_env_var] = config.llm.api_key`. This must happen **before** any ADK / LiteLlm import or construction so the provider client picks it up. The env-var-name ergonomics heuristic (plan ambiguity #22) may rewrite the env var when the model_string prefix mismatches the configured env_var; when it fires, log one startup line to `execution_log.txt` so the override is visible.
4. Load the CSV with `pd.read_csv(data_path, keep_default_na=False, na_values=[""])` so the categorical null literals (`"None"`, `"NA"`, `"N/A"`, etc.) reach SemanticTypeInfer intact (spec §4 "Null detection in categoricals"). Only empty strings become `NaN`. Separate target column from features; validate target column exists.
5. Resolve `task`:
   - If `--task` supplied: validate it is `"classification"` or `"regression"`, raise `ValueError` otherwise.
   - If `--task` omitted: infer from target — if target is numeric **and** `nunique > task_infer_nunique_threshold` (from config) → `"regression"`, else `"classification"`.
6. Call `ray.init(num_cpus=config.pipeline.max_workers)` — once, here only.
7. Generate `run_id`.
8. Set `state.output_dir`; create directory.
9. Initialise `PipelineState` (no `model_fn` field).
10. Construct the ADK model wrapper with `max_tokens=config.llm.max_tokens` baked into every call's `GenerateContentConfig`.
11. Run **structured-output** smoke test (spec §5 "Startup smoke test"): build a one-column `SemanticTypeInferEvidence`, send it through `_make_model_call` with the same `## RESPONSE SHAPE` header SemanticTypeInfer uses, and parse the response via `validate_response` with thresholds relaxed (`min_rationale_chars=1`, `min_alternatives=0`, denylist empty). Abort startup on `failures=['parse']`; log-only on other content failures. The non-empty-text check stays as a pre-filter before parsing.
12. Construct **one** `JudgeAgent` instance (its own Agent + InMemoryRunner, isolated context). Inject it into FeatureCreator and FeatureSelector at construction time. Judge reuses the same model string, api_key, base_url, and `max_tokens` as the orchestrator agent. Construction site:
    ```python
    judge = JudgeAgent(model_string=cfg.llm.model, api_key=cfg.llm.api_key,
                       base_url=cfg.llm.base_url, max_tokens=cfg.llm.max_tokens)
    creator = FeatureCreator(judge=judge)
    selector = FeatureSelector(judge=judge)
    ```
    All other tools are constructed without a judge argument.

**Run:**
```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()
runner = Runner(agent=orchestrator_agent, session_service=session_service)
runner.run(user_message=PIPELINE_START_MESSAGE, session_id=run_id)
```

On `PostconditionError` raised by a tool: the tool function catches it, logs to `execution_log.txt`, and returns `{"status": "error", "detail": "..."}`. The ADK agent sees this and retries the tool up to `max_tool_retries` times based on its instruction prompt.

**Returns** to caller: `(output_dir, run_id)`.

---

### `pipeline/tools/adk_tools.py` — ADK Tool Wrappers

One module containing all ADK-facing tool functions. Each wraps the corresponding BaseTool. State is injected at run start via a module-level `_state` reference set by the orchestrator before `runner.run()` is called. Tools are stateless from ADK's perspective; state mutations happen inside each function.

```python
_state: PipelineState | None = None  # set by orchestrator before runner.run()

def set_pipeline_state(s: PipelineState) -> None:
    global _state
    _state = s
```

---

## Phase 4: Entry Point

### `main.py`
CLI using `argparse`.

```
python main.py run data.csv --target churn --model <model_string> [--task classification|regression]
```

- `--task` is **optional**. If omitted, the orchestrator infers it from the target column using `task_infer_nunique_threshold` from config. If supplied, it must be `classification` or `regression`.
- `--model` is required. The user supplies their own model string. No default model is set — the pipeline refuses to start without one.
- **API key**: set `llm.api_key` in `config/config.yaml`. The orchestrator copies it into `os.environ[llm.api_key_env_var]` at startup before any ADK import. The pipeline aborts if `llm.api_key` is empty or missing.

### `schema.md`
Documents all inputs (dataset path, task, target column, model string, `config.llm.api_key`), all outputs (artifact schema, report sections, log format), and PipelineState field contract.

### `README.md`
Quick-start, link to schema.md, example CLI invocation.

---

## Agent Call Summary

All model calls are made by the ADK orchestrator agent. The user supplies the model and API key; the pipeline never calls a model directly. ADK tool functions return plain dicts; the agent parses them and decides what to call next.

| ADK Tool Function | Underlying Tool | Call type | Agent input (EvidencePacket) | Agent output (Pydantic model) |
|---|---|---|---|---|
| `infer_types` | SemanticTypeInfer | one-shot + ≤1 revision | `SemanticTypeInferEvidence` | `SemanticTypeInferResponse` |
| `handle_missing` | MissingValueHandler | one-shot batched + ≤1 revision | `MissingValueEvidence` | `MissingValueResponse` |
| `handle_outliers` | OutlierHandler | one-shot batched + ≤1 revision | `OutlierEvidence` | `OutlierResponse` |
| `create_features_pre/post` | FeatureCreator | one-shot + ≤1 revision → Judge | `FeatureCreatorEvidence` | `FeatureCreatorResponse` (then Judge-ranked) |
| `scale_features` | Scaler | one-shot batched + ≤1 revision | `ScalerEvidence` | `ScalerResponse` |
| `select_features` | FeatureSelector (via Judge) | one-shot to Judge | `FeatureSelectorEvidence` | `SelectionPlanResponse` |
| `write_report` | FeatureReporter | one-shot | structured pipeline summary | Markdown text (no validation) |
| Orchestrator (ADK loop) | — | loop | tool return dict | next tool to call |

---

## Key Constraints Checklist

- [ ] No hardcoded values — all in `config.yaml`
- [ ] No hardcoded model — user supplies model string in CLI and API key in `config.yaml`
- [ ] `llm.api_key` read from config at startup and injected into `os.environ[llm.api_key_env_var]` **before any ADK import**; pipeline aborts if empty
- [ ] `llm.max_tokens` from config wired into every ADK model call's `GenerateContentConfig`
- [ ] `--task` optional: infer from target if omitted using `task_infer_nunique_threshold`; validate if supplied
- [ ] `random_state=42` on: IterativeImputer, IsolationForest, PowerTransformer, RandomForest
- [ ] KNNImputer: no seed (deterministic by algorithm)
- [ ] Ray: `ray.init(num_cpus=8)` called once in orchestrator startup — no tool calls `ray.init`
- [ ] ADK `Agent` constructed with user-supplied model string; `adk_tools.set_pipeline_state(state)` called before `runner.run()`
- [ ] ADK tool functions are plain Python — no PipelineState in ADK tool signatures; state accessed via module-level `_state`
- [ ] Target column: imputed + encoded if categorical, never scaled or selected, always last column
- [ ] `cross_categorical`: runs before encoder; any spec missing `temporal_class` rejected
- [ ] `drop_row` outlier action: drops target rows atomically with feature rows
- [ ] `id` and `text` columns: dropped before any transformation
- [ ] All model output validated before touching dataframe
- [ ] Every LLM call uses a typed `EvidencePacket` from `pipeline/evidence.py` — no ad-hoc f-string prompts over `state.profile`
- [ ] Every LLM response parsed via a Pydantic model in `pipeline/responses.py` carrying `rationale`, `evidence_cited`, `alternatives_considered`
- [ ] Content failures (lazy rationale, empty `evidence_cited`, degenerate batch) trigger exactly **one** revision retry with a delta-evidence pack; second failure → deterministic fall-through with rejected response logged to `state.warnings`
- [ ] Prompt templates describe what each strategy does mechanically; they **must not** enumerate when to use it
- [ ] Each tool with a strategy choice has a module-level `STRATEGY_DEFINITIONS: dict[str, str]` constant; prompt template injects it verbatim; no when-to-use mapping in the same file or template
- [ ] Single `JudgeAgent` instance constructed once at orchestrator startup; injected into FeatureCreator and FeatureSelector
- [ ] FeatureSelector method choice **always** goes through Judge — no standalone selector LLM call
- [ ] Coercion (float → timestamp → label) handles malformed types only at LLM-response parse; FeatureValidator coercion is strict (float → datetime only, no LabelEncode); revision is the only path for lazy content
- [ ] `_field_known` in responses.py strips `[<int>]` brackets from `evidence_cited` entries before whitelist membership; both dotted and indexed forms pass
- [ ] Orchestrator runs numeric-placeholder normalization (`pd.to_numeric(..., errors="coerce")`) on every column typed `numeric` with non-numeric dtype, before MissingValueHandler
- [ ] Startup smoke test sends one structured-output dry run through `_make_model_call` and parses via `validate_response` with relaxed thresholds; aborts on `failures=['parse']`
- [ ] Every ADK tool wrapper is idempotent: returns `{"status":"ok","detail":"already done"}` if its postcondition predicate is satisfied at entry; append-only state lists do not grow on re-entry
- [ ] No `hash()` builtin in any name that lands in `state.df` or in `feature_artifact.json`; use `hashlib.md5("|".join(sources).encode()).hexdigest()[:8]`
- [ ] No `random_state=42` literal in tool code; every sklearn seed reads `cfg.pipeline.random_state`
- [ ] `ORCHESTRATOR_INSTRUCTION` is a template formatted at init with `cfg.pipeline.max_tool_retries`; no `"3"` literal
- [ ] Profiler emits `_clusters`, `_linear_baseline_score`, `_joint_mi_pairs` into `state.profile` so Creator consumes true joint MI (not the product proxy)
- [ ] FeatureSelector cluster execution runs in parallel via Ray `run_parallel`; PCA columns are materialized into `state.df` serially after `ray.get` returns
- [ ] OutlierDecision.detector is `Literal[...] | None`; required for `flag`/`drop_row`, optional for `scale`
- [ ] `validation.lazy_min_batch_size` (default 3) controls when the joint-strategy degeneracy check fires
- [ ] `validation.raw_log_max_chars` (default 60000) controls per-attempt truncation in `raw_responses.txt`
- [ ] `run_id` generated at orchestrator init, stored in state, returned to caller

---

## Plan Ambiguities

### 1. `state.profile` shape undefined
Multiple tools read from `state.profile` but its structure is never specified.

Solution: `profile` is a dict keyed by column name. Each value is a flat dict of stats:
```python
profile[col] = {
    "dtype": str,
    "null_rate": float,
    "nunique": int,
    "top_values": list,          # top 5 values
    "mean": float | None,        # numeric only
    "std": float | None,
    "skewness": float | None,
    "kurtosis": float | None,
    "outlier_rate": float | None,
    "null_mask_corr": dict,      # {other_col: corr_with_null_mask}
    "mi_with_target": float | None,
}
# plus top-level keys:
profile["_correlation_matrix"] = pd.DataFrame   # pairwise Pearson
```

---

### 2. FeatureCreator pre/post split mechanism
The plan says "orchestrator handles this split" but never defines how creator executes in two phases.

Solution: FeatureCreator makes its one model call and caches all specs on the instance as `self._specs`. It exposes two methods: `run_pre(state)` and `run_post(state)`. `run_pre` executes only `temporal_class: pre_encoding` specs. `run_post` executes only `temporal_class: post_encoding` specs. The orchestrator calls `run_pre` before encoder and `run_post` after. The model call happens inside `run_pre` on first invocation.

---

### 3. mRMR not in packages list
scikit-learn has no mRMR. The spec §2 packages list is missing the library.

Solution: Add `mrmr-selection` to §2 packages list. Use `from mrmr import mrmr_classif, mrmr_regression` for the two task variants.

---

### 4. `state.expected_row_count` missing
Validator checks row count against post-outlier count but PipelineState has no field to store it.

Solution: Add `row_count_after_outlier: int | None = None` to PipelineState. OutlierHandler sets it to `len(state.df)` after all `drop_row` actions complete. Validator compares `len(state.df)` against this value.

---

### 5. MI ranking proxy for proposed features
Profile MI scores exist only for original columns. Proposed features (col1/col2, col1×col2) don't exist yet so can't be ranked by actual MI.

Solution: Use the mean MI of the source columns as the proxy score for each proposed feature. For a `ratio` of col1 and col2: `proxy_mi = mean(profile["col1"]["mi_with_target"], profile["col2"]["mi_with_target"])`. Sort all proposals by proxy_mi descending, take top `max_created_features`.

---

### 6. New columns have no profile stats for scaler
Scaler's model call needs skewness, kurtosis, outlier rate per column. Columns created by FeatureCreator were not in the original profile.

Solution: After `run_post` completes, compute a mini-profile for new columns only — skewness, kurtosis, mean, std, outlier rate (IQR-based, no model call) — and merge into `state.profile`. Scaler then has stats for all columns including new ones.

---

### 7. Reporter mixes responsibilities
Reporter currently writes `feature_artifact.json`, `execution_log.txt`, and `report.md`. Execution log should be incremental and artifact writing belongs to the orchestrator.

Solution: Orchestrator owns `execution_log.txt` — appends one line after each tool completes. Orchestrator writes `feature_artifact.json` after validator completes. Reporter only writes `report.md`. Reporter's precondition stays "validation complete."

---

### 8. `run()` return type vs. mutation
`BaseTool.run()` returns `PipelineState` and `__call__` does `state = self.run(state)`, but all tools mutate state in place — the two patterns conflict.

Solution: All tools mutate `state` in place. `run()` returns `None`. `__call__` calls `self.run(state)` without reassignment and returns the mutated `state`.

---

### 9. Encoder precondition incomplete
Encoder's precondition says "outlier handling complete" but cross_categorical columns from FeatureCreator's pre-encoding pass must also exist before encoder runs.

Solution: Add `pre_encoding_done: bool = False` to PipelineState. Orchestrator sets it to `True` after calling `creator.run_pre(state)`. Encoder's precondition checks both `state.column_types not None` and `state.pre_encoding_done is True`.

---

### 10. Outlier model call description is contradictory
Line says "One `model_fn` call per numeric column (batched into one prompt)" — either one call total or one per column, not both.

Solution: One batched call for all numeric columns. All column profiles are included in a single prompt. Response is a JSON array covering all columns at once.

---

### 11. `base.py` code block contradicts ambiguity #8 solution
The code block still shows `run() -> PipelineState`, but ambiguity #8 says `run()` returns `None` and mutates in place.

Solution: Update the `base.py` code block to `run(self, state: PipelineState) -> None`. `__call__` calls `self.run(state)` without reassignment, then returns the mutated `state`.

---

### 12. ~~PipelineState missing fields from ambiguity solutions~~
Resolved. Both `row_count_after_outlier: int | None = None` and `pre_encoding_done: bool = False` are now in the PipelineState definition in Phase 1.

---

### 13. FeatureCreator precondition is wrong
Creator's listed precondition is "encoding complete." But `run_pre` runs before the encoder. Its precondition should be "outlier handling complete." Only `run_post` requires "encoding complete."

Solution: Split the precondition — `run_pre` checks `state.column_types not None` (infer ran) and outlier handling done. `run_post` checks `state.pre_encoding_done is True` (encoder ran).

---

### 14. FeatureCreator doesn't fit BaseTool interface
BaseTool has one `run()` method. FeatureCreator needs `run_pre()` and `run_post()`. The standard dispatch chain doesn't apply.

Solution: FeatureCreator subclasses BaseTool but overrides `__call__` to a no-op. The orchestrator calls `run_pre` and `run_post` directly. FeatureCreator still implements `precondition` and `postcondition` scoped to the post-encoding phase only, for validator retry compatibility.

---

### 15. Datetime-decomposed columns also missing from profile
Ambiguity #6 fix adds a mini-profile after FeatureCreator. But datetime decomposition happens earlier inside SemanticTypeInfer — the year/month/day/quarter columns it produces also won't have profile stats. Scaler will hit the same gap.

Solution: Mini-profile is a shared utility function in `parallel.py`. It is called twice — once by SemanticTypeInfer after datetime decomposition, and once by FeatureCreator after `run_post`. Both merge results into `state.profile`. Scaler always has stats for every column.

---

### 16. Fitted sklearn objects are not JSON-serializable for the artifact
`state.transformers` stores fitted imputers and scalers as sklearn objects. The artifact schema expects flat JSON params (`fill_value`, `mean`, `std`). KNNImputer stores the full training matrix; IterativeImputer stores fitted estimators — neither serializes to flat JSON.

Solution: Each tool appends only replay-sufficient params, not the sklearn object. For `median`: fill value. For `knn`: strategy name only (KNN replay requires re-fitting on the stored training set). For `iterative`: strategy name + `random_state`. For scalers: strategy name + fitted params (`mean`/`std`, `center`/`scale`, `data_min`/`data_max`, `lambdas`). The artifact records what was done, not the object.

---

### 17. `output_dir` is never set
`output_dir` starts as `None` in PipelineState but the orchestrator startup sequence never sets it. Reporter and orchestrator both write files there.

Solution: Orchestrator sets `state.output_dir = Path("pipeline_output") / state.run_id` at startup step 5, immediately after `run_id` is generated, and creates the directory before the run loop starts.

---

### 18. `mrmr_k` from model response conflicts with `top_k_features` in config
Selector model response can return `{"method": "mrmr+pca", "mrmr_k": 20}` but `top_k_features: 20` is already in config.yaml. Two sources for the same value with no precedence rule.

Solution: Remove `mrmr_k` from the model response schema. Model picks method only. `top_k_features` from config always controls how many features are kept.

---

### 19. Profiler needs `state.target` explicitly but this is not stated
Profiler computes MI between features and target, but target is in `state.target` (separated at orchestrator startup), not in `state.df`. The profiler section never mentions reading `state.target`.

Solution: Profiler precondition checks both `state.df not None` and `state.target not None`. MI computation explicitly uses `state.target`.

---

### 22. API-key injection ordering and pre-existing env vars
Spec says the orchestrator reads `llm.api_key` and injects it into `os.environ[llm.api_key_env_var]` before ADK is initialised. Three questions: (a) what if the env var is already set externally? (b) how do we guarantee the injection happens before any ADK module reads the var? (c) what if the configured `api_key_env_var` does not match the provider implied by `model_string` (e.g. env_var left at default `OPENAI_API_KEY` but model targets `gemini/gemini-2.0-flash`)?

Solution: (a) The config value always wins — `os.environ[llm.api_key_env_var] = config.llm.api_key` unconditionally overwrites any pre-existing value. The single source of truth for *which key* is used is `config.yaml`. (b) Injection is the **first** step in orchestrator startup after config load, and crucially happens **before** any `from google.adk...` import inside `_run_adk_agent` and before `_make_model_call` constructs the LLM client. Place ADK imports inside method bodies, not at module top, so the env var is set before the provider client reads it. (c) An ergonomic env-var-name override is preserved: when `config.llm.api_key_env_var` is left at the default `OPENAI_API_KEY` *and* the prefix of `model_string` resolves to a different provider (`gemini` / `google` / `anthropic`), the orchestrator rewrites the env-var name to the provider-canonical one (`GOOGLE_API_KEY` / `ANTHROPIC_API_KEY`) so the right provider client picks the key up. The override fires only when the configured env_var is the default; an explicitly-set non-default env_var is honored exactly. When the override fires, the orchestrator writes one line to `execution_log.txt` (`env_var_override: configured=OPENAI_API_KEY effective=GOOGLE_API_KEY (from model prefix 'gemini')`) so the behavior is visible to the user. The single source of truth for *the key itself* is still `config.yaml`.

---

### 23. `llm.max_tokens` wiring
Spec adds `llm.max_tokens` to config but doesn't say where it is applied — ADK Agent construction, per-call config, or both.

Solution: `max_tokens` is applied **per call** via `GenerateContentConfig(max_output_tokens=config.llm.max_tokens)` inside the model-call wrapper in `orchestrator.py`. ADK's `Agent` constructor does not receive it — only the per-call `LlmRequest.config`. This keeps the wiring in one place and lets the smoke-test call use the same limit.

---

### 21. Task inference edge cases
If `--task` is omitted and the target column has exactly `task_infer_nunique_threshold` unique values (boundary), or the target is non-numeric, the inference rule is ambiguous.

Solution: Non-numeric target → always `classification` (label encoding will follow). Numeric target: strictly greater than threshold → `regression`; at-or-below → `classification`. Inferred task is logged to `execution_log.txt` at startup so callers can verify.

---

### 20. ~~Ray serialization: `model_fn` callable is not Ray-serializable~~
Resolved by ADK adoption. `model_fn` has been removed from PipelineState entirely. ADK manages all model calls on the main process through the orchestrator agent. Ray remote functions still receive only plain data (numpy arrays, dicts, lists) — that constraint holds for other reasons (zero-copy Arrow serialization), but the `model_fn` serialization problem no longer exists.

---

### 24. Where the `evidence_cited` whitelist comes from
Pydantic responses require `evidence_cited` to be a subset of the EvidencePacket field names sent. The list of "field names sent" needs a deterministic source so the validator can check membership.

Solution: the EvidencePacket serializer in `pipeline/evidence.py` returns `(prompt_text, sent_field_names: set[str])`. Tools pass `sent_field_names` directly into `validate_response(...)`. The validator membership-checks against this set. Nested dataclass fields are flattened with dotted paths (`columns.null_run_lengths`) so the model can cite them precisely.

---

### 25. Delta-evidence pack format for the revision retry
The spec mandates one retry with a delta-evidence pack contrasting the columns that received the same answer, but the contrast format is not specified.

Solution: when the lazy-batch check fires, the validator groups items by their assigned strategy. For each group with >1 member, it picks the two columns whose EvidencePacket fields differ most (max-L1 over normalised numeric fields). The delta pack is a JSON block: `{"columns": ["col_a","col_b"], "differing_fields": {"skewness": [0.1, 2.3], "outlier_rate": [0.0, 0.18]}, "your_previous_answer": "standard"}` for each contrast pair. Appended to the revision prompt under a `## REVISION` header.

---

### 26. Judge Agent invocation site for the selector
FeatureCreator already calls Judge after its own LLM call. For FeatureSelector, who calls Judge — the selector's `run()`, the orchestrator, or an ADK tool wrapper?

Solution: FeatureSelector's `run()` calls `self.judge.plan(evidence=..., cfg=...)` directly. Judge instance is injected at construction (same pattern as FeatureCreator). The selector never makes its own LLM call. If `self.judge is None`, falls through to mRMR.

---

### 27. Histogram counts in EvidencePacket are not Ray-serializable issues but verbose
10-bin and 20-bin histograms per column inflate prompt size linearly with `n_features`. For wide datasets this can blow `llm.max_tokens`.

Solution: `evidence.py` truncates verbose EvidencePackets when the rendered prompt exceeds 70% of `llm.max_tokens`. Truncation drops histograms first, then `random_samples`, then `extreme_top5/bottom5`. A trailing `## TRUNCATED` note lists which fields were removed and how many. The Pydantic validator does not require those fields to appear in `evidence_cited`.

---

### 28. "NA"-style tokens in categorical columns should not be treated as nulls
Many real datasets use `"None"`, `"NA"`, `"N/A"` as meaningful category labels (e.g. `Alley="NA"` in House Prices means "no alley access"). Pandas' default `read_csv` coerces these to `NaN`, erasing the signal and triggering spurious imputation.

Solution (spec §4 "Null detection in categoricals"):
1. Orchestrator loads the CSV with `pd.read_csv(data_path, keep_default_na=False, na_values=[""])` so only empty strings become `NaN`. The tokens reach SemanticTypeInfer as plain strings.
2. The token list lives in `config.yaml/imputation.categorical_null_literals` (default: `["None","NA","N/A","na","n/a","none","NaN"]`). Callers can extend or shrink it without code changes.
3. `MissingValueHandler` excludes categorical and binary columns from `cols_with_nulls` entirely. For numeric/datetime/target columns the imputer behaves as before — true `NaN`s in those columns still trigger imputation.
4. `LabelEncoder` in `Encoder` sees these tokens as ordinary string categories and assigns them an integer code alongside the other labels.

---

### 29. Numeric columns arriving as object dtype because of `"NA"` placeholders
Spec §4 "Null detection in categoricals" disables pandas' `NaN` coercion to preserve `"NA"` as a category label. The same disable-flag traps numeric columns that use `"NA"` for missing measurements: `LotFrontage`, `MasVnrArea`, `GarageYrBlt` in House Prices all arrive as object dtype mixing numeric strings and `"NA"`. SemanticTypeInfer correctly types them `numeric`, but `df[col].isna()` is False everywhere, so MissingValueHandler skips them and downstream tools see object-typed data they cannot scale.

Solution (spec §4 "Numeric placeholder normalization", spec §7-W): orchestrator runs a normalization pass after SemanticTypeInfer. The implementation site is `adk_tools.handle_missing` — it runs `pd.to_numeric(state.df[col], errors="coerce")` on every column with `state.column_types[col] == "numeric"` and non-numeric pandas dtype, *before* invoking `MissingValueHandler`. The pass is idempotent and lives in the wrapper, not inside MissingValueHandler.run, so the imputer's contract ("imputer sees nulls") stays clean. Categorical columns are untouched because the pass is gated on the assigned semantic type.

---

### 30. `evidence_cited` whitelist rejects every indexed-form citation
The renderer in `evidence.py::_collect_field_names` emits dotted paths only (`columns.dtype`), but models naturally cite array entries by index (`columns[3].dtype`) because the serialized JSON they see is `{"columns": [...]}`. `_field_known` accepts only exact match or `s.endswith("." + cited)` — neither admits the indexed form. The first attempt fails the evidence check, the revision prompt does not advertise either form, the second attempt fails the same way, and every LLM-driven decision falls through to its deterministic default.

Solution (spec §7-V): `_field_known` normalizes the cited string by stripping `[<int>]` brackets before whitelist membership (`re.sub(r"\[\d+\]", "", cited)`). Both dotted and indexed forms pass. The whitelist itself stays dotted; the equivalence lives in the matcher. Prompts do not advertise either form.

---

### 31. Startup smoke test only checks for non-empty text
A model that returns reasoning-channel content stripped to None by `_strip_harmony`, or that ignores the response-shape header entirely, still passes the existing smoke test. The cascading failures only surface after an hour-long pipeline run.

Solution (spec §5 "Startup smoke test", spec §7-Y): the orchestrator startup runs one structured-output dry run through `_make_model_call` — a minimal `SemanticTypeInferEvidence` (one column) plus the SemanticTypeInfer prompt — and parses the response via `validate_response` with thresholds relaxed (`min_rationale_chars=1`, `min_alternatives=0`, denylist empty). `failures=['parse']` aborts startup with the raw response attached to the error. Other content failures are logged but do not abort. The legacy non-empty-text check stays as a pre-filter.

---

### 32. Tool idempotency under ADK re-dispatch
The orchestrator's ADK agent can call any tool multiple times. Tool side effects (`state.transformers.append(...)`, `state.dropped_columns.extend(...)`, `state.created_columns.append(...)`) would duplicate and bloat the artifact.

Solution (spec §5 "Tool idempotency", spec §7-Z): each tool wrapper in `adk_tools.py` checks a postcondition predicate at entry. If the predicate is satisfied, the wrapper returns `{"status": "ok", "detail": "already done"}` and does not invoke the underlying tool. The per-wrapper predicates are listed in the Phase 3 "Idempotency wrapper" paragraph; they are kept as module-level functions next to each wrapper so a single grep reveals the full contract.

---

### 33. Deterministic component naming
FeatureSelector's `_pca` currently names new components via `abs(hash(tuple(X.columns))) % 10_000`. Python's `hash` is salted per process; running the pipeline twice on the same data produces different column names, which breaks `feature_artifact.json` replay.

Solution (spec §5 "Deterministic naming", spec §7-AA): use `hashlib.md5("|".join(sources).encode()).hexdigest()[:8]`. PCA component names take the form `pca_<md5_8>_<i>`. The Python built-in `hash` is forbidden anywhere a name lands in `state.df` or in the artifact. Any future tool that synthesizes column names uses the same construction.

---

### 34. Outlier `detector` field when action is `scale`
The Pydantic `OutlierDecision` schema requires a detector for every decision, but when the action is `scale` the detector mask is computed and immediately discarded — the whole column is RobustScaled. The model is being asked to pick a detector with no consequence, and the lazy-batch joint-strategy check fires falsely on legitimate batches where many columns share `(<any_detector>, scale)`.

Solution (spec §4 "Outlier decisions", spec §7-CC): `OutlierDecision.detector: Literal["iqr","zscore","isolation_forest"] | None`. The prompt instructs the model to omit `detector` when picking `scale`. `flag` and `drop_row` still require a detector. The degeneracy tuple uses `(detector or "n/a", action)` so an all-`(None, scale)` batch is still caught by the existing check.

---

### 35. `raw_responses.txt` truncation cap is hardcoded
The raw response logger caps each entry at 15,000 chars (`responses.py::_RAW_SNIPPET_CHARS`). Long batched responses (SemanticTypeInfer on a wide dataset can run past 20kB) get clipped mid-JSON, making post-hoc analysis impossible.

Solution (spec §6 "Observability detail", config `validation.raw_log_max_chars`): the cap moves to config with a default of 60000. The logger reads `cfg.validation.raw_log_max_chars`. Truncation marker stays `... [truncated, total N chars]`. Callers running wide-dataset pipelines bump the cap; tight-budget callers shrink it.
