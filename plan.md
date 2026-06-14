# Implementation Plan: Feature Engineering Agent

---

## Build Order

Four phases. Each phase only starts when the previous is complete. Later phases depend on earlier ones structurally.

```
Phase 1 — Foundation       config.yaml, config.py, state.py, base.py, parallel.py
Phase 2 — Tools            all 10 tools in pipeline/tools/ (each also exposed as an ADK tool function)
Phase 3 — Orchestrator     orchestrator.py — ADK Agent + ADK tool registration
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

scaling:
  power_transformer_method: yeo-johnson

pipeline:
  max_tool_retries: 3
  random_state: 42
  max_workers: 8
```

---

### `pipeline/config.py`
- Pydantic model `ConfigSchema` mirroring every key above with types and defaults.
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

---

## Phase 2: Tools

Build in this order — each tool depends on the state fields populated by tools before it.

```
profiler → infer → imputer → outlier → encoder → creator → scaler → selector → validator → reporter
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
- pairwise Pearson correlation matrix
- mutual information between each feature and target

Output: populates `state.profile`.

---

### `pipeline/tools/infer.py` — `SemanticTypeInfer`

**Precondition:** `state.profile` not None.
**Postcondition:** `state.column_types` not None.

One `model_fn` call. Prompt includes column names, dtypes, null rates, sample values from profile. Response is a JSON array `[{"column": "...", "type": "..."}]`.

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

One `model_fn` call. Prompt includes null rate, MCAR/MAR signal (correlation of null mask with other columns), column type. Response is JSON `[{"column": "...", "strategy": "..."}]`.

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

One `model_fn` call per numeric column (batched into one prompt). Response is JSON `[{"column": "...", "detector": "...", "action": "..."}]`.

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

**Precondition:** encoding complete.
**Postcondition:** new feature columns added to `state.df`.

One `model_fn` call. Prompt includes column stats from profile. Response is JSON list of operation specs:

```json
[{"operation": "ratio", "sources": ["col1", "col2"], "name": "col1_div_col2", "temporal_class": "post_encoding"}]
```

Each spec must have `temporal_class`: `pre_encoding` or `post_encoding`. Specs missing `temporal_class` are rejected at validation.

`cross_categorical` specs → `temporal_class: pre_encoding` (already executed before encoder ran).
All other specs → `temporal_class: post_encoding`, executed now.

Cap: top `max_created_features` proposals kept, ranked by expected MI with target using profile data. Remainder discarded before any column is materialized.

Each created column appended to `state.created_columns`.

---

### `pipeline/tools/scaler.py` — `Scaler`

**Precondition:** feature creation complete.
**Postcondition:** all numeric feature columns are scaled floats.

One `model_fn` call. Prompt includes skewness, kurtosis, outlier rate per column from profile. Response is JSON `[{"column": "...", "scaler": "..."}]`.

Scalers: `standard`, `robust`, `minmax`, `power` (Yeo-Johnson, `random_state=42`).

- Target column: never scaled.
- Columns are batched and scaled in parallel via Ray (`run_parallel` from `parallel.py`).
- Each fitted scaler appended to `state.transformers`.

---

### `pipeline/tools/selector.py` — `FeatureSelector`

**Precondition:** scaling complete.
**Postcondition:** `state.selected_columns` not None.

One `model_fn` call. Prompt includes task type, feature count, dataset size, correlation matrix summary, MI scores from profile. Response is JSON `{"method": "..."}` or `{"method": "mrmr+pca", "mrmr_k": 20}`.

Decision logic (from spec):
- High inter-correlation + low MI → PCA
- Low inter-correlation + high MI → mRMR
- Mixed → mRMR on significant features, PCA on residual block
- Default fallback → mRMR

Selection methods run in parallel where independent (e.g., multiple scoring methods before final pick). Uses Ray (`run_parallel` from `parallel.py`).

Target column excluded from selection entirely. Populates `state.selected_columns` and `state.selection_method`.

---

### `pipeline/tools/validator.py` — `FeatureValidator`

**Precondition:** selection complete.
**Postcondition:** `state.df` contains only `selected_columns` + target, all float64.

No model call. Code only. Checks:
- All selected columns present in df.
- No NaNs in selected columns.
- All selected columns are float64 (attempt coercion: float → timestamp → label encode; if all fail, skip column + log warning).
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

## Phase 3: Orchestrator

### `pipeline/orchestrator.py` — `FeatureEngineerOrchestrator`

The orchestrator is a **Google ADK `Agent`**. It owns ADK agent construction, startup checks, and run invocation. It does not implement a manual loop — ADK's runner handles tool dispatch.

**ADK Agent definition:**
```python
from google.adk.agents import Agent

orchestrator_agent = Agent(
    name="feature_engineer_orchestrator",
    model="gemini-2.0-flash-exp",        # or caller-supplied model string
    description="Orchestrates the feature engineering pipeline.",
    instruction=ORCHESTRATOR_SYSTEM_PROMPT,  # pipeline sequencing rules in plain text
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

**Startup (called before ADK runner):**
1. Validate `task` — raise `ValueError` if not `classification` or `regression`.
2. Load and validate config via `ConfigSchema`.
3. Call `ray.init(num_cpus=config.pipeline.max_workers)` — once, here only.
4. Generate `run_id`.
5. Separate target column from features; validate target column exists.
6. Set `state.output_dir`; create directory.
7. Initialise `PipelineState` (no `model_fn` field).
8. Run ADK connectivity smoke test — send a minimal test message to the agent, verify a non-empty response.

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
python main.py run data.csv --task classification --target churn --model <model_string>
```

`--model` is required. The user supplies their own model string (e.g., `gemini/gemini-2.0-flash`, `openai/gpt-4o`, or any ADK-compatible identifier) and sets the corresponding API key as an environment variable before running. The model string is passed directly to the ADK `Agent` constructor. No default model is set — the pipeline refuses to start without one.

### `schema.md`
Documents all inputs (dataset path, task, target column, model_fn), all outputs (artifact schema, report sections, log format), and PipelineState field contract.

### `README.md`
Quick-start, link to schema.md, example CLI invocation.

---

## Agent Call Summary

All model calls are made by the ADK orchestrator agent. The user supplies the model and API key; the pipeline never calls a model directly. ADK tool functions return plain dicts; the agent parses them and decides what to call next.

| ADK Tool Function | Underlying Tool | Call type | Agent input | Agent output |
|---|---|---|---|---|
| `infer_types` | SemanticTypeInfer | one-shot | column names + stats | JSON array of type assignments |
| `handle_missing` | MissingValueHandler | one-shot (batched) | null profiles per column | JSON array of strategies |
| `handle_outliers` | OutlierHandler | one-shot (batched) | outlier profiles per column | JSON array of detector + action |
| `create_features_pre/post` | FeatureCreator | one-shot | column stats | JSON list of operation specs |
| `scale_features` | Scaler | one-shot (batched) | distribution profiles | JSON array of scaler assignments |
| `select_features` | FeatureSelector | one-shot | profile summary | JSON method selection |
| `write_report` | FeatureReporter | one-shot | structured pipeline summary | Markdown text |
| Orchestrator (ADK loop) | — | loop | tool return dict | next tool to call |

---

## Key Constraints Checklist

- [ ] No hardcoded values — all in `config.yaml`
- [ ] No hardcoded model — user supplies model string and API key via env var
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

### 20. ~~Ray serialization: `model_fn` callable is not Ray-serializable~~
Resolved by ADK adoption. `model_fn` has been removed from PipelineState entirely. ADK manages all model calls on the main process through the orchestrator agent. Ray remote functions still receive only plain data (numpy arrays, dicts, lists) — that constraint holds for other reasons (zero-copy Arrow serialization), but the `model_fn` serialization problem no longer exists.
