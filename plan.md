# Implementation Plan: Feature Engineering Agent

---

## Build Order

Four phases. Each phase only starts when the previous is complete. Later phases depend on earlier ones structurally.

```
Phase 1 — Foundation       config.yaml, config.py, state.py, base.py, parallel.py
Phase 2 — Tools            all 10 tools in pipeline/tools/
Phase 3 — Orchestrator     orchestrator.py
Phase 4 — Entry point      main.py, schema.md, README.md
```

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
    executor: ThreadPoolExecutor
    model_fn: Callable[[str], str]

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
- Module-level `EXECUTOR = ThreadPoolExecutor(max_workers=8)`.
- Helper `run_parallel(fn, items, executor) -> list` — submits all items, collects results.
- This is the only executor in the entire pipeline.

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

Runs univariate and multivariate analysis in parallel via `state.executor`.

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
- Runs across 4 column batches in parallel via `state.executor`.
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

Selection methods run in parallel where independent (e.g., multiple scoring methods before final pick). Uses `state.executor`.

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

Entry point for callers. Owns the run loop.

**Startup:**
1. Validate `task` — raise `ValueError` if not `classification` or `regression`.
2. Load and validate config via `ConfigSchema`.
3. Run smoke test on `model_fn` (single test prompt, check response is non-empty string).
4. Generate `run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]`.
5. Separate target column from features, validate target column exists.
6. Initialise `PipelineState`.

**Run loop:**
- Default tool sequence: profiler → infer → imputer → outlier → [creator pre-encoding] → encoder → [creator post-encoding] → scaler → selector → validator → reporter.
- Orchestrator checks input satisfaction before dispatching each tool.
- On `PostconditionError`: retry tool up to `max_tool_retries` (3) times. On 3rd failure, skip tool and log warning.
- Tools whose inputs are all satisfied simultaneously are dispatched in parallel via `state.executor`.

**Returns** to caller: `(output_dir, run_id)`.

---

## Phase 4: Entry Point

### `main.py`
CLI using `argparse`.

```
python main.py run data.csv --task classification --target churn [--model openai]
```

Instantiates orchestrator, wires up `model_fn` from supplied provider, calls `run()`, prints output path.

### `schema.md`
Documents all inputs (dataset path, task, target column, model_fn), all outputs (artifact schema, report sections, log format), and PipelineState field contract.

### `README.md`
Quick-start, link to schema.md, example CLI invocation.

---

## Agent Call Summary

| Tool | Call type | Input | Output |
|---|---|---|---|
| SemanticTypeInfer | one-shot | column names + stats | JSON array of type assignments |
| MissingValueHandler | one-shot (batched) | null profiles per column | JSON array of strategies |
| OutlierHandler | one-shot (batched) | outlier profiles per column | JSON array of detector + action |
| FeatureCreator | one-shot | column stats | JSON list of operation specs |
| Scaler | one-shot (batched) | distribution profiles | JSON array of scaler assignments |
| FeatureSelector | one-shot | profile summary | JSON method selection |
| FeatureReporter | one-shot | structured pipeline summary | Markdown text |
| Orchestrator loop | loop | tool result | next tool to run |

---

## Key Constraints Checklist

- [ ] No hardcoded values — all in `config.yaml`
- [ ] `random_state=42` on: IterativeImputer, IsolationForest, PowerTransformer, RandomForest
- [ ] KNNImputer: no seed (deterministic by algorithm)
- [ ] Single `EXECUTOR` in `parallel.py` — no tool creates its own
- [ ] Target column: imputed + encoded if categorical, never scaled or selected, always last column
- [ ] `cross_categorical`: runs before encoder; any spec missing `temporal_class` rejected
- [ ] `drop_row` outlier action: drops target rows atomically with feature rows
- [ ] `id` and `text` columns: dropped before any transformation
- [ ] All model output validated before touching dataframe
- [ ] `run_id` generated at orchestrator init, stored in state, returned to caller
