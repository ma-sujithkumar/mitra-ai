# Plan: Feature Engineering Pipeline Visualizations
**Date:** 2026-06-21  
**Scope:** `backend/agents/feature_engineering/visuals/`  
**Goal:** Generate visual summaries of every pipeline step's results AND the LLM reasoning behind each decision, reading from post-run JSON artifacts (no re-running the pipeline).

---

## 1. Input Artifacts (what already exists after a pipeline run)

| Artifact | Location | Used For |
|---|---|---|
| `feature_artifact.json` | `pipeline_output/<run_id>/` | dropped cols, created cols, transformers (imputation/encoding/scaling decisions), selected cols, selection_method, warnings |
| `mutual_info.json` | `.mitra/<run_id>/stats/` | MI score per feature |
| `rf_importance.json` | `.mitra/<run_id>/stats/` | RF importance per feature |
| `mrmr_ranking.json` | `.mitra/<run_id>/stats/` | mRMR feature ranking order |
| `variance.json` | `.mitra/<run_id>/stats/` | variance per feature + low-variance list |
| `correlation_pearson.json` | `.mitra/<run_id>/stats/` | pairwise Pearson correlation triplets (a, b, r) |
| `correlation_spearman.json` | `.mitra/<run_id>/stats/` | pairwise Spearman correlation triplets |
| `clusters.json` | `.mitra/<run_id>/stats/` | cluster_id -> [member cols] |
| `linear_baseline.json` | `.mitra/<run_id>/stats/` | baseline model score + task |
| `pca.json` | `.mitra/<run_id>/stats/` | explained_variance_ratio, n_components_for_threshold |
| `raw_responses.txt` | `pipeline_output/<run_id>/` | Full LLM responses -- parse rationale per step per column |
| `execution_log.txt` | `pipeline_output/<run_id>/` | Per-step duration and status |

> NOTE: `raw_responses.txt` contains full LLM JSON responses. We parse these blocks to extract
> `rationale`, `evidence_cited`, and `alternatives_considered` per decision item, matching
> them to feature names from `feature_artifact.json`.

---

## 2. Output Structure

All visuals saved to `pipeline_output/<run_id>/plots/`:

```
pipeline_output/<run_id>/
  plots/
    01_feature_importance.html      -- Interactive: MI + RF + mRMR combo chart
    02_correlation_clusters.html    -- Interactive: heatmap + cluster coloring
    03_imputation_decisions.html    -- Table: column, strategy, rationale, evidence
    04_outlier_decisions.html       -- Table: column, detector, action, rationale
    05_scaling_decisions.html       -- Table: column, scaler, rationale, evidence
    06_created_features.html        -- Chart + table: operation, sources, rationale
    07_selection_rationale.html     -- Bar chart: keep/drop with rationale tooltip
    08_pca_variance.html            -- Scree plot: cumulative explained variance
    09_pipeline_timeline.html       -- Timeline: step, duration, llm status
    dashboard.html                  -- Master page linking/embedding all the above
```

All charts use **Plotly** (interactive HTML, no server required).

---

## 3. Modules to Create

### `visuals/artifact_reader.py`
**Class: `ArtifactReader`**

- Constructor takes `run_output_dir: Path`
- Loads all JSON artifacts from `pipeline_output/<run_id>/` and `.mitra/<run_id>/stats/`
- Parses `raw_responses.txt` to extract per-step LLM rationale blocks (keyed by caller tag)
- Exposes typed properties:
  - `mi_scores` -> `dict[str, float]`
  - `rf_importance` -> `dict[str, float]`
  - `mrmr_ranking` -> `list[str]`
  - `clusters` -> `dict[str, list[str]]`
  - `imputation_decisions` -> `list[dict]` (column, strategy, rationale, evidence, status)
  - `outlier_decisions` -> `list[dict]` (column, detector, action, rationale, status)
  - `scaling_decisions` -> `list[dict]` (column, scaler, rationale, status)
  - `creation_specs` -> `list[dict]` (name, operation, sources, rationale, phase)
  - `selection_decision` -> `dict` (keep, drop, use_pca, rationale, method)
  - `pca_data` -> `dict` (explained_variance_ratio, n_components)
  - `timeline_events` -> `list[dict]` (step, timestamp, duration_s, status, llm_source)

---

### `visuals/base.py`
**Abstract Class: `BaseVisualizer`**

```python
class BaseVisualizer(ABC):
    def __init__(self, reader: ArtifactReader, output_dir: Path): ...
    @abstractmethod
    def build(self) -> go.Figure: ...
    def save(self, filename: str) -> Path: ...  # saves self-contained HTML
```

---

### `visuals/importance.py`
**Class: `FeatureImportanceVisualizer(BaseVisualizer)`**

- Reads: `mi_scores`, `rf_importance`, `mrmr_ranking`, `selected_columns`, `linear_baseline`
- Builds a **grouped horizontal bar chart** -- 3 bars per feature: MI, RF importance, mRMR rank (inverted so higher = more important)
- Green bars = selected by pipeline, Red bars = dropped
- Vertical annotation line at linear baseline score
- Hover: feature name, all 3 scores, selection status
- Output: `01_feature_importance.html`

---

### `visuals/correlation.py`
**Class: `CorrelationClusterVisualizer(BaseVisualizer)`**

- Reads: `correlation_pearson.json`, `clusters.json`, `selected_columns`
- Builds a **square heatmap** of pairwise Pearson correlations:
  - Rows/cols reordered by cluster membership
  - Cluster boundary rectangle overlays with colored borders
  - Color scale: RdBu diverging (-1 to +1)
- Hover: (col_a, col_b, pearson_r, cluster_id, both selected/dropped status)
- Output: `02_correlation_clusters.html`

---

### `visuals/decisions.py`
**Class: `DecisionTableVisualizer(BaseVisualizer)`**

Generic Plotly table for per-column LLM decisions. Used for 3 pipeline steps.

Columns rendered: `Column | Decision | Rationale | Evidence Cited | Alternatives | Status`

Row coloring convention:
- White  = LLM decision accepted on first attempt (ok)
- Amber  = Rule-based fallback used (LLM failed validation)
- Blue   = LLM decision accepted after revision (ok:revised)

Instantiated for:
- `03_imputation_decisions.html` -- strategy per column from MissingValueHandler
- `04_outlier_decisions.html` -- (detector, action) per column from OutlierHandler
- `05_scaling_decisions.html` -- scaler per column from Scaler

---

### `visuals/creation.py`
**Class: `FeatureCreationVisualizer(BaseVisualizer)`**

- Reads: `creation_specs` from ArtifactReader
- Two-panel layout:
  - Left: horizontal bar chart of operation-type counts (ratio, product, log1p, etc.)
  - Right: table of all created features (name, operation, sources, rationale, phase)
- Hover on bars: list of feature names for that operation
- Output: `06_created_features.html`

---

### `visuals/selection.py`
**Class: `SelectionRationaleVisualizer(BaseVisualizer)`**

- Reads: `mi_scores`, `selected_columns`, `dropped_columns`, `selection_decision`, `pca_data`
- Horizontal sorted bar chart of MI scores:
  - Green = selected, Red = dropped
  - If PCA used: second panel showing scree plot + "N components chosen"
- Hover: feature name, MI score, keep/drop, LLM rationale text
- Title includes `selection_method` tag
- Output: `07_selection_rationale.html`

---

### `visuals/pca.py`
**Class: `PCAVarianceVisualizer(BaseVisualizer)`**

- Reads: `pca_data` (explained_variance_ratio list, n_components_for_threshold)
- Scree plot: bar chart of per-component variance + cumulative variance line
- Dashed vertical line at n_components chosen
- Annotation: "Threshold: X% variance retained with N components"
- Output: `08_pca_variance.html` (skipped if PCA was not used)

---

### `visuals/timeline.py`
**Class: `PipelineTimelineVisualizer(BaseVisualizer)`**

- Reads: `timeline_events` from ArtifactReader (parsed from `execution_log.txt`)
- Horizontal Gantt/bar chart -- each step is a row, bar length = duration in seconds
- Color: ok=green, fallback=amber, error=red
- Hover: step name, duration (s), llm_source tag, status detail
- Output: `09_pipeline_timeline.html`

---

### `visuals/dashboard.py`
**Class: `VisualDashboard`**

- Constructor: `__init__(self, run_output_dir: Path)`
- Creates `ArtifactReader` + all 8 visualizers
- `build_all(self) -> list[Path]` -- calls `.build()` + `.save()` on each
- `build_dashboard_html(self, plot_paths: list[Path]) -> Path`
  - Generates `dashboard.html`:
    - Nav sidebar linking to each visual
    - Embeds each in an `<iframe>`
    - Run metadata at top: run_id, task, target_column, n_features_in -> n_features_out, selection_method, warnings
- `run(self)` -- orchestrates build_all + build_dashboard_html, prints final path

---

### `visuals/cli.py`
**CLI entry point**

```
python -m backend.agents.feature_engineering.visuals.cli \
    --run-dir pipeline_output/<run_id>  \
    [--out-dir pipeline_output/<run_id>/plots] \
    [-v]
```

- Uses `argparse`, all args required (no hardcoded defaults per CLAUDE.md rule 27)
- Calls `VisualDashboard(run_output_dir).run()`
- Prints path to `dashboard.html` on completion

---

## 4. Dependencies Required

| Package | Purpose |
|---|---|
| `plotly >= 5.x` | All interactive charts (verify if already in requirements) |
| `pandas` | Already present -- matrix manipulation for heatmap |
| `numpy` | Already present -- correlation matrix reordering |

No new heavyweight dependencies. Only Plotly needs to be confirmed/added.

---

## 5. Key Design Decisions

1. **Read-only from artifacts** -- visualizers never re-run the pipeline.
2. **LLM rationale extraction** -- parse `raw_responses.txt` blocks (keyed by caller tag like `MissingValueHandler`) to get rationale/evidence per column decision.
3. **Plotly HTML only** -- self-contained, no server needed, hover shows rationale text.
4. **Color convention across all charts**:
   - Green  = selected / ok / used
   - Red    = dropped / removed
   - Amber  = fallback (rule-based, LLM failed validation)
   - Blue   = revised (LLM second attempt accepted)
5. **Output to `plots/` under run_id** -- follows CLAUDE.md rule 24.
6. **OOP throughout** -- every visualizer is a class (CLAUDE.md rule 9).
7. **No hardcoded paths** -- everything resolved from `run_output_dir` argument (CLAUDE.md rule 7).
8. **Typing on all methods** (CLAUDE.md rule 29).

---

## 6. File Creation Order (when executing)

1. `visuals/artifact_reader.py` -- foundation, everything else depends on it
2. `visuals/base.py` -- abstract base
3. `visuals/importance.py`
4. `visuals/correlation.py`
5. `visuals/decisions.py` -- covers imputation + outlier + scaler in one file
6. `visuals/creation.py`
7. `visuals/selection.py`
8. `visuals/pca.py`
9. `visuals/timeline.py`
10. `visuals/dashboard.py`
11. `visuals/cli.py`
12. `visuals/__init__.py` -- expose `VisualDashboard`

---

## 7. Scope Boundaries (what this does NOT do)

- Does NOT re-run any pipeline step
- Does NOT call any LLM
- Does NOT modify `feature_artifact.json` or any existing artifact
- Does NOT touch any existing pipeline code (orchestrator, tools, state, etc.)
