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
| `raw_responses.txt` | `pipeline_output/<run_id>/` | Parse LLM rationale per step per column |
| `execution_log.txt` | `pipeline_output/<run_id>/` | Per-step duration and status |

> NOTE: `raw_responses.txt` contains full LLM JSON responses. We will parse these to extract `rationale` and `evidence_cited` fields per decision item, matching them to the feature names from `feature_artifact.json`.

---

## 2. Output Structure

All visuals are saved to `pipeline_output/<run_id>/plots/`:

```
pipeline_output/<run_id>/
└── plots/
    ├── 01_feature_importance.html        # Interactive: MI + RF + mRMR combo chart
    ├── 02_correlation_clusters.html      # Interactive: heatmap + cluster coloring
    ├── 03_imputation_decisions.html      # Table: column, strategy, rationale, evidence
    ├── 04_outlier_decisions.html         # Table: column, detector, action, rationale
    ├── 05_scaling_decisions.html         # Table: column, scaler, rationale, evidence
    ├── 06_created_features.html          # Chart + table: operation, sources, rationale
    ├── 07_selection_rationale.html       # Bar chart: keep/drop with rationale tooltip
    ├── 08_pca_variance.html              # Scree plot: cumulative explained variance
    ├── 09_pipeline_timeline.html         # Timeline: step name, duration, llm status
    └── dashboard.html                    # Master page: embeds/links all the above
```

All charts use **Plotly** (interactive HTML, no server required) so they work standalone.

---

## 3. Modules to Create

### `visuals/artifact_reader.py`
**Class: `ArtifactReader`**  
- Constructor: `__init__(self, run_output_dir: Path)`
- Loads all JSON artifacts from `pipeline_output/<run_id>/` and `.mitra/<run_id>/stats/`
- Parses `raw_responses.txt` to extract per-step LLM rationale blocks
- Exposes typed properties: `mi_scores`, `rf_importance`, `mrmr_ranking`, `clusters`, `transformers`, `selected_columns`, `created_columns`, `dropped_columns`, `pca_data`, `imputation_decisions`, `outlier_decisions`, `scaling_decisions`, `feature_creator_decisions`, `selection_decision`
- Method: `parse_raw_responses(txt: str) -> dict[str, list[dict]]` — keyed by caller tag (MissingValueHandler, OutlierHandler, etc.)

---

### `visuals/base.py`
**Abstract Class: `BaseVisualizer`**  
```python
class BaseVisualizer(ABC):
    def __init__(self, reader: ArtifactReader, output_dir: Path): ...
    @abstractmethod
    def build(self) -> go.Figure: ...
    def save(self, filename: str) -> Path: ...  # saves HTML
```

---

### `visuals/importance.py`
**Class: `FeatureImportanceVisualizer(BaseVisualizer)`**  
- Reads: `mi_scores`, `rf_importance`, `mrmr_ranking`, `selected_columns`, `linear_baseline`
- Builds a **grouped horizontal bar chart** (3 bars per feature: MI, RF importance, mRMR rank inverted)
- Color-codes bars: green = selected by pipeline, red = dropped
- Adds vertical annotation line at baseline score
- Title: "Feature Importance (MI / RF / mRMR) — Selected vs Dropped"
- Hover: feature name, all three scores, selection status
- Output: `01_feature_importance.html`

---

### `visuals/correlation.py`
**Class: `CorrelationClusterVisualizer(BaseVisualizer)`**  
- Reads: `correlation_pearson.json`, `clusters.json`, `selected_columns`
- Builds a **square heatmap** of pairwise Pearson correlations
  - Rows and columns reordered by cluster membership
  - Rectangle overlays per cluster (colored borders)
  - Color scale: RdBu diverging (-1 to +1)
- Hover: (col_a, col_b, pearson_r, cluster_id, both selected/dropped status)
- Annotation: cluster labels on diagonal
- Output: `02_correlation_clusters.html`

---

### `visuals/decisions.py`
**Class: `DecisionTableVisualizer(BaseVisualizer)`**  
Generic table visualizer for any per-column LLM decision.  
Instantiated three times (imputation, outlier, scaler).

For each decision type it builds a **Plotly table** with columns:
- Column Name | Decision | Rationale | Evidence Cited | Alternatives Considered | Status (ok / fallback / revised)

**Imputation Table:**
- Decision = imputation strategy (median / knn / iterative / mode / drop)
- Source = `transformers` filtered by `step == "imputation"` + rationale from `raw_responses.txt` block `MissingValueHandler`
- Color rows: fallback decisions in amber, ok in white, revised in light blue
- Output: `03_imputation_decisions.html`

**Outlier Table:**
- Decision = `(detector, action)` pair (e.g., iqr / flag)
- Source = `raw_responses.txt` block `OutlierHandler`
- Color: fallback in amber
- Output: `04_outlier_decisions.html`

**Scaling Table:**
- Decision = scaler name (standard / robust / minmax / power)
- Source = `transformers` filtered by `step == "scaling"` + rationale from `raw_responses.txt` block `Scaler`
- Output: `05_scaling_decisions.html`

---

### `visuals/creation.py`
**Class: `FeatureCreationVisualizer(BaseVisualizer)`**  
- Reads: `created_columns` from `feature_artifact.json` + rationale from `raw_responses.txt` blocks `FeatureCreator`
- Builds a **two-panel layout**:
  - Left: horizontal bar chart of operation type counts (ratio, product, log1p, etc.)
  - Right: table of all created features (name, operation, sources, rationale, phase: pre/post encoding)
- Hover on bars: list of feature names for that operation type
- Output: `06_created_features.html`

---

### `visuals/selection.py`
**Class: `SelectionRationaleVisualizer(BaseVisualizer)`**  
- Reads: `mi_scores`, `selected_columns`, `dropped_columns`, selection rationale from `raw_responses.txt` block `FeatureSelector`, `selection_method`, `pca_data`
- Builds a **horizontal sorted bar chart** of MI scores:
  - Green bars = selected features
  - Red bars = dropped features
  - If PCA was used, show a separate panel: scree plot of explained variance + "n components chosen = X"
- Hover: feature name, MI score, keep/drop decision, rationale (from LLM `FeatureSelectionResponse.rationale`)
- Title: "Feature Selection — Method: {selection_method}"
- Output: `07_selection_rationale.html`

---

### `visuals/pca.py`
**Class: `PCAVarianceVisualizer(BaseVisualizer)`**  
- Reads: `pca.json` (explained_variance_ratio list, n_components_for_threshold)
- Builds a **scree plot**: bar chart of per-component variance + line of cumulative variance
- Vertical dashed line at n_components chosen to meet `pca_variance_retained` threshold
- Annotation: "Threshold: X% variance retained with N components"
- Output: `08_pca_variance.html` (skipped/empty if PCA was not used)

---

### `visuals/timeline.py`
**Class: `PipelineTimelineVisualizer(BaseVisualizer)`**  
- Reads: `execution_log.txt` lines
- Parses: timestamp, step_name, status (ok / fallback / error), duration (seconds), llm_source tag
- Builds a **horizontal Gantt/bar chart**: each step is a row, bar length = duration
- Color-codes: ok=green, fallback=amber, error=red
- Hover: step name, duration, llm_source, status detail
- Output: `09_pipeline_timeline.html`

---

### `visuals/dashboard.py`
**Class: `VisualDashboard`**  
- Constructor: `__init__(self, run_output_dir: Path)`
- Creates `ArtifactReader` + all 8 visualizers
- Method: `build_all(self) -> list[Path]` — calls `.build()` + `.save()` on each
- Method: `build_dashboard_html(self, plot_paths: list[Path]) -> Path`
  - Generates a single `dashboard.html` that:
    - Has a nav sidebar with links to each visual
    - Embeds each plot in an `<iframe>` (or inline if small)
    - Shows run metadata at top: run_id, task, target_column, n_features_in, n_features_out, selection_method, warnings
- Method: `run(self)` — orchestrates build_all + build_dashboard_html, prints paths

---

### `visuals/cli.py`
**Entry point: `visualize` CLI**  
```
python -m backend.agents.feature_engineering.visuals.cli \
    --run-dir pipeline_output/<run_id>  \
    [--out-dir pipeline_output/<run_id>/plots] \
    [-v]
```
- Uses `argparse`
- Calls `VisualDashboard(run_output_dir).run()`
- Prints path to `dashboard.html` on completion

---

## 4. Dependencies Required

| Package | Version | Purpose |
|---|---|---|
| `plotly` | >= 5.x | All interactive charts (already likely installed) |
| `pandas` | (already present) | Data manipulation for heatmap matrix |
| `numpy` | (already present) | Correlation matrix reordering |

No new heavyweight deps. Plotly is the only addition (check if already in requirements).

---

## 5. Key Design Decisions

1. **Read-only from artifacts**: The visualizers never re-run the pipeline. They only read JSON + txt artifacts. This makes them fast and safe.

2. **LLM rationale extraction**: `raw_responses.txt` contains full LLM JSON per block. We parse these blocks to extract `rationale`, `evidence_cited`, `alternatives_considered` per column decision. Matching is done by column name from `feature_artifact.json`.

3. **Plotly HTML (not matplotlib PNG)**: Plotly generates self-contained HTML files with embedded JS. No server needed. Hovering shows rationale text.

4. **Color convention across all charts**:
   - Green = selected / ok / used
   - Red = dropped / removed
   - Amber = fallback (LLM decision overridden by rule-based)
   - Blue = revised (LLM gave bad first attempt, revision was accepted)

5. **Output to `plots/` under run_id**: Follows CLAUDE.md rule 24 (organize output into named folders, not current dir).

6. **All imports at top of each file** (CLAUDE.md rule 1).

7. **No hardcoded paths**: paths resolved from `run_output_dir` argument (CLAUDE.md rule 7).

8. **OOP throughout**: Every visualizer is a class, not a free function (CLAUDE.md rule 9).

---

## 6. File Creation Order (when executing)

1. `visuals/artifact_reader.py` — foundation, everything else depends on it
2. `visuals/base.py` — abstract base
3. `visuals/importance.py`
4. `visuals/correlation.py`
5. `visuals/decisions.py` (covers imputation + outlier + scaler in one file)
6. `visuals/creation.py`
7. `visuals/selection.py`
8. `visuals/pca.py`
9. `visuals/timeline.py`
10. `visuals/dashboard.py`
11. `visuals/cli.py`
12. Update `visuals/__init__.py` to expose `VisualDashboard`

---

## 7. Scope Boundaries (what this does NOT do)

- Does NOT re-run any pipeline step
- Does NOT call any LLM
- Does NOT generate new features or modify `feature_artifact.json`
- Does NOT touch any existing pipeline code (orchestrator, tools, etc.)
- Does NOT generate matplotlib static images unless Plotly is unavailable
