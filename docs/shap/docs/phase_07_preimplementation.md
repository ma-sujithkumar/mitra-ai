# Phase 7 Pre-Implementation Review
**Date:** 2026-06-17
**Branch:** epic4-shap
**Preceding phase:** Phase 6 complete -- exporters implemented, SHAPResult in models/

---

## 1. Scope

Phase 7 implements the three visualization plots defined in spec.md Sections 16.1–16.3.
All inputs are already produced by Phase 5 (SHAPService) and frozen in SHAPResult
(Phase 6). Phase 7 is pure visualization I/O -- no SHAP computation, no CSV, no metadata.

Do NOT implement in this phase:
- Pipeline orchestration (Phase 8)
- Additional error handling beyond plot generation failures
- Any new SHAP computation logic

---

## 2. Visualization Architecture

### 2.1 Placement in the module graph

PlotGenerator lives in the `visualizations/` sub-package. It is the only class in that
package. It sits downstream of SHAPService and upstream of the pipeline orchestrator.

```
SHAPService (Phase 5)
    => SHAPResult (Phase 6)
        => PlotGenerator (Phase 7)
            => summary_plot.png
            => feature_importance_bar.png
            => beeswarm_plot.png
```

### 2.2 Dependency rule (star topology from architecture.md Section 2)

PlotGenerator imports from:
- `shap_explainability.models.shap_result` (SHAPResult)
- `shap_explainability.errors` (VisualizationError -- new, Phase 7)
- `shap_explainability.utils.logger` (ExecutionLogger)
- `shap_explainability.utils.output_manager` (OutputManager, for plot_path())

PlotGenerator does NOT import from `explainers/`, `exporters/`, `validators/`,
or `loaders/`.

### 2.3 matplotlib isolation

All matplotlib interaction is confined to `plot_generator.py`. No other module in the
pipeline imports `matplotlib.pyplot`. This is the single boundary for backend
configuration and figure lifecycle management.

`matplotlib.use("Agg")` must be called in `plot_generator.py` **before** importing
`matplotlib.pyplot` to guarantee headless operation in CI, server, and test
environments. This is a module-level side effect with no runtime branching.

### 2.4 Architecture.md discrepancy

architecture.md Section 5 step 10 writes:
> `PlotGenerator.render_all(shap_values, feature_names, output_manager)`

The actual signature requires `feature_dataframe` as well, because:
- summary_plot and beeswarm_plot use the feature matrix to color/shape points
  by actual feature values (the SHAP `features` argument)
- Without it, both plots degrade to unstyled bar-like output

Corrected signature: `render_all(shap_result, feature_dataframe, output_manager)`
This is a signature refinement, not a design change.

---

## 3. Required Classes

### 3.1 PlotGenerator (`src/shap_explainability/visualizations/plot_generator.py`)

One primary class. Responsibilities (architecture.md Section 3):
- One method per plot type (summary, bar, beeswarm).
- One combined `render_all()` that calls all three and returns paths.
- Injected with `output_path` values from `OutputManager.plot_path()` -- no
  internal path construction.
- Logs the `plot_generation` event before and after each file write.
- Raises `VisualizationError` on any matplotlib or I/O failure.

**Constructor:**
```
PlotGenerator(
    execution_logger: ExecutionLogger,
    plot_format: str,          # from AppConfig.plot_format ("PNG")
    max_display_features: int, # from AppConfig.max_display_features
)
```

**Methods:**
```
render_summary_plot(
    shap_result: SHAPResult,
    feature_dataframe: pd.DataFrame,
    output_path: Path,
) -> Path

render_feature_importance_bar(
    shap_result: SHAPResult,
    output_path: Path,
) -> Path

render_beeswarm_plot(
    shap_result: SHAPResult,
    feature_dataframe: pd.DataFrame,
    output_path: Path,
) -> Path

render_all(
    shap_result: SHAPResult,
    feature_dataframe: pd.DataFrame,
    output_manager: OutputManager,
) -> dict[str, Path]
```

`render_all()` returns a dict keyed by plot filename constant so the pipeline can
reference artifact paths without re-deriving them.

### 3.2 VisualizationError (`src/shap_explainability/errors.py`)

New exception class, added to errors.py alongside ExportError:
```
class VisualizationError(SHAPModuleError):
    """Raised when a plot cannot be generated or saved to disk."""
```

Wraps exceptions from `shap.summary_plot()`, `plt.savefig()`, or `shap.plots.beeswarm()`
so the pipeline failure path can catch all domain failures via the common
`SHAPModuleError` base.

### 3.3 AppConfig change (`src/shap_explainability/config_loader.py`)

One new field added to the frozen `AppConfig` dataclass:
```
max_display_features: int  # max features shown in any one plot (CFG-04 extension)
```

Read from `[plot] MAX_DISPLAY_FEATURES` in `config.ini`. Default value: 20.
If the dataset has fewer features than `max_display_features`, all features are shown.

`ConfigLoader.load()` reads and validates this key: must be a positive integer.

### 3.4 config.ini addition

```ini
[plot]
PLOT_FORMAT = PNG
MAX_DISPLAY_FEATURES = 20
```

No other section is touched.

---

## 4. SHAP Plot Mapping

| Plot name | Spec ref | Output filename | SHAP function | plot_type arg | Needs feature_dataframe |
|---|---|---|---|---|---|
| Summary plot | Sec 16.1 | summary_plot.png | `shap.summary_plot()` | default ("dot") | Yes -- feature value coloring |
| Feature importance bar | Sec 16.2 | feature_importance_bar.png | `shap.summary_plot()` | "bar" | No -- mean abs SHAP only |
| Beeswarm plot | Sec 16.3 | beeswarm_plot.png | `shap.summary_plot()` | "violin" | Yes -- distribution shaping |

**Design rationale:** All three plots use the SHAP legacy API (`shap.summary_plot()`).
This avoids the `shap.Explanation` wrapper complexity for multiclass arrays, which
requires stacking list-of-arrays into a 3D ndarray. The `plot_type` argument
differentiates the three outputs visually:
- "dot" -- colored scatter plot; shows direction and magnitude of SHAP contributions
- "bar" -- horizontal bar chart of mean(|SHAP|); shows global importance ranking
- "violin" -- violin distribution; shows SHAP value spread per feature

All three use `show=False` to suppress any GUI display. `plt.savefig()` follows
immediately after each call. `plt.close("all")` runs after every savefig to prevent
figure accumulation across multiple pipeline runs in the same process.

---

## 5. Summary Plot Strategy

**Spec ref:** Sec 16.1. "Global explainability overview."
**Output:** `plots/summary_plot.png`
**Objective:** Show each feature's SHAP contribution direction (positive/negative) and
magnitude, colored by the actual feature value (high/low), for all samples.

### 5.1 API call

```python
shap.summary_plot(
    shap_values,          # see 5.2 for per-prediction-type preparation
    features=feature_dataframe,
    feature_names=list(shap_result.feature_names),
    max_display=max_display_features,
    show=False,
)
plt.tight_layout()
plt.savefig(output_path, format=plot_format.lower(), bbox_inches="tight", dpi=150)
plt.close("all")
```

### 5.2 Multiclass handling

- Binary / Regression: pass `shap_result.shap_values_array` directly (2D ndarray).
- Multiclass: pass the list of K 2D ndarrays directly.
  `shap.summary_plot()` with a list input computes mean(|values|) across classes
  and renders a single summary across all classes. No manual stacking required.

---

## 6. Beeswarm Plot Strategy

**Spec ref:** Sec 16.3. "SHAP value distributions across samples."
**Output:** `plots/beeswarm_plot.png`
**Objective:** Show the distribution shape of SHAP values per feature using a violin
representation, making the spread/density visible in a way that the dot plot
compresses.

### 6.1 API call

```python
shap.summary_plot(
    shap_values,          # same preparation as Section 5.2
    features=feature_dataframe,
    feature_names=list(shap_result.feature_names),
    plot_type="violin",
    max_display=max_display_features,
    show=False,
)
plt.tight_layout()
plt.savefig(output_path, format=plot_format.lower(), bbox_inches="tight", dpi=150)
plt.close("all")
```

### 6.2 Multiclass handling

Same as Section 5.2: pass the list of K arrays directly. SHAP aggregates across
classes for the violin plot.

### 6.3 Distinction from summary plot

The summary plot (dot type) shows individual sample points and their sign (positive
or negative SHAP). The beeswarm (violin type) shows the density distribution of SHAP
values per feature -- complementary information, not redundant.

---

## 7. Feature Importance Bar Chart Strategy

**Spec ref:** Sec 16.2. "Ranked feature importance."
**Output:** `plots/feature_importance_bar.png`
**Objective:** Show features ranked by mean(|SHAP value|), providing a clean
model-global importance ranking without per-sample information.

### 7.1 API call

```python
shap.summary_plot(
    shap_values,          # same preparation as Section 5.2
    features=feature_dataframe,
    feature_names=list(shap_result.feature_names),
    plot_type="bar",
    max_display=max_display_features,
    show=False,
)
plt.tight_layout()
plt.savefig(output_path, format=plot_format.lower(), bbox_inches="tight", dpi=150)
plt.close("all")
```

### 7.2 Multiclass handling

For multiclass with `plot_type="bar"`, SHAP shows a stacked bar chart per class.
This is SHAP library behavior and requires no special handling -- passing the list
of K arrays activates this automatically.

### 7.3 Note: feature_dataframe is passed but not strictly required

`shap.summary_plot(plot_type="bar")` does not use feature values for rendering.
Passing `features=feature_dataframe` is harmless and keeps the API call uniform.
Alternatively, `render_feature_importance_bar()` may omit passing `feature_dataframe`
and set `features=None` to minimize memory footprint for very large datasets --
either is correct. The method signature still accepts it for interface consistency.

---

## 8. Output File Naming Strategy

### 8.1 Filename constants

Three module-level constants in `plot_generator.py`:

```
_SUMMARY_PLOT_FILENAME: str = "summary_plot.png"
_FEATURE_IMPORTANCE_BAR_FILENAME: str = "feature_importance_bar.png"
_BEESWARM_PLOT_FILENAME: str = "beeswarm_plot.png"
```

These filenames are spec-mandated (Sec 21). They are not read from config, not
parameterized, and not overridable by the caller.

### 8.2 Extension

The extension is always `.png`, derived from `AppConfig.plot_format = "PNG"` via
`plot_format.lower()` in the `savefig` call. The filename constant already includes
the extension to match the spec verbatim. If `PLOT_FORMAT` changes in config.ini
in a future phase, the constant names will also need updating -- this is intentional
coupling to the spec's specified filenames.

### 8.3 Path resolution

Paths are resolved exclusively via `OutputManager.plot_path(filename)`:
```
output_manager.plot_path(_SUMMARY_PLOT_FILENAME)
output_manager.plot_path(_FEATURE_IMPORTANCE_BAR_FILENAME)
output_manager.plot_path(_BEESWARM_PLOT_FILENAME)
```

`render_all()` calls `output_manager.plot_path()` internally and does not accept
raw path strings -- this keeps path knowledge inside `OutputManager`, as required
by architecture.md Section 3.

### 8.4 Directory creation

Each render method calls `output_path.parent.mkdir(parents=True, exist_ok=True)`
before writing, consistent with CLAUDE.md rule 17 and the pattern established in
Phase 6 exporters.

---

## 9. Files to Create

| File | Class | Purpose |
|---|---|---|
| `src/shap_explainability/visualizations/__init__.py` | -- | Package marker |
| `src/shap_explainability/visualizations/plot_generator.py` | `PlotGenerator` | Renders 3 PNG plots from SHAPResult |
| `tests/unit/test_plot_generator.py` | -- | Unit tests for PlotGenerator |

## 10. Files to Modify

| File | Change |
|---|---|
| `src/shap_explainability/errors.py` | Add `VisualizationError` |
| `src/shap_explainability/config_loader.py` | Add `max_display_features: int` to AppConfig; read `MAX_DISPLAY_FEATURES` from `[plot]` |
| `config/config.ini` | Add `MAX_DISPLAY_FEATURES = 20` to `[plot]` section |

---

## 11. Test Plan (`tests/unit/test_plot_generator.py`, ~15 tests)

Tests use small synthetic SHAPResult instances (no SHAP computation, pre-computed
numpy arrays from FixtureFactory or inline construction) and a tmp_path fixture.

| Test | Covers |
|---|---|
| `test_render_summary_plot_creates_nonempty_file` | AC-07 |
| `test_render_feature_importance_bar_creates_nonempty_file` | AC-08 |
| `test_render_beeswarm_plot_creates_nonempty_file` | AC-09 |
| `test_render_all_creates_all_three_files` | AC-07, AC-08, AC-09 |
| `test_render_all_returns_expected_paths` | render_all() return value |
| `test_summary_plot_binary_prediction_type` | SHAPResult with 2D ndarray |
| `test_summary_plot_multiclass_prediction_type` | SHAPResult with list of arrays |
| `test_summary_plot_regression_prediction_type` | SHAPResult with 2D ndarray |
| `test_beeswarm_plot_multiclass_prediction_type` | multiclass violin handling |
| `test_bar_chart_multiclass_prediction_type` | multiclass stacked bar handling |
| `test_parent_directory_created_if_missing` | mkdir -p before write |
| `test_visualization_error_raised_on_save_failure` | VisualizationError wrapping |
| `test_logger_called_for_plot_generation_event` | Sec 19 log event |
| `test_no_figure_leak_after_render_all` | plt.close("all") called |
| `test_max_display_features_respected` | max_display parameter forwarded |

---

## 12. Acceptance Criteria Traceability

| AC | Description | Phase 7 Coverage |
|---|---|---|
| AC-07 | summary_plot.png generated | `PlotGenerator.render_summary_plot()` |
| AC-08 | feature_importance_bar.png generated | `PlotGenerator.render_feature_importance_bar()` |
| AC-09 | beeswarm_plot.png generated | `PlotGenerator.render_beeswarm_plot()` |
| AC-14 | All artifacts stored under session-specific directory | `OutputManager.plot_path()` used exclusively |

---

## 13. Risks and Assumptions

### Risks

| ID | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R-01 | `shap.summary_plot()` with `plot_type="violin"` raises for a specific SHAP version | Low | Verify against installed SHAP version in environment; fall back to `"dot"` if violin unavailable |
| R-02 | Multiclass list-of-arrays passed to `shap.summary_plot()` produces unexpected shape errors for some model families | Medium | Unit test all three prediction types with small synthetic arrays; do not test with real model SHAP values |
| R-03 | `matplotlib.use("Agg")` is a no-op if pyplot was already imported before `plot_generator.py` is loaded | Medium | Ensure `plot_generator.py` is the first module to import matplotlib in the dependency chain; document in module docstring |
| R-04 | Figure state leaks between successive `render_all()` calls in integration tests | Medium | `plt.close("all")` after every savefig; verified by `test_no_figure_leak_after_render_all` |
| R-05 | Large feature count (e.g. 200 features) causes SHAP plot rendering to be very slow or memory-intensive | Low | `max_display_features` caps the number of features rendered per plot; default 20 |
| R-06 | Summary plot (dot) and beeswarm plot (violin) appear visually too similar to reviewers unfamiliar with SHAP | Low | Document the distinction in PlotGenerator docstring; this is a spec-mandated requirement, not a design decision |
| R-07 | `shap.summary_plot()` ignores `show=False` in some versions and opens a GUI window | Low | Use `plt.switch_backend("Agg")` as secondary guard inside each render method if `matplotlib.use("Agg")` has already been called at module import time |

### Assumptions

| ID | Assumption |
|---|---|
| A-01 | SHAP library >= 0.40 is installed in the `shap_epic4` conda environment; both legacy `shap.summary_plot()` and `plot_type="violin"` are available at this version |
| A-02 | `SHAPResult.shap_values_array` is a 2D ndarray for binary/regression and a list of K 2D ndarrays for multiclass -- produced by `SHAPService._normalize_shap_values()` in Phase 5 |
| A-03 | `feature_dataframe` passed to `render_all()` is the same cleaned, target-excluded DataFrame used for SHAP computation, with columns in the same order as `SHAPResult.feature_names` |
| A-04 | `OutputManager.initialize()` has already been called before `PlotGenerator.render_all()` is invoked; `plots/` subdirectory already exists; each `render_*` method's `mkdir -p` call is a safety net only |
| A-05 | `AppConfig.plot_format` is always "PNG" for Phase 1; the `plt.savefig(format=...)` call and filename constants are aligned |
| A-06 | matplotlib is installed in the `shap_epic4` conda environment at a version compatible with the installed SHAP library's internal plotting calls |

---

## 14. What Phase 8 Can Assume After Phase 7

- `PlotGenerator` importable from `shap_explainability.visualizations.plot_generator`
- `VisualizationError` importable from `shap_explainability.errors`
- `PlotGenerator(execution_logger, plot_format, max_display_features)` accepts an
  already-resolved `max_display_features` int (from `AppConfig`) -- no config reading
  inside PlotGenerator
- `PlotGenerator.render_all(shap_result, feature_dataframe, output_manager)` writes
  all three PNG files and returns a `dict[str, Path]` keyed by the three filename
  constants
- `AppConfig` has a new field `max_display_features: int`; `ConfigLoader.load()`
  reads it from `[plot] MAX_DISPLAY_FEATURES` -- Phase 8 pipeline construction must
  pass this value when constructing `PlotGenerator`
- All three PNG files land at `OutputManager.plot_path(filename)` paths, consistent
  with the session folder layout in architecture.md Section 1.2
