# Phase 7 Implementation
**Date:** 2026-06-18
**Branch:** epic4-shap
**Preceding phase:** Phase 6 complete -- exporters implemented, SHAPResult in models/
**Tests at phase start:** 187 (Phases 1-6)
**Tests at phase end:** 273 (86 new Phase 7 tests)

---

## 1. Files Created

| File | Purpose |
|---|---|
| `src/shap_explainability/visualizations/__init__.py` | Package marker; configures matplotlib Agg backend before any pyplot import |
| `src/shap_explainability/visualizations/summary_plot_generator.py` | `SummaryPlotGenerator` class -- renders summary_plot.png |
| `src/shap_explainability/visualizations/beeswarm_plot_generator.py` | `BeeswarmPlotGenerator` class -- renders beeswarm_plot.png |
| `src/shap_explainability/visualizations/feature_importance_plot_generator.py` | `FeatureImportancePlotGenerator` class -- renders feature_importance_bar.png |
| `tests/unit/test_summary_plot_generator.py` | 15 unit tests for SummaryPlotGenerator |
| `tests/unit/test_beeswarm_plot_generator.py` | 15 unit tests for BeeswarmPlotGenerator |
| `tests/unit/test_feature_importance_plot_generator.py` | 17 unit tests for FeatureImportancePlotGenerator |

---

## 2. Files Modified

| File | Change |
|---|---|
| `src/shap_explainability/errors.py` | Added `VisualizationError(SHAPModuleError)` |
| `src/shap_explainability/config_loader.py` | Added `max_display_features: int = 20` to `AppConfig`; reads `[plot] MAX_DISPLAY_FEATURES` with fallback |
| `config/config.ini` | Added `MAX_DISPLAY_FEATURES = 20` to `[plot]` section |
| `tests/conftest.py` | Added `MAX_DISPLAY_FEATURES = 20` to default `[plot]` section written by `config_ini_writer` |
| `tests/fixtures/fixture_factory.py` | Added `make_feature_dataframe`, `make_shap_result_binary`, `make_shap_result_regression`, `make_shap_result_multiclass` static methods |

---

## 3. Classes Created

### 3.1 SummaryPlotGenerator
- **File:** `src/shap_explainability/visualizations/summary_plot_generator.py`
- **Constructor:** `SummaryPlotGenerator(execution_logger, plot_format, max_display_features)`
- **Method:** `render(shap_result, feature_dataframe, output_path) -> Path`
- **Plot:** `summary_plot.png` -- dot scatter, shows per-sample SHAP direction and magnitude colored by feature value
- **SHAP API:** `shap.summary_plot(shap_values_array, features=feature_dataframe, ...)` -- default plot_type ("dot")

### 3.2 BeeswarmPlotGenerator
- **File:** `src/shap_explainability/visualizations/beeswarm_plot_generator.py`
- **Constructor:** `BeeswarmPlotGenerator(execution_logger, plot_format, max_display_features)`
- **Method:** `render(shap_result, feature_dataframe, output_path) -> Path`
- **Plot:** `beeswarm_plot.png` -- violin distribution, shows SHAP value density/spread per feature
- **SHAP API:** `shap.summary_plot(shap_values_array, features=feature_dataframe, plot_type="violin", ...)`

### 3.3 FeatureImportancePlotGenerator
- **File:** `src/shap_explainability/visualizations/feature_importance_plot_generator.py`
- **Constructor:** `FeatureImportancePlotGenerator(execution_logger, plot_format, max_display_features)`
- **Method:** `render(shap_result, output_path) -> Path`
- **Plot:** `feature_importance_bar.png` -- horizontal bar chart of mean(|SHAP|), features ranked descending
- **SHAP API:** `shap.summary_plot(shap_values_array, features=None, plot_type="bar", ...)`
- **Note:** No `feature_dataframe` argument -- bar charts use mean absolute SHAP values only

---

## 4. Plot Generation Strategy

All three generators use the legacy `shap.summary_plot()` API (not the modern `shap.Explanation` wrapper). The `plot_type` argument differentiates the three outputs:

| Generator | plot_type | Needs feature_dataframe | Output |
|---|---|---|---|
| SummaryPlotGenerator | "dot" (default) | Yes -- colors points by feature value | summary_plot.png |
| BeeswarmPlotGenerator | "violin" | Yes -- shapes violin by feature value distribution | beeswarm_plot.png |
| FeatureImportancePlotGenerator | "bar" | No -- only mean(|SHAP|) used | feature_importance_bar.png |

**Common pattern in all three render() methods:**
1. Log plot_generation event (start)
2. `output_path.parent.mkdir(parents=True, exist_ok=True)` -- safety mkdir
3. Call `shap.summary_plot(...)` with `show=False`
4. `plt.tight_layout()` then `plt.savefig(output_path, format="png", bbox_inches="tight", dpi=150)`
5. `plt.close("all")` in a `finally` block -- guaranteed even on failure
6. Log plot_generation event (done)
7. Return output_path

**matplotlib backend:** `matplotlib.use("Agg")` is called once in `visualizations/__init__.py`. Python always imports a package's `__init__.py` before any submodule, so the backend is set before any `import matplotlib.pyplot` in the three generator modules. This guarantees headless operation in CI, server, and test environments.

**Error handling:** Any exception from `shap.summary_plot()` or `plt.savefig()` is caught and re-raised as `VisualizationError(SHAPModuleError)`, ensuring the pipeline's base-exception handler can catch it uniformly.

---

## 5. Multiclass Visualization Handling

All three generators pass `shap_result.shap_values_array` directly to `shap.summary_plot()` without branching on prediction type. The SHAP library handles both inputs natively:

- **Binary / Regression:** `shap_values_array` is a 2D ndarray shape `(n_samples, n_features)` -- passed as-is.
- **Multiclass:** `shap_values_array` is a list of K 2D ndarrays each `(n_samples, n_features)` -- passed as-is.
  - Summary and beeswarm: SHAP aggregates across classes (mean of per-class contributions).
  - Bar chart: SHAP renders stacked bars, one stack segment per class, automatically when given a list.

No `np.stack()` or manual aggregation is performed -- the list-of-arrays form produced by `SHAPService._normalize_shap_values()` (Phase 5) is exactly what the SHAP legacy API accepts.

---

## 6. Test Coverage

### 6.1 test_summary_plot_generator.py (15 tests)

| Test | Validates |
|---|---|
| `test_render_creates_nonempty_file` | PNG file exists and is non-empty after render |
| `test_render_returns_output_path` | render() return value equals output_path |
| `test_render_creates_parent_directory_if_missing` | mkdir -p before write |
| `test_render_binary_prediction_type` | 2D ndarray shap_values_array accepted |
| `test_render_regression_prediction_type` | Regression prediction type accepted |
| `test_render_multiclass_prediction_type` | List-of-arrays shap_values_array accepted |
| `test_render_passes_shap_values_array_to_summary_plot` | Correct first positional arg |
| `test_render_passes_feature_names_to_summary_plot` | feature_names kwarg forwarded |
| `test_render_passes_show_false_to_summary_plot` | show=False prevents GUI display |
| `test_render_passes_max_display_features` | max_display kwarg forwarded |
| `test_render_raises_visualization_error_on_shap_failure` | VisualizationError on shap failure |
| `test_render_raises_visualization_error_on_save_failure` | VisualizationError on savefig failure |
| `test_render_logs_plot_generation_events` | log_plot_generation called >= 2 times |
| `test_no_figure_leak_after_render` | plt.close("all") called on success |
| `test_figures_closed_even_on_error` | plt.close("all") called in finally on failure |

### 6.2 test_beeswarm_plot_generator.py (15 tests)

Same structure as summary tests, plus:

| Test | Validates |
|---|---|
| `test_render_passes_violin_plot_type` | plot_type="violin" passed to SHAP |

### 6.3 test_feature_importance_plot_generator.py (17 tests)

Same structure as summary tests, plus:

| Test | Validates |
|---|---|
| `test_render_passes_bar_plot_type` | plot_type="bar" passed to SHAP |
| `test_render_passes_features_none` | features=None passed (no feature_dataframe needed) |

---

## 7. Test Results

```
============================= test session starts =============================
collected 273 items

tests/unit/test_beeswarm_plot_generator.py         15 passed
tests/unit/test_config_loader.py                   11 passed
tests/unit/test_dataset_loader.py                  11 passed
tests/unit/test_explainer_factory.py               15 passed
tests/unit/test_feature_importance_plot_generator.py  17 passed
tests/unit/test_feature_shap_mapping_exporter.py   11 passed
tests/unit/test_fixture_factory.py                  5 passed
tests/unit/test_global_importance_exporter.py      10 passed
tests/unit/test_logger.py                          17 passed
tests/unit/test_metadata_exporter.py               15 passed
tests/unit/test_model_loader.py                    17 passed
tests/unit/test_model_validator.py                 18 passed
tests/unit/test_output_manager.py                   9 passed
tests/unit/test_schema_validator.py                18 passed
tests/unit/test_session_context.py                 13 passed
tests/unit/test_shap_service.py                    36 passed
tests/unit/test_summary_plot_generator.py          15 passed

273 passed, 3 warnings in 6.54s
```

3 warnings are `PendingDeprecationWarning` from the SHAP library's internal colormap setup -- unrelated to Phase 7 code.

---

## 8. What Phase 8 Can Assume After Phase 7

- `SummaryPlotGenerator` importable from `shap_explainability.visualizations.summary_plot_generator`
- `BeeswarmPlotGenerator` importable from `shap_explainability.visualizations.beeswarm_plot_generator`
- `FeatureImportancePlotGenerator` importable from `shap_explainability.visualizations.feature_importance_plot_generator`
- `VisualizationError` importable from `shap_explainability.errors`
- All three constructors accept `(execution_logger, plot_format, max_display_features)` from `AppConfig`
- `SummaryPlotGenerator.render(shap_result, feature_dataframe, output_path) -> Path`
- `BeeswarmPlotGenerator.render(shap_result, feature_dataframe, output_path) -> Path`
- `FeatureImportancePlotGenerator.render(shap_result, output_path) -> Path`
- All three write to `OutputManager.plot_path(filename)` paths
- `AppConfig` has a new field `max_display_features: int` (default 20); `ConfigLoader.load()` reads it from `[plot] MAX_DISPLAY_FEATURES` with fallback
- `FixtureFactory` has `make_shap_result_binary`, `make_shap_result_regression`, `make_shap_result_multiclass`, and `make_feature_dataframe` methods available for integration test construction

---

## 9. Design Notes

### 9.1 matplotlib backend initialization
`matplotlib.use("Agg")` lives in `visualizations/__init__.py`, not in the individual generator modules. Python always executes a package's `__init__.py` before any submodule import, so the backend is guaranteed to be set before any `import matplotlib.pyplot` in `summary_plot_generator.py`, `beeswarm_plot_generator.py`, or `feature_importance_plot_generator.py`, regardless of which generator class the caller imports first. If the generators were imported directly as top-level scripts (bypassing the package), the backend would fall back to whatever was already active -- this is an acceptable risk documented in architecture.md Risk R-03.

### 9.2 Legacy shap.summary_plot() API chosen over shap.Explanation wrapper
All three generators use the `shap.summary_plot()` legacy API instead of the modern `shap.Explanation` wrapper. The reason: the `shap.Explanation` wrapper requires multiclass SHAP values to be stacked into a 3D ndarray `(n_samples, n_features, n_classes)`, but `SHAPResult.shap_values_array` stores them as a list of K 2D ndarrays -- the form produced by `SHAPService._normalize_shap_values()`. Converting between these forms requires explicit `np.stack()` and axis transposition, adding complexity with no gain. The legacy API accepts both 2D ndarrays and lists of arrays directly, making the call site prediction-type-agnostic.

### 9.3 No prediction-type branching in render methods
All three `render()` methods pass `shap_result.shap_values_array` directly to SHAP without checking `shap_result.prediction_type`. The SHAP library detects the input type internally. This eliminates a class of bugs where a new prediction type would require changes to the generator code, and keeps the render methods single-responsibility -- they own the matplotlib lifecycle, not SHAP value interpretation.

### 9.4 plt.close("all") in finally blocks
`plt.close("all")` is placed in a `finally` block in each render method, not in the success path only. This ensures figures are cleaned up even when `shap.summary_plot()` or `plt.savefig()` raises an exception. Without this, repeated failures in the same process (e.g., in a test suite) would accumulate open matplotlib figures, degrading performance and eventually triggering matplotlib's `MaxNLocator` or figure count warnings.

### 9.5 FeatureImportancePlotGenerator takes no feature_dataframe
`FeatureImportancePlotGenerator.render()` has no `feature_dataframe` parameter. The bar chart (`plot_type="bar"`) computes `mean(|SHAP values|)` per feature and renders horizontal bars -- it does not use actual feature values for coloring or shaping. Passing `features=None` explicitly to `shap.summary_plot()` avoids any potential shape mismatch if a mismatched DataFrame were passed. The asymmetric signature compared to the other two generators is intentional and spec-correct (spec.md Sec 16.2).

### 9.6 AppConfig.max_display_features default and fallback
`AppConfig.max_display_features` has a Python-level default of `20` in the frozen dataclass. `ConfigLoader.load()` reads `[plot] MAX_DISPLAY_FEATURES` with `configparser`'s `fallback="20"`, so existing `config.ini` files that predate Phase 7 (which lack the key) continue to work without a `ConfigValidationError`. When the key is present, the loader validates it is a positive integer and raises `ConfigValidationError` if not. This preserves backward compatibility while enforcing correctness on new explicit configurations.
