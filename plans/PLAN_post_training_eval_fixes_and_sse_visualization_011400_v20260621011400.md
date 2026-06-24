# Plan: Post-Training Evaluation Fixes and Live SSE Visualization

This plan outlines the fixes and enhancements implemented to address model explainer failures and add live SSE streaming and HPT block visualization in the Mitra training pipeline.

## 1. Problem Statement
The post-training evaluation phase (consisting of SHAP explainers, overfitting audits, and Optuna hyperparameter tuning (HPT) leading into the LLM Judge ranking) suffered from:
- A configuration-coverage gap in SHAP model-family detection for models like SVC, NuSVC, HistGradientBoosting, ExtraTrees, GradientBoosting, LinearSVC, and Ridge.
- ExplainerFactory failing because Kernel SVMs (like SVC and NuSVC) were not mapped to any explainer type.
- Lack of live feedback during the post-training evaluation phase on the frontend.
- Absence of a dedicated Optuna/HPT progress block in the UI.

## 2. Implemented Fixes

### 2.1 Backend SHAP Model Mappings and Explainers
- **`model_type_detection.json`**:
  - Mapped `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` to the `HistGradientBoosting` family (associated with `TreeExplainer`).
  - Mapped `ExtraTreesClassifier` and `ExtraTreesRegressor` to the `ExtraTrees` family (associated with `TreeExplainer`).
  - Mapped `GradientBoostingClassifier` and `GradientBoostingRegressor` to the `GradientBoosting` family (associated with `TreeExplainer`).
  - Mapped `LinearSVC` and `LinearSVR` to the `LinearSVM` family (associated with `LinearExplainer`).
  - Mapped `RidgeClassifier` and `Ridge` to the `Ridge` family (associated with `LinearExplainer`).
  - Mapped `SVC`, `NuSVC`, `SVR`, and `NuSVR` to the `KernelSVM` family (associated with a new `KernelExplainer`).
- **`explainer_factory.py`**:
  - Implemented `_build_kernel_explainer` which maps to `shap.KernelExplainer` for kernel SVMs.
  - Used a small random sample of background data (capped at 100 rows) to ensure extremely fast SHAP computation for SVMs.

### 2.2 Live SSE Evaluation Progress
- **`contracts.py`**:
  - Extended the `TrainingEvent` schema to support `stage: Literal["training", "evaluation"]`.
- **`training_service.py`**:
  - Emitted `TrainingEvent` updates at each stage of the post-training evaluation chain (SHAP/overfitting/HPT start, LLM Judge start, plotting, completion).
  - Ensured that `close_session` is called inside a `finally` block of `_run_post_training_evaluation` or immediately when evaluation is skipped/cancelled, allowing the EventSource to cleanly close exactly when all work settles.

### 2.3 Frontend HPT Live Rendering
- **`client.js`**:
  - Added the `fetchHpt(sessionId)` API client method mapping to `/api/runs/{session_id}/hpt`.
- **`evaluation.py`**:
  - Added a backend GET route at `/api/runs/{session_id}/hpt` to expose the JSON study tuning results from Optuna.
- **`TrainingPage.jsx`**:
  - Polled the HPT endpoint periodically when connected to a session.
  - Rendered a custom `<HptTuningSection>` block that displays a loading/optimizing placeholder during evaluation and details of tuned hyperparameters (validation score, tuning time, parameters) once complete.
