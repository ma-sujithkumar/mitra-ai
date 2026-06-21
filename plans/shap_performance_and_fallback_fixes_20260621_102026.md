# Plan: SHAP Fallback and Performance Fixes

## 1. Objectives
- **Robust SHAP Computation**: Fall back to `KernelExplainer` when `TreeExplainer` construction fails for a tree-based model (such as scikit-learn's `GradientBoostingClassifier` on multi-class classification tasks).
- **Performance Guard for KernelExplainer**: Subsample the evaluation dataset to `20` rows if `KernelExplainer` is selected (either natively or via fallback). This prevents the pipeline from stalling or taking hours to evaluate thousands of perturbed model evaluations via black-box inference.
- **Strict Adherence to Guidelines**: Clean OOP patterns, typed arguments, descriptive variables, proper logging.

## 2. Changes Proposed

### A. Update `ExplainerFactory` in [explainer_factory.py](file:///home/sujithma/mitra/backend/agents/evaluation/shap/explainers/explainer_factory.py)
- Modify `_dispatch_explainer_build` to pass `feature_dataframe` to `_build_tree_explainer`.
- Modify `_build_tree_explainer` to accept `feature_dataframe`.
- Wrap `shap.TreeExplainer(model_object)` inside `_build_tree_explainer` with a `try-except` block.
- If it fails, log the warning and fall back to `self._build_kernel_explainer(model_object, feature_dataframe)`. Ensure that the returned `explainer_type_name` / `explainer_name` is updated or set to `"KernelExplainer"` when fallback occurs.
- To handle this, we can return the constructed explainer object and the updated explainer name from `_dispatch_explainer_build`, rather than having it static from the mapping config.

### B. Update `SHAPRunner` in [runner.py](file:///home/sujithma/mitra/backend/agents/evaluation/shap/runner.py)
- After calling `ExplainerFactory.create()`, check if `built_explainer.explainer_name == "KernelExplainer"`.
- If it is `"KernelExplainer"`, check if `len(feature_df) > 20`.
- If so, subsample `feature_df` to `20` rows using `.sample(n=20, random_state=42)` and log the action.

## 3. Verification Plan
- Run existing unit tests:
  ```bash
  ~/venv/bin/pytest backend/agents/evaluation/shap/tests/
  ```
- Verify training and evaluation in the application.
