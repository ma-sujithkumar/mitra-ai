---
name: model_library
path: model_library
purpose: Unified model registry and wrapper framework providing standard scikit-learn, XGBoost, and PyTorch wrappers for classification and regression tasks.
interfaces:
  inputs:
    - name: DataBundle / CommonData
      format: Python objects wrapping numpy features/labels
      upstream: training / overfitting_analysis_tool
      description: Train/test dataset splits loaded into RAM.
    - name: model_name + hyperparameters
      format: string + dictionary
      upstream: model_selection / training_orchestrator
      description: Valid estimator name from registry and hyperparameter arguments.
  outputs:
    - name: trained model wrapper
      format: Python Object / pickled class
      downstream: training / overfitting_analysis_tool
      description: Trained wrapper exposing .predict() and .predict_proba() APIs.
    - name: MetricsResult
      format: Python Object / metrics dict
      downstream: training / evaluator / judge
      description: Accuracy, F1-macro, MSE, MAE, R², etc. computed on the splits.
entry_points:
  - name: model_library.ml_kit:MLKit
    type: Python API / Class
    description: Core library facade exposing model initialization, loading, and validation helper functions.
  - name: model_library.ml_kit:MODEL_REGISTRY
    type: Python Dictionary Registry
    description: Authoritative map of 60+ model names (LogisticRegression, XGBClassifier, PyTorchCNNRegressor, etc.) to wrapper classes.
  - name: model_library.metrics.evaluators:compute_metrics
    type: Python API
    description: Computes task-specific metrics and handles NaN/inf constraints.
dependencies:
  - scikit-learn
  - xgboost
  - torch
  - numpy
  - pandas
---

# Technical Architecture: Model Library

## Overview
The `model_library` (often referenced as `MLKit`) is a unified, framework-agnostic registry for machine learning models. It wraps scikit-learn estimators, XGBoost packages, and PyTorch neural networks inside a common `BaseModel` interface. This eliminates direct framework imports in the upstream orchestrators and ensures standard metric evaluation.

## Core Component Walkthrough
1. **`ml_kit.py`**: Exports the unified registry `MODEL_REGISTRY` and provides static utility tools for model instantiation and type validation.
2. **`core/data_bundle.py`**: Exports `DataBundle` (holds raw data splits) and `CommonData` (lightweight array containers), wrapping training resources safely.
3. **`models/base.py`**: Exposes the abstract `BaseModel` API which requires subclasses to implement:
   - `fit(X, y)`
   - `predict(X)`
   - `predict_proba(X)` (for classifiers)
4. **`models/classifiers/` & `models/regressors/`**: Contain wrapper classes (e.g. `XGBClassifierWrapper`, `PyTorchCNNRegressorWrapper`, `RandomForestClassifierWrapper`) that handle parameter mapping, device placement (CPU/GPU), and conversion of inputs into PyTorch Tensors.
5. **`metrics/evaluators.py`**: Computes classification matrices (accuracy, f1-macro, weighted f1, precision, recall) and regression evaluation statistics (MSE, RMSE, MAE, R2).

## Interfacing Guide
- **Upstream Integration:** Imported by `model_selection`, `training`, `training_orchestrator`, and `overfitting_analysis_tool` to validate model names and run training jobs.
- **Downstream Integration:** Exports serialized pickle weights and metrics outputs.

## Suggested Cleanup/Refactoring
- **Consolidate Config Loading:** The package houses a custom `core/config_loader.py` which duplicates config reading logic in the `backend`. Standardizing on a single config loader across the project will reduce maintenance overhead.
- **Pickle Security:** Replace standard pickle-based model loading with a safer serializing package (e.g., ONNX, joblib, or safetensors) to prevent security risks associated with untrusted model loading.
