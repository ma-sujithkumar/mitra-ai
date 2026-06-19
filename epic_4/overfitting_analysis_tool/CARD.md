---
name: epic_4/overfitting_analysis_tool
path: epic_4/overfitting_analysis_tool
purpose: Overfitting analyzer that computes train-validation metric gaps, runs K-Fold cross validation, and determines overfitting status for trained models.
interfaces:
  inputs:
    - name: input_data (input.json)
      format: JSON
      upstream: backend / session / model_selection
      description: Specifies model_name, model_type, dataset_path, and precomputed train/test metrics.
    - name: dataset.npz
      format: NPZ (numpy zipped archive)
      upstream: backend / session / data_split
      description: Preprocessed train/test features and target splits.
    - name: config.yaml
      format: YAML
      upstream: config
      description: Specifies gap thresholds, epsilon boundaries, folding strategy parameters, and default metric directions.
  outputs:
    - name: overfitting_analysis_report.json
      format: JSON
      downstream: epic_4/judge_agent
      description: Overfitting diagnosis reporting holdout gaps, K-fold scores, mean/std variance, and is_overfitting boolean flag.
entry_points:
  - name: epic_4.overfitting_analysis_tool.overfitting_analysis:OverfittingAnalyzer
    type: Python API
    description: Core execution class loading datasets, training wrappers, computing metrics, and running cross-validation loops.
  - name: epic_4.overfitting_analysis_tool.overfitting_analysis:main
    type: CLI
    description: CLI script to trigger overfitting checks from a configuration input file.
dependencies:
  - model_library
  - numpy
  - pyyaml
  - scikit-learn
---

# Technical Architecture: Overfitting Analysis Tool

## Overview
The `overfitting_analysis_tool` determines if trained models suffer from overfitting or high-variance generalizations. It loads model wrappers, runs K-Fold or Stratified K-Fold cross-validations on train splits, computes holdout set gaps, and flags models that exceed configured thresholds (e.g. F1 gap > 0.15).

## Core Component Walkthrough
1. **`overfitting_analysis.py`**:
   - `OverfittingAnalyzer`: Loads configuration specifications from its internal `config.ini` / `config.yaml` mapping.
   - `load_dataset`: Extracts `X_train`, `y_train`, `X_test`, `y_test` arrays from `.npz` files.
   - `run_analysis`: Reconstructs or evaluates metric results across classification/regression schemas.
   - `run_kfold`: Spawns a K-fold splitting loop to evaluate stability, calculating standard deviation of validation metrics across folds.
   - Outputs a unified `overfitting_analysis_report.json` detailing CV scores, metric gaps, and structural overfitting alerts.

## Interfacing Guide
- **Upstream Integration:** Receives dataset paths and baseline metrics from active training session logs.
- **Downstream Integration:** Serves as a diagnostic report used by the Judge Agent to accept/reject models or trigger tuning retries.

## Suggested Cleanup/Refactoring
- **Consolidate Preprocessing:** The analyzer loads `.npz` splits directly. Standardizing the loader to read CSV data splits or share loader utilities with `epic_3/training` will remove duplicate parsing codes.
- **Ray Compatibility:** Run cross-validation folds as remote Ray tasks rather than sequentially, speeding up evaluation.
