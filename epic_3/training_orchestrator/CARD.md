---
name: epic_3/training_orchestrator
path: epic_3/training_orchestrator
purpose: Router and job configuration compiler that maps model candidates from model selection onto validated training jobs, generating the final execution manifest.
interfaces:
  inputs:
    - name: model_config.json
      format: JSON
      upstream: epic_3/model_selection
      description: Ranked candidates from model selection.
    - name: metadata.json
      format: JSON
      upstream: backend / agents / metadata_gen_agent
      description: Pipeline metadata specifying problem type, data formats, and column listings.
    - name: train_path / test_path
      format: CSV / NPZ
      upstream: backend / session / data_split
      description: Paths to preprocessed train/test splits.
  outputs:
    - name: training_jobs.json (TrainingJobManifest)
      format: JSON
      downstream: ray_executor / training
      description: Atomic JSON manifest listing validated, structured training jobs containing parameters, priority, and outputs directory definitions.
entry_points:
  - name: epic_3.training_orchestrator.orchestrator:TrainingOrchestrator
    type: Python API
    description: Core orchestrator class validating configurations and routing candidates.
  - name: epic_3.training_orchestrator.model_router:ModelRouter
    type: Python API
    description: Looks up model configurations in the MLKit catalog and determines appropriate TrainerType.
  - name: epic_3.training_orchestrator.cli:main
    type: CLI
    description: Executable CLI tool for preparing training jobs lists.
dependencies:
  - model_library
  - epic_3/model_selection
  - pydantic
---

# Technical Architecture: Training Orchestrator

## Overview
The `training_orchestrator` compiles ranked selection candidates into concrete `TrainingJob` configurations. It reads the default hyperparameters from the model library catalog and constructs a structured execution manifest (`training_jobs.json`) to serve as input for the Ray execution worker.

## Core Component Walkthrough
1. **`model_router.py`**:
   - `ModelRouter`: Resolves model wrappers against the available `model_library` catalog.
   - It guarantees that default parameters are sourced from the model library rather than trusting LLM guesses in `model_config.json`.
   - Maps each model descriptor to a `TrainerType` (`scikit-learn`, `pytorch`, or `xgboost`).
2. **`orchestrator.py`**:
   - `TrainingOrchestrator`: Entry point that loads files, parses schema formats with `pydantic` types, runs routing checks, creates separate model directory folders (`model_001`, `model_002`), and performs an atomic write of the manifest.
3. **`contracts.py`**: Defines standard Pydantic models for validation (`TrainingJob`, `TrainingJobManifest`, `SelectedModelConfig`).

## Interfacing Guide
- **Upstream Integration:** Receives outputs from `model_selection` (`model_config.json`) and data split paths from session directory metadata.
- **Downstream Integration:** Writes `training_jobs.json` to the session directory. The execution harness reads this file to spawn workers concurrently.

## Suggested Cleanup/Refactoring
- **Consolidate Schemas:** Merge `SelectedModelConfig` in `training_orchestrator` and `ModelCandidate` in `model_selection` to reduce duplicate schemas.
- **Remove Catalog Duplication:** Reusing the same catalog loader between selection and orchestration prevents catalog out-of-sync errors.
