---
name: epic_3/model_selection
path: epic_3/model_selection
purpose: Agentic pipeline step that matches dataset profiles (from metadata and feature selection) against MLKit models library registry, outputting ranked candidate model configurations.
interfaces:
  inputs:
    - name: metadata.json
      format: JSON
      upstream: backend / agents / metadata_gen_agent
      description: Columns, problem type, data formats, and class balance profile.
    - name: feature_selection.json
      format: JSON
      upstream: backend / agents / feature_selection (or upstream features agent)
      description: Lists of kept, dropped, and engineered feature columns.
    - name: mini_data.csv
      format: CSV (Optional)
      upstream: backend / session / mini_data
      description: Capped dataset sample for verifying shape/null density.
  outputs:
    - name: model_config.json
      format: JSON
      downstream: epic_3/training_orchestrator
      description: List of ranked candidate models, including default Hyperparameter search spaces and expected resource priorities.
    - name: model_selection_report.json
      format: JSON (Optional)
      downstream: backend / session
      description: Detailed selection rationale, catalog constraints, and fallback status.
entry_points:
  - name: epic_3.model_selection.selector:select_models
    type: Python API
    description: Convienence wrapper that initializes ModelSelectionOrchestratorAgent and outputs candidate configurations.
  - name: epic_3.model_selection.agents:ModelLibraryCatalogAgent
    type: Python API
    description: Automatically scans the MLKit MODEL_REGISTRY and retrieves valid class names.
  - name: epic_3.model_selection.cli:main
    type: CLI
    description: Executable CLI script for launching model selection independently.
dependencies:
  - model_library
  - pydantic
  - google.adk
---

# Technical Architecture: Model Selection

## Overview
The `model_selection` submodule implements a multi-agent sub-pipeline to match dataset profiles against the set of estimators defined in `model_library.ml_kit`. It ensures only models existing in the catalog are suggested, preventing model hallucinations.

## Core Component Walkthrough
1. **`catalog.py`**: Interacts with the `model_library` module to verify registered wrappers.
2. **`agents.py`**:
   - `DatasetProfileAgent`: Extracts a lightweight `DatasetProfile` from metadata and feature selections.
   - `LLMModelRankingAgent`: Translates the profile and catalog list into a strict Jinja2 prompt, requesting the LLM to select and rank optimal estimators.
   - `DeterministicRankingAgent`: Serves as a zero-network fallback, ranking models heuristically (e.g. tree-based models for high-dimensional tabular data, simple linear models for small tabular data).
   - `ModelSelectionValidationAgent`: Performs strict schema validations, ensuring no suggested model names fall outside the catalog.
   - `ModelSelectionOrchestratorAgent`: Orchestrates the agents sequentially, writing output configuration to `model_config.json`.
3. **`schemas.py`**: Defines input/output dataclass contracts with `pydantic`.

## Interfacing Guide
- **Upstream Integration:** Consumes `metadata.json` and `feature_selection.json` from preceding pipeline steps.
- **Downstream Integration:** Writes the ranked JSON candidate array to `model_config.json` inside the session directory, serving as input for `training_orchestrator`.

## Suggested Cleanup/Refactoring
- **Unsupervised Model Selection:** Add catalog support and routing for clustering/anomaly detection models if unsupervised tasks are enabled.
- **Unified config.ini Lookup:** Ensure `cli.py` references path mappings using the global config loader, reducing ad-hoc environment checks.
- **Remove Duplications:** Ensure metadata-parsing logic is shared with the backend schemas to avoid field naming disparities.
