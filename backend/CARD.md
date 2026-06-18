---
name: backend
path: backend
purpose: FastAPI backend server coordinating session states, dataset uploads, validation, metadata generation, and job status tracking.
interfaces:
  inputs:
    - name: dataset_file
      format: CSV / XLS / XLSX
      upstream: User Upload UI (Page 1)
      description: Raw dataset uploaded by the user.
    - name: metadata_file
      format: JSON / CSV
      upstream: User Upload UI (Page 1) (Optional)
      description: User-provided metadata overrides or hints.
  outputs:
    - name: upload summary
      format: JSON REST response
      downstream: Frontend UI
      description: Returns rows count, column count, and types detected.
    - name: validation_report.json
      format: JSON
      downstream: Frontend UI / Pipeline orchestrator
      description: Blockers and warnings found during data validation.
    - name: metadata.json
      format: JSON
      downstream: epic_3/model_selection / training_orchestrator
      description: Structured column types, target column, row count, and problem type.
entry_points:
  - name: backend.main:app
    type: REST endpoint
    description: FastAPI app instance containing all router configurations.
  - name: backend.validator:DataValidator
    type: Python API
    description: Handles chunked CSV checks (format, row count, null density, variance, PII, target separability).
  - name: backend.agents.metadata_gen_agent:MetadataAgentRunner
    type: Python API / ADK Agent
    description: Invokes the ADK LlmAgent to parse data description and output metadata.json.
dependencies:
  - fastapi
  - pandas
  - google.adk
  - config.ini
---

# Technical Architecture: Backend

## Overview
The `backend` submodule acts as the orchestrator API for the MITRA platform. It serves REST endpoints for uploading files, validating datasets, and calling LLM-based metadata generation. It leverages `google.adk` for the metadata generation agent.

## Core Component Walkthrough
1. **`main.py`**: Initializes the FastAPI app, configures logging to ensure stdout/stderr capture, loads configuration, manages CORS middleware, and mounts routers.
2. **`validator.py`**: Contains `DataValidator`, a rule-based validator performing data validation in chunks (default: 50,000 rows) to prevent memory OOM. It outputs `ValidationReport` with keys: `format`, `rows`, `nulls`, `variance`, `pii`, and `target`.
3. **`session.py`**: Contains `SessionManager`, which manages directory storage in `.mitra/sessions/<session_id>` and tracks recent uploads.
4. **`mini_data.py`**: Caches a 1000-row sample of the uploaded dataset called `mini_data.csv` to avoid exposing the full dataset to the LLM agent, saving tokens and preserving privacy.
5. **`agents/metadata_gen_agent.py`**: Integrates `LiteLlm` and `LlmAgent` from ADK. It reads `mini_data.csv` and user descriptions to infer target column and task type (classification vs regression vs unsupervised).

## Interfacing Guide
- **Upstream Integration:** Frontend React app posts files to `/api/upload` and receives a `session_id`. It then queries `/api/validate` and `/api/metadata` to advance the pipeline.
- **Downstream Integration:** Saves artifacts (`metadata.json`, `mini_data.csv`, and training data splits) inside the session directory. Upstream paths are stored in `session_config.yaml`. Downstream training and selection agents read from this directory.

## Suggested Cleanup/Refactoring
- **A2A / Unified Agent Structure:** `MetadataAgentRunner` should use a shared LLM factory rather than loading environment variables manually in the agent files.
- **Common Directory Structure:** Align session storage paths with a globally shared config manager rather than creating custom path resolvers in both `backend` and `epic_3`.
- **Remove Duplications:** Ensure `mini_data.py` and `validator.py` reuse the same CSV reader configuration to prevent mismatch in parsing column headers.
