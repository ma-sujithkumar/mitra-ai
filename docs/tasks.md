# MITRA E2E Demo Integration Task List

This document lists the low-level tasks required to connect all epics (Epics 1, 2, 3, 4) and the React/Vite frontend into a single, fully functional end-to-end AutoML pipeline demo.

---

## 1. PRE-TRAINING INTEGRATION (EPIC 2 & EPIC 3)

### [TASK-1.1] Create Pre-Training Pipeline Service
*   **Description:** Implement `backend/services/pipeline_prep.py` to orchestrate preprocessing before model training.
*   **File to create:** `backend/services/pipeline_prep.py`
*   **Details:**
    *   Initialize and execute Epic 2's `FeatureEngineerOrchestrator` on the session's raw dataset (`data.csv`).
    *   Output `engineered_dataset.csv` and `feature_artifact.json` in the session's directory.

### [TASK-1.2] Build Epic 2 to Epic 3 Contract Adapter
*   **Description:** Convert Epic 2's feature selection output format to match Epic 3's expectation.
*   **File to edit:** `backend/services/pipeline_prep.py`
*   **Details:**
    *   Read `feature_artifact.json` (specifically `selected_columns` and `dropped_columns`).
    *   Write `feature_selection.json` with keys: `keep`, `drop`, `engineered`, `rationale`.

### [TASK-1.3] Implement Dataset Train/Test Splitter
*   **Description:** Split the preprocessed dataset into train/test files.
*   **File to edit:** `backend/services/pipeline_prep.py`
*   **Details:**
    *   Split `engineered_dataset.csv` into `train.csv` and `test.csv` (using split ratio from `config.ini`).
    *   Save both to `.mitra/<session_id>/data/`.

### [TASK-1.4] Run Model Selection to Generate `model_config.json`
*   **Description:** Run the Epic 3 model selection agent.
*   **File to edit:** `backend/services/pipeline_prep.py`
*   **Details:**
    *   Call `select_models` from `epic_3/model_selection/selector.py`.
    *   Pass `metadata.json` and the adapted `feature_selection.json`.
    *   Write `model_config.json` in the session directory.

### [TASK-1.5] Register Bridge Hook in Training Service
*   **Description:** Inject the preparation pipeline as a precondition to Ray training.
*   **File to edit:** `backend/services/training_service.py`
*   **Details:**
    *   In `TrainingService.start()`, check if `model_config.json`, `train.csv`, and `test.csv` are missing.
    *   If missing, run tasks 1.1 - 1.4 asynchronously before launching Ray cluster jobs.

---

## 2. POST-TRAINING INTEGRATION (EPIC 4)

### [TASK-2.1] Build SHAP Pipeline CLI Runner
*   **Description:** Provide a script to run SHAP evaluation on a trained model wrapper.
*   **File to create:** `epic_4/SHAP/run_shap.py`
*   **Details:**
    *   Implement standard CLI parsing for: `--session_id`, `--model_name`, `--pickle_file_path`, `--engineered_dataset_path`, `--output_dir`.
    *   Instantiate `SHAPService`, generate metrics, and write explainability PNGs (`feature_importance_bar.png` and `summary_plot.png`).

### [TASK-2.2] Create Post-Training Evaluation Orchestrator
*   **Description:** Orchestrate overfitting analysis, SHAP, and the Judge agent.
*   **File to create:** `epic_4/run_evaluation_pipeline.py`
*   **Details:**
    *   For each trained model, run `OverfittingAnalyzer` to compute validation metric gaps and CV scores.
    *   Execute SHAP CLI Runner on the top model candidate.
    *   Combine metrics, overfitting reports, and SHAP highlights into `JudgeInput`.
    *   Execute the `JudgeAgent` rule engine and LLM to nominee the winner and write `judge_decision.json`.

### [TASK-2.3] Hook Post-Training Steps into Training Service
*   **Description:** Execute evaluation pipeline immediately after Ray jobs finish.
*   **File to edit:** `backend/services/training_service.py`
*   **Details:**
    *   Once the Ray worker sweeps complete and write `training_summary.json`, trigger `run_evaluation_pipeline.py` as a background subprocess.

### [TASK-2.4] Integrate dataset2Vec in Model Selection
*   **Description:** Leverage dataset2Vec warm-start recommendations during model selection.
*   **File to edit:** `epic_3/model_selection/agents.py`
*   **Details:**
    *   In model selection steps, query dataset2Vec's `MetaKnowledgeStore` using FAISS to recommend model candidates based on similarity of embeddings.

---

## 3. API & FRONTEND INTEGRATIONS

### [TASK-3.1] Add Leaderboard and Explainability Endpoints
*   **Description:** Expose the final post-training metrics and plots to the client.
*   **File to create:** `backend/routers/evaluation.py`
*   **Details:**
    *   `GET /api/runs/{session_id}/leaderboard`: Return ranked model metrics.
    *   `GET /api/runs/{session_id}/verdict`: Return nominated champion and Judge rationale.
    *   `GET /api/runs/{session_id}/shap`: Return SHAP feature contributions.
    *   Mount static directory serving for generated PNG plots.
*   **File to edit:** `backend/main.py` (register evaluation router).

### [TASK-3.2] Connect React Leaderboard to Endpoints
*   **Description:** Replace frontend mockup data with live API results.
*   **File to edit:** `frontend/src/api/client.js` (add endpoint callers).
*   **File to edit:** `frontend/src/App.jsx` (pass `activeSessionId` to leaderboard).
*   **File to edit:** `frontend/src/screens/LeaderboardScreen.jsx`:
    *   Fetch leaderboard data and Judge verdicts dynamically on mount.
    *   Embed the dynamic `feature_importance_bar.png` asset plot.

---

## 4. REFACTORING & CODE CLEANUP

### [TASK-4.1] Consolidate `config.ini` Settings
*   **Description:** Merge local configurations into the single global config file.
*   **File to edit:** `config.ini` (append `[judge]`, `[overfitting]`, `[shap]` settings).
*   **Files to delete:**
    *   `epic_4/judge_agent/config/config.ini`
    *   `epic_4/overfitting_analysis_tool/config/config.ini`
    *   `epic_4/dataset2Vec/config/config.ini`
    *   `epic_4/SHAP/config/config.ini`

### [TASK-4.2] Eliminate Duplicate Model Registries and Data Loaders
*   **Description:** Reuse validator model names and data wrappers to reduce duplication.
*   **File to edit:** `epic_4/overfitting_analysis_tool/overfitting_analysis.py` (import model wrappers from `model_library`).
*   **Files to edit:** SHAP and dataset2Vec loaders (use `DataBundle` from `model_library`).

---

## 5. TESTING & VERIFICATION

### [TASK-5.1] Implement End-to-End Test Suite
*   **Description:** Write an automated test verifying the entire chained pipeline execution.
*   **File to create:** `backend/tests/test_e2e_pipeline.py`
*   **Details:**
    *   Simulate file upload, validation, metadata generation, preprocessing, model selection, training, explainability, and judging steps.
    *   Assert presence and schema validity of all intermediate and final JSON/CSV artifacts.
