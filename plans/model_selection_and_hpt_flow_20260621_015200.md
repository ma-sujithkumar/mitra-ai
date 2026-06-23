# Implementation Plan: Model Selection, Evaluation Sub-Stages & HPT Flow

This plan outlines the changes required to implement visual progress tracking for Model Selection, hide candidate models until selection finishes, add UI status indicators for all evaluation stages, and run Hyperparameter Tuning (HPT) on-demand from the Leaderboard page.

## 1. Backend Modifications

### A. Update `TrainingEvent` Stage Literal
In [contracts.py](file:///home/sujithma/mitra/backend/orchestration/events/contracts.py):
* Change `stage: Literal["training", "evaluation"] = "training"` to `stage: str = "training"` to support all stages (`d2v`, `model_selection`, `shap`, `overfitting`, `hpt`, `training`, `evaluation`) without raising Pydantic validation errors.

### B. Add SSE Event Emission in Pre-training
In [pipeline_prep.py](file:///home/sujithma/mitra/backend/services/pipeline_prep.py):
* Import `TrainingEvent` and `TrainingEventBus` at the top of the file.
* Update `PipelinePrep.__init__` to accept an optional `event_bus: Optional[TrainingEventBus] = None`.
* Emit SSE events during:
  * **Dataset2Vec Matcher** (`stage="d2v"`): Emit when lookup starts (status="running", pct=20), when it succeeds (status="completed", pct=100), or fails (status="failed", pct=100).
  * **Model Selection** (`stage="model_selection"`): Emit when selection starts (status="running", pct=10) and finishes (status="completed", pct=100).

### C. Bridge Event Bus to Pre-training Router
In [feature_engineering.py](file:///home/sujithma/mitra/backend/routers/feature_engineering.py):
* Import `get_training_event_bus` and `TrainingEventBus` at the top of the file.
* Inject `event_bus: TrainingEventBus = Depends(get_training_event_bus)` in `start_feature_engineering`.
* Pass `event_bus` to `_run_pipeline_prep` thread worker and initialize `PipelinePrep` with it.

### D. Emit Evaluation Sub-Stage SSE Events
In [eval_runner.py](file:///home/sujithma/mitra/backend/orchestration/eval_runner.py):
* Import `TrainingEvent` and `TrainingEventBus` at the top of the file.
* Update `EvalRunner.__init__` to accept `event_bus: Optional[TrainingEventBus] = None`.
* Emit SSE events during:
  * **SHAP explainability** (`stage="shap"`): Emit start (status="running", pct=10) and completion (status="completed", pct=100).
  * **Overfitting Analysis** (`stage="overfitting"`): Emit start (status="running", pct=10) and completion (status="completed", pct=100).

### E. Pass Event Bus to EvalRunner
In [training_service.py](file:///home/sujithma/mitra/backend/services/training_service.py):
* Pass `self.event_bus` when instantiating `EvalRunner`.

---

## 2. Frontend Modifications

### A. Sub-stage Event Tracking in Live Training Page
In [TrainingPage.jsx](file:///home/sujithma/mitra/frontend/src/screens/TrainingPage.jsx):
* Add a `stageStatuses` React state dictionary mapping each stage (`d2v`, `model_selection`, `training`, `shap`, `overfitting`, `evaluation`, `hpt`) to its current status (`pending`, `running`, `complete`, `failed`), progress percentage (`progress`), and status message.
* Update `onEvent` in `streamTrainingEvents` to intercept incoming stage-specific events and dynamically update `stageStatuses`.
* Reset `stageStatuses` in the `connect` callback.

### B. Display Pipeline Stage Cards
In [TrainingPage.jsx](file:///home/sujithma/mitra/frontend/src/screens/TrainingPage.jsx):
* Render a new panel called **Pipeline execution stages** under `TrainingProgress`.
* Display visual indicator cards for:
  1. **Dataset2Vec Matcher**
  2. **Model Selection Agent**
  3. **Parallel Model Training**
  4. **SHAP explainers**
  5. **Overfitting Analysis**
  6. **Judge Evaluation Loop**
* Apply animations (such as pulsing icons, micro-progress bars) for the active/running stage and proper green checkmarks / red warnings for success/failure.

### C. Gate Candidate Models List
In [TrainingPage.jsx](file:///home/sujithma/mitra/frontend/src/screens/TrainingPage.jsx):
* Derived state `isModelSelectionComplete`: returns true if `stageStatuses.model_selection.status === 'complete'` or if `models.length > 0`.
* Hide the "Selected Models / Training queue" card when `isModelSelectionComplete` is false.
* Render a beautiful loader placeholder card ("Selecting candidate models...") showing the live progress bar and status message of the Model Selection Agent.

### D. On-demand HPT Trigger on Leaderboard
In [LeaderboardScreen.jsx](file:///home/sujithma/mitra/frontend/src/screens/LeaderboardScreen.jsx):
* Add `hptStatus` (`idle`, `running`, `complete`, `failed`) and `hptData` states.
* Fetch HPT status initially when `activeSessionId` is loaded.
* Implement a poll loop that runs if `hptStatus === 'running'` to fetch updated HPT results.
* Add a "Tune Hyperparameters (HPT)" button in the action banner when the leaderboard is complete.
* Render a dedicated `HptTuningSection` under the leaderboard table to show active tuning status (n_trials progress) and final Optuna tuning results.

---

## 3. Styles Modification

In [theme.css](file:///home/sujithma/mitra/frontend/src/theme.css):
* Add CSS styles for the pipeline stages layout and cards (glassmorphism look, keyframes for pulsing indicators, micro-progress bar lines, grid alignment).
