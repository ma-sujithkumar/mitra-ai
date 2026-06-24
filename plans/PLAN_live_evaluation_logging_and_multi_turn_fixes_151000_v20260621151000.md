# Plan: Live Evaluation Logging and Judge Multi-turn UI & Agent Interaction Fixes

This plan outlines the changes required to solve the following issues:
1. **Evaluation Event Stream Empty**: Merge polled status messages (`shap`, `overfitting`, `judge`) into the UI's evaluation event stream box in real time (every 2 seconds).
2. **Judge Agent Reasoning Invisible / Multi-turn Bypass**: Display judge agent reasoning and logs live in the right-hand aside panel right below the "Evaluation Status" card, and correctly disable the "Continue to leaderboard" button while the judge loop is running in a multi-turn scenario.
3. **Agent to Agent Interaction in Multi-turn**: Refactor multi-turn training to re-run model selection, retaining models already selected/approved by the judge, ranking rejected models at the bottom, and only executing training for the newly added candidates.

## Proposed Changes

### 1. Backend: Correct Verdict Status for Multi-turn Runs
In [`backend/routers/evaluation.py`](file:///home/sujithma/mitra/backend/routers/evaluation.py#L150):
- Check the current `judge_status`. If not terminal (`"all_completed"`, `"completed"`, `"failed"`), return `{"status": "pending"}` to keep the frontend in the evaluating state.

### 2. Backend: Re-select Models and Skip Already-Trained Candidate Runs
In [`backend/orchestration/judge_loop.py`](file:///home/sujithma/mitra/backend/orchestration/judge_loop.py):
- Refactor the retraining flow to separate previously approved models from rejected ones.
- When calling model selection, we will pass approved models (to retain at the top), and rejected models (to put at the bottom).

In [`backend/agents/training_orchestrator/orchestrator.py`](file:///home/sujithma/mitra/backend/agents/training_orchestrator/orchestrator.py):
- Implement `_get_existing_completed_result` to check if a model was successfully trained in the previous turn by reading `training_summary.json` and checking if its model artifact file exists on disk.
- If it exists, skip submitting the model candidate to the training worker/executor (Ray or local) and directly construct and return a `"completed"` `TrainingResult`.
- Ensure directory creation handles `exist_ok=True`.

### 3. Frontend: Live Progress, Deduplication, and Aside Panel
In [`frontend/src/screens/TrainingPage.jsx`](file:///home/sujithma/mitra/frontend/src/screens/TrainingPage.jsx):
- Define `polledEvalLogs` to accumulate unique messages polled from SHAP, Overfitting, and Judge status files.
- Deduplicate and merge polled logs with SSE logs in `EvaluationLogs`.
- Render the concise Judge Agent card in the `aside` panel below "Evaluation Status" to show logs, active tool calls, and final LLM commentary in real-time.

## Verification Plan

### Manual Verification
1. Run backend server using python binary.
2. Trigger training from the UI.
3. Observe:
   - Live event stream logs and aside reasoning panels update every 2 seconds.
   - Retraining ONLY executes for newly added model candidates. Existing models immediately transition to "completed" in the UI.

### Automated Tests
- Run all backend router and orchestrator tests:
  ```bash
  ~/venv/bin/pytest backend/tests/
  ```
