# Plan: Judge Agent Response & Leaderboard Flow

## Objective
Implement proper UI flow where:
1. The Leaderboard is only accessible *after* the Judge Agent finishes evaluating models.
2. The Judge's responses and reasoning are displayed on the TrainingPage.
3. The user can trigger a re-run of the training pipeline directly from the TrainingPage using the Judge's feedback.

## Detailed Steps

### 1. Backend Updates
- In [training_service.py](file:///home/sujithma/mitra/backend/services/training_service.py), enhance `reset_run(session_id)` to also delete existing report artifacts (`reports/judge_decision.json`, `reports/training_summary.json`) so subsequent runs start with a clean state and do not report stale judge results.

### 2. Frontend API Client Updates
- We have already added `resetTraining(sessionId)` to [training.js](file:///home/sujithma/mitra/frontend/src/api/training.js) and verified it with tests.

### 3. Frontend TrainingPage UI & Logic Updates
- In [TrainingPage.jsx](file:///home/sujithma/mitra/frontend/src/screens/TrainingPage.jsx):
  - Add state variables:
    - `verdictData` (for the polled judge decision).
    - `isRestarting` (loading state for restarting training run).
  - Implement a `useEffect` polling loop that fetches `/api/runs/{session_id}/verdict` every 2 seconds once training finishes, or during execution, and stops when `verdictData?.status === 'complete'`.
  - Update `TrainingSummary` component invocation to pass `canContinue={verdictData?.status === 'complete'}` so the leaderboard is gated on the Judge's completion.
  - Render the Judge Reasoning panel on the main training page. If the judge hasn't finished, display a pending status/spinner indicating that the Judge Agent is evaluating the models.
  - Implement `handleRestartTraining()` which:
    1. Fetches feature engineering state or uses the active metadata.
    2. Resets the training run using `resetTraining(sessionId)`.
    3. Starts the run using `startTraining` with original settings.
    4. Resets frontend reducer state and reconnects the SSE event stream.
  - Add a "Re-run Training with Judge Feedback" button in the Judge section once the verdict is ready.
