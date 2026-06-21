# Design Plan: UI, Backend Integration & Epic 4 Judge Agent Fixes

This plan outlines the changes to resolve all the requirements in `SPEC_for_enhancements_v2.md` and address the user's feedback.

---

## 1. Task 1: Visualization Page Trigger

### Backend
- **Add Route:** Implement `POST /api/runs/{session_id}/plots/generate` in [backend/routers/evaluation.py](file:///home/sujithma/mitra/backend/routers/evaluation.py).
- **Behavior:**
  - Retrieve the session path from `session_manager`.
  - Import `PipelinePlotGenerator` from `backend.orchestration.plotting`.
  - Execute `PipelinePlotGenerator(session_dir=session_dir).generate_all()`.
  - Return `{"status": "success", "message": "Visualizations generated successfully"}` on success.
  - Handle exceptions and return appropriate HTTP errors.

### Frontend
- **Add API Client Function:** Export `generatePlots(sessionId)` in [frontend/src/api/client.js](file:///home/sujithma/mitra/frontend/src/api/client.js).
- **Update UI component:** Modify [frontend/src/screens/VisualizationPage.jsx](file:///home/sujithma/mitra/frontend/src/screens/VisualizationPage.jsx).
  - Add a **"Generate Visualizations"** (or **"Refresh Visualizations"**) button in the top header section.
  - Implement local state variables: `isGenerating` (boolean) and `generationMessage` (string for success/error alerts).
  - While generating: Disable the button and display `"Generating visualizations..."` with a spinner.
  - On success: Display a success alert message: `"Visualizations generated successfully"`. Trigger a re-fetch of the plots list.
  - On error: Display the backend error message in an error alert/callout.
  - Restructure page rendering so the header panel (with the button and status alerts) is visible even when no plots are loaded yet (`plots.length === 0`).

---

## 2. Task 2: Training Page Missing from UI

### Frontend
- **Fix state initialization:** Update [frontend/src/screens/TrainingPage.jsx](file:///home/sujithma/mitra/frontend/src/screens/TrainingPage.jsx) around line 339.
  - Add `judge: { status: 'pending', progress: 0, message: '' }` to the initial state of the `stageStatuses` hook.
  - This prevents `stageStatuses.judge` from being undefined, resolving the `TypeError: Cannot read properties of undefined (reading 'status')` crash that blanked out the Training tab.

---

## 3. Task 3: Hyperparameter Tuning (HPT) Status & Streaming

### Frontend
- **Update HPT logs and info state:** In [frontend/src/screens/LeaderboardScreen.jsx](file:///home/sujithma/mitra/frontend/src/screens/LeaderboardScreen.jsx):
  - Add state hooks for HPT details: `hptLogs` (array), `hptTrialNum` (integer), `hptTotalTrials` (integer), and `hptBestScore` (float/null).
  - Reset these details to empty/default when `handleRunHpt` is clicked.
  - Listen to `stage === 'hpt'` events in the `streamEvaluationEvents` listener.
  - On HPT event:
    - Update `hptProgress` with `event.pct`.
    - Update `hptMessage` with `event.msg`.
    - Append unique events to `hptLogs`.
    - Update trial statistics (`hptTrialNum`, `hptTotalTrials`, `hptBestScore`) from `event.details` if present.
- **Implement HPT UI widgets:**
  - Render live trial statistics (e.g. `Trial 2/5 | Best Score: 0.924`) above the HPT progress bar when tuning is active.
  - Render an **Optuna Trial Logs** scrollable terminal under the progress bar using standard terminal styling classes (`terminal-body`, `terminal-line`).
  - Auto-scroll the terminal on new logs using a ref hook.

---

## 4. Task 4: Evaluation Pipeline SSE Stream

### Frontend
- **Implement Evaluation event stream panel:** In [frontend/src/screens/TrainingPage.jsx](file:///home/sujithma/mitra/frontend/src/screens/TrainingPage.jsx):
  - Add a sub-component `EvaluationLogs` that filters `state.logs` for `entry.stage === 'evaluation' || entry.stage === 'judge'`.
  - Maintain a live status state: `running` (if last event status is running), `complete` (if last event matches complete), `failed` (if last event level is error/failed), or `pending`.
  - Format log timestamps using a `formatTime` helper.
  - Highlight warnings and error logs with appropriate class colors.
  - Enable auto-scrolling to the latest event using a container ref.
  - Render the `<EvaluationLogs logs={state.logs} />` panel in the layout stack of the training page right next to (below) `<TrainingLogs />`.

---

## 5. Task 5: Cleanup customAnthropic / Verify LLM configuration

- **Completed checks:**
  - Confirmed `custom_anthropic_client.py` and `claude_adk_llm.py` have been deleted in git history.
  - Cleaned up leftover python cache files in `backend/agents/evaluation/judge/__pycache__/` to prevent any Python import issues.
  - Confirmed `JudgeAgent` uses `build_llm_model` via `llm/adk_client.py`, which instantiates standard Google ADK models (`LiteLlm`) using API keys and standard settings, matching the metadata generation agent.

---

## Validation Strategy
- Start frontend and backend servers.
- Trigger model training and monitor the Training page. Verify the Evaluation Event Stream Panel populates live evaluation states (SHAP, Overfitting, etc.).
- Navigate to the Leaderboard page. Trigger HPT tuning. Verify the live status bar, trial stats, and scrollable logs panel update in real-time.
- Navigate to the Visualizations page. Click the "Generate Visualizations" button. Verify the loading state, successful completion message, and that the page updates with the new top-10 model plots.
- Run automated tests (`pytest`) to ensure no regressions occur in the rule engine or LLM judge routing.
