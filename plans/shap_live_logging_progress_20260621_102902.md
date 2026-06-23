# Plan: SHAP Live Logging and Progress Events

## 1. Objectives
- **Live SHAP Progress in UI**: Emit a `TrainingEvent` on the `event_bus` as each model completes or fails its SHAP computation in `EvalRunner`.
- **Dynamic Percentage Updates**: Incrementally increase the SHAP stage progress percent (from 10% to 90%) as models finish.
- **Verbose Error Logging**: Surface SHAP warnings and errors (such as fallback messages or skipped models) to the UI event stream.

## 2. Changes Proposed
- Modify `run` method in `EvalRunner` (in `backend/orchestration/eval_runner.py` lines 310-318) to:
  - Track `completed_count` and compute `progress_pct = 10 + int(80 * (completed_count / total_models))`.
  - Emit a `TrainingEvent` inside the `as_completed` loop on success and failure.

## 3. Verification Plan
- Run tests:
  ```bash
  ~/venv/bin/pytest backend/agents/evaluation/shap/tests/
  ```
- Trigger training in the UI and watch the Live SSE stream logs.
