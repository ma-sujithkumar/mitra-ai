# Plan to Commit and Push Judge Runner Async Update

This plan stages, commits, and pushes updates to the judge agent runner to correctly pass a structured `types.Content` to `run_async`.

## Commit Details
- **Files**:
  - `backend/agents/evaluation/judge/judge_agent.py`
- **Message**: `judge: pass types.Content instead of plain dict to run_async`

## Execution Plan
1. Add modified `backend/agents/evaluation/judge/judge_agent.py` and `plans/commit_and_push_judge_runner_20260621.md`.
2. Commit with the specified message.
3. Push to origin `dev`.
