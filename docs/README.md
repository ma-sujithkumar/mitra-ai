# Epic-3 Demo Guide and Frontend Acceptance Report

## Scope

This document covers Onkar's final Page-2 frontend hardening for Epic-3. It validates the browser flow from Page-1 training start through Page-2 live progress, backend status polling, cancellation, reconnect and terminal summaries.

## Preconditions

1. Pull the latest `dev` branch.
2. Apply the Epic-3 backend E2E hardening patch before this frontend patch.
3. Install frontend dependencies with `npm ci` from `frontend/`.
4. Start the backend and frontend in separate terminals.

## Demo steps

1. Open the MITRA UI and go to **New Run**.
2. Select an uploaded dataset or upload a new CSV.
3. Configure problem type and target column.
4. Run **Validate & Review**.
5. After metadata and model selection complete, click **Start Ray training**.
6. Confirm the app navigates to **Live Training** with the active session ID.
7. Verify the Page-2 model cards update independently as events arrive:
   - queued
   - submitted
   - running
   - completed
   - failed
   - timed_out
   - cancelled
8. Disconnect and reconnect the event stream. The UI should preserve current backend status through status polling.
9. Refresh the browser on Page-2. The model cards should be restored from `GET /api/training/status/{session_id}`.
10. Run a cancellation demo and confirm active model cards move to cancelled.
11. After terminal completion, verify the summary and leaderboard button state.

## Acceptance matrix

| Scenario | Expected result |
|---|---|
| Page-1 start training | Calls `POST /api/training/start`, stores session ID and opens Page-2 |
| SSE running events | Updates model cards and terminal logs without page refresh |
| Completed model | Shows validation score, duration and model artifact path |
| Failed model | Shows error message and terminal failed state |
| Timed-out model | Shows timeout error and 100 percent terminal progress |
| Cancel button | Calls `POST /api/training/cancel/{session_id}` and keeps polling backend status |
| Browser refresh | Restores cards and summary from backend status payload |
| SSE disconnect | Shows reconnecting state without crashing the page |
| Backend terminal status | Stops polling and enables completion summary |
| Production build | `npm run build` succeeds |

## Validation commands

```bash
cd frontend
npm test
npm run build
```

Backend regression checks remain:

```bash
python -m pytest -q \
  backend/tests/test_training_service.py \
  backend/tests/test_training_api.py \
  tests/e2e \
  epic_3/events/tests \
  epic_3/training_orchestrator/tests \
  epic_3/ray_wrapper/tests \
  epic_3/training/tests \
  epic_3/model_selection/tests
```

## Notes

- Page-2 now hydrates model cards from the persisted backend status response, so a user does not lose the visible training state after refresh or SSE replay gaps.
- SSE transport errors are intentionally non-fatal because native `EventSource` reconnects automatically.
- The frontend does not start Ray directly. It calls the backend start API and then listens to the existing SSE endpoint.
