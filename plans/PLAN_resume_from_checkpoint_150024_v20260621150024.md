# Resume-from-checkpoint for existing sessions

Timestamp: 20260621_150024
Branch: dev

## Goal
When a user selects an existing on-disk session (recent-runs list), phases whose
artifacts already exist (metadata generation, feature engineering, training) must
NOT re-run. The app continues from the last checkpoint and jumps to the first
incomplete phase. Each phase exposes an explicit "Re-run" control that forces the
agents to run again for that phase only.

## Decisions (confirmed with user)
1. "Uploaded artifact associated with a session ID" == an existing session already
   under `.mitra/<session_id>` selected from the recent-runs list. No new
   bundle-upload feature.
2. Re-run is triggered per phase via an explicit "Re-run this phase" button.
   Default behavior = skip when the phase's artifacts already exist.
3. On resume, navigate to the first incomplete phase.

## Current behavior (verified)
- `POST /api/metadata` (backend/routers/metadata.py:52) always runs the metadata
  agent; no skip when `reports/metadata.json` exists.
- `POST /api/feature-engineering` (backend/routers/feature_engineering.py:68)
  always runs PipelinePrep.
- Training: `POST /api/training/start` raises 409 TRAINING_ALREADY_EXISTS if a run
  exists; `POST /api/training/reset/{id}` (reset_run) already clears
  training_summary.json + judge_decision.json so training can be re-run. REUSE.
- `runs.py` already computes validation/metadata/leaderboard status from artifact
  presence (`_build_run_record`). REUSE / generalize.
- Frontend `UploadScreen.handleReview` re-runs validation + metadata even for a
  selected recent session. FE page "Continue to training" always starts FE.

## Phase -> "complete" artifact map (single source of truth)
Ordered phases and the artifact whose presence means "complete":
| phase                | artifact (under session_dir)                                   |
|----------------------|----------------------------------------------------------------|
| validation           | reports/validation_report.json with passed == true             |
| metadata             | reports/metadata.json                                           |
| feature_engineering  | reports/feature_engineering/feature_artifact.json AND reports/model_config.json |
| training             | reports/training_summary.json                                  |
| evaluation           | reports/judge_decision.json                                    |

Filenames already known to config_loader (training_api.session_output_dir,
run_status_filename, feature_engineering_api.run_status_filename). Add the phase
ordering + remaining filenames to a single `[pipeline_phases]` section in
config.ini (no new config file) to avoid hardcoded lists (CLAUDE.md rule 5).

## Backend changes
B1. New `backend/services/session_progress.py`:
    - `class SessionProgress` reads config + session_dir, exposes
      `phase_status() -> dict[str, str]` ("complete"/"pending"/"passed"/"failed")
      and `first_incomplete_phase() -> str | None`.
    - Drives off the phase->artifact map (dict, not if-else ladder).
B2. `GET /api/runs/{session_id}/progress` in runs.py returns
    `{session_id, phases: {...}, next_phase}`. Refactor `_build_run_record` to use
    SessionProgress so list_runs/run_stats gain feature_engineering/training
    status without duplicating detection.
B3. metadata starter skip-aware: add `force: bool = False` to MetadataRequest.
    If metadata.json exists and not force -> log "skipped (cached)", mark job
    done, return {status: "skipped", artifact: "metadata.json"}.
B4. feature-engineering starter skip-aware: add `force` to its request. If
    feature_artifact.json + model_config.json exist and not force -> return
    {status: "skipped"}.
B5. training: no backend change. Frontend re-run = POST reset then POST start.

## Frontend changes
F1. api/client.js: `getRunProgress(sessionId)`; add `force` to startMetadata +
    startFeatureEngineering; add `resetTraining(sessionId)`.
F2. UploadScreen: on selecting a recent session, fetch progress; render per-phase
    status pills. Primary action becomes "Continue": skip completed
    validation/metadata, then resume-navigate to first incomplete phase.
F3. App.jsx: `resumeSession(sessionId, progress)` -> route to first incomplete
    phase using existing setters (enterFeatureEngineering / startRun / go).
F4. Per-phase "Re-run" buttons:
    - UploadScreen: "Re-run metadata" -> startMetadata(force=true).
    - FeatureEngineeringPage: "Re-run feature engineering" -> startFeatureEngineering(force=true).
    - TrainingPage: "Re-run training" -> resetTraining then startTraining.

## Tests (backend/tests)
- SessionProgress: artifact-combination -> expected phase_status + next_phase.
- /runs/{id}/progress endpoint shape.
- metadata skip vs force; feature-engineering skip vs force.

## Coding-standard notes (CLAUDE.md)
- One config.ini, new `[pipeline_phases]` section; no hardcoded lists in code.
- Phase dispatch via dict map, not if-else ladder.
- Typed function signatures; descriptive names; ASCII only; reuse runs.py/reset_run.
```
```
