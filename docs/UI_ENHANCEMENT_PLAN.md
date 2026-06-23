# Plan: Feature-Engineering tab, Agent Reasoning display, Visualization tab, richer Leaderboard + model downloads

## Context

MITRA is an agentic AutoML app (FastAPI backend + React/Vite frontend, state-based tab routing). Several phases produce rich artifacts on disk but the UI surfaces almost none of them, and a lot of screen space sits empty:

1. **Feature engineering (epic2)** runs as a deterministic 11-step pipeline (`FeatureEngineerOrchestrator`), but there is **no UI** for it and **no status endpoint**. Its status only exists as flat files (`execution_log.txt`, `feature_artifact.json`) written to `pipeline_output/<run_id>/` â outside the per-session tree the API serves, so the UI cannot read them.
2. The **Training page** (`TrainingPage.jsx`) only shows live training. **HPT, SHAP, and Judge outputs are never visualized** there despite lots of empty space and the endpoints/artifacts already existing.
3. There is **no Visualization page**. The backend already generates matplotlib PNGs for eda/training/overfitting/hpt/judge/shap and exposes them via `GET /api/runs/{sid}/plots`, and frontend helpers `fetchPlots`/`plotUrl` exist â **but no screen consumes them**.
4. The **Leaderboard** shows only 2 numeric columns; the user wants **all 5 metrics + an overfitting score**.
5. The user wants a **download button for the trained models** in the leaderboard.
6. **[VERY IMPORTANT NEW REQUIREMENT]** Every AI agent's reasoning must be **prominently and clearly displayed** in the UI. This means: the FeatureSelector LLM's `rationale` text on the Feature Engineering page, the Judge's full `llm_commentary` + `rule_outcomes` + per-model `reasons[]` on the Leaderboard/Training pages. Agent icons (`AgentAvatar`) must appear throughout wherever an agent is referenced.

**Outcome:** a new Feature Engineering tab (epic2 step/agent status + LLM reasoning + agent icons + D2V similar-dataset panel), HPT/SHAP/Judge visualizations + full judge reasoning added to the Training page (+ model config panel), a new Visualization tab (PNG gallery), a richer Leaderboard (5 metrics + overfitting + judge reasoning + agent icons + token usage + download buttons).

## Decisions (confirmed with user)

- **Plot rendering:** reuse the backend-generated PNGs via the existing `/plots` API; SHAP and the leaderboard stay interactive (HBars / table).
- **FE status liveness:** polled â persist a structured `feature_run.json` per session and poll it (FE finishes in seconds, synchronously, before training).
- **FE artifact location:** add an optional `output_dir` param to `FeatureEngineerOrchestrator`; `PipelinePrep` points it at `<session>/reports/feature_engineering` so artifacts land per-session (no copying). Default behavior unchanged when the param is omitted.

## Key reuse references (do not duplicate)

- Tab registration pattern: `frontend/src/data.js` `NAV_ITEMS` (l.267) + `frontend/src/App.jsx` `ROUTE_META` (l.12) + `screens` map (l.105).
- Status/step UI: `components/StatusPill.jsx`, `components/MetadataProgress.jsx` (closest step-list template), `components/Stepper.jsx`, `components/AgentAvatar.jsx` + `AGENTS` in `data.js` (already has `feature` hue:212/short:FS and `judge` hue:38/short:JD agents).
- API wrapper: `frontend/src/api/client.js` `requestJson` + existing `fetchShap`/`fetchVerdict`/`fetchLeaderboard`/**`fetchPlots`/`plotUrl`** (l.116-122, currently unused).
- Backend artifact reader: `backend/routers/evaluation.py` `EvaluationArtifactReader` + `_build_reader` (reuse for all new endpoints).
- Interactive widgets: `components/HBars.jsx` (SHAP bars), leaderboard table markup in `screens/LeaderboardScreen.jsx`.
- Metrics source of truth: `backend/agents/training/metrics.py` â classification = `accuracy, f1_macro, f1_weighted, precision_macro, recall_macro`; regression = `mse, rmse, mae, r2`.
- Overfitting artifact: `<session>/evaluation/overfitting/<model>/overfitting_analysis.json` â `{is_overfitted, primary_metric, gaps:{metric:gap}, ...}` (written by `OverfittingAnalyzer`).

### Agent reasoning artifacts (key data sources)

- **FeatureSelector reasoning** (`backend/agents/feature_engineering/tools/selector.py`): The LLM response JSON has a `"rationale"` field (1-2 sentences on feature selection and PCA decision). Written to `raw_responses.txt` in the FE output dir. The `selection_method` field in `feature_artifact.json` records outcome (`llm_select`, `llm_select+pca(N)`, `fallback:mrmr_all`).
- **Judge reasoning** (`backend/agents/evaluation/judge/schemas.py`): `judge_decision.json` â `decision_trace.rule_outcomes` (dict of rule names to scores/flags) + `decision_trace.llm_commentary` (full LLM text, `str | None`) + `ranked_models[].reasons[]` (list of per-model reason strings) + `ranked_models[].llm_flags[]`.

---

## Part A â Feature Engineering status tab (epic2)

### Backend
1. **`backend/agents/feature_engineering/orchestrator.py`** â add optional `output_dir: str | Path | None = None` to `__init__`. In `run()`, if provided use it as the run output directory (`mkdir(parents=True, exist_ok=True)`) instead of `Path("pipeline_output")/run_id`. Default path unchanged when omitted. (~lines 82-99, 169-171.)
2. **`backend/services/pipeline_prep.py`** `_run_feature_engineering` â pass `output_dir=self.reports_dir / "feature_engineering"`. After `orchestrator.run()`, call a new helper `_write_feature_run_status` that:
   - Parses `execution_log.txt` (per-step `ok`/`error` + elapsed + llm source).
   - Reads `feature_artifact.json` (selection_method, dropped/created/selected columns, warnings).
   - **Parses `raw_responses.txt` to extract the FeatureSelector's `rationale` field** (JSON `{..., "rationale": "..."}` blob; truncate at 2000 chars). Store as `llm_reasoning`.
   - Writes structured **`feature_run.json`** atomically (tempfile + `os.replace`, mirroring `TrainingService._persist_state`).
3. **New `backend/services/feature_status.py`** â `FeatureEngineeringStatusReader(session_dir)`:
   - Canonical step list (11 steps mirroring `orchestrator._run_pipeline`) with display label + `agent_type` (`llm` for `select_features`; `rule` otherwise).
   - Build/read step states `{name, label, agent_type, status: ok|error|pending, elapsed_sec, llm_source, detail}` from `feature_run.json`.
   - Summary block: `task, target_column, dropped_columns[], created_columns[], selected_columns[], selection_method, warnings[]`.
   - **Reasoning block**: `llm_reasoning` (the `rationale` text from FeatureSelector) + `selection_method` badge string.
   - Agents block: `{id: "feature", state: ok|error|pending}` + `{id: "judge", state: ...}` derived from `select_features` outcome / `llm_source`.
4. **`backend/routers/evaluation.py`** â add `GET /api/runs/{session_id}/feature-engineering` reusing `_build_reader`/session-dir resolution; returns `{session_id, status, steps[], agents[], summary, reasoning: {llm_reasoning, selection_method}}`.
5. **`config.ini`** â add `[feature_engineering_api]` with `OUTPUT_SUBDIR=reports/feature_engineering` and `RUN_STATUS_FILENAME=feature_run.json` (no hardcoded paths in code).

### Frontend
6. **`data.js`** â add `NAV_ITEMS` entry `{ key: 'features', label: 'Feature Engineering', icon: 'layers' }` (between `upload` and `pipeline`). Add a `FEATURE_STEPS` metadata array (label/agent-type/agent-id) for ordering/labels.
7. **`App.jsx`** â add `ROUTE_META.features` + `screens.features: <FeatureEngineeringPage activeSessionId={activeSessionId} go={go} />`.
8. **`api/client.js`** â add `fetchFeatureEngineering(sessionId)` â `/api/runs/{sid}/feature-engineering`.
9. **New `screens/FeatureEngineeringPage.jsx`** â layout: three panels using the available space:
   - **Step list panel** (left/main): 11 steps styled after `MetadataProgress`; each step shows `StatusPill` (ok/error/pending) + label + elapsed time. LLM steps (select_features) show an inline mini `AgentAvatar` for the `feature` agent + `llm_source` badge (e.g. "judge" / "fallback"). Poll `fetchFeatureEngineering` every 1.5s until `status === "done"`.
   - **[VERY IMPORTANT] Agent Reasoning panel** (right/aside): a prominent dedicated card titled "Agent Reasoning" with `AgentAvatar` for the `feature` agent (hue:212). Shows:
     - `selection_method` as a `StatusPill`-style badge (e.g. "llm_select+pca(3)").
     - The `llm_reasoning` rationale text in a styled `.reasoning-block` (scrollable, contrasting background, mono font for the LLM output text).
     - `warnings[]` as a compact error callout if present.
   - **Summary panel** (below reasoning): task type, target column, counts (#dropped/#created/#selected columns), selection method, collapsible warnings list.
   - **Agents row**: `AgentAvatar` cards for `feature` (Feature Selection) + `judge` (Judge) agents side by side, showing their state (idle/running/done/error), type badge (LLM), and role description. Reuse `AGENTS` from `data.js`.

---

## Part B â Training page: visualize HPT / SHAP / Judge + Judge reasoning

10. **`screens/TrainingPage.jsx`** â when the run completes (`state.complete` or backend status terminal), render a new **"Training analytics"** section filling the empty space, laid out as two columns:
    - **Left column:**
      - **SHAP** â interactive `HBars` via `fetchShap` (reuse `LeaderboardScreen` logic), with `AgentAvatar` (judge agent) in the section header.
      - **HPT / overfitting / training PNGs** â embed via `fetchPlots` + `plotUrl`, filtered to `hpt`, `overfitting`, `training` stages. Each image in a `.plot-card`.
    - **[VERY IMPORTANT] Right column â Judge Reasoning panel:**
      - `AgentAvatar` for the `judge` agent (hue:38/JD) in the panel header, shown prominently with state=done.
      - Full `decision_trace.llm_commentary` text in a `.reasoning-block` (scrollable, contrasting background). If null, show "Rule-based decision â no LLM commentary."
      - `decision_trace.rule_outcomes` rendered as a compact key-value table (rule name â outcome/score).
      - Per-model `reasons[]` list for the selected/winning model.
      - Data from `fetchVerdict(sessionId)`.
11. **Backend HPT plot fix** â the `plots/hpt/optimization_history.png` is currently empty because per-trial history is dropped before persisting. In `backend/orchestration/eval_runner.py` `HPTRunner.run` (canonical write ~l.189) include the per-trial `(trial_number, value, best_so_far)` history that `backend/agents/evaluation/hpt/agent.py` already computes (consumed by `compute_param_sensitivity` then dropped) so `PipelinePlotGenerator._plot_hpt_optimization_history` renders. Keep payload additive/back-compatible.

---

## Part C â Visualization tab (PNG gallery)

12. **`data.js`** â add `NAV_ITEMS` entry `{ key: 'visualize', label: 'Visualization', icon: 'chart' }` (after `leaderboard`).
13. **`App.jsx`** â add `ROUTE_META.visualize` + `screens.visualize: <VisualizationPage activeSessionId={activeSessionId} />`.
14. **New `screens/VisualizationPage.jsx`** â `fetchPlots(activeSessionId)`, group by `stage` (eda/training/overfitting/hpt/judge/shap), render a responsive gallery of `<img src={plotUrl(sid, plot.path)}>` cards with section headers and a click-to-enlarge lightbox (plain CSS overlay, no new dep). Empty state when no session/plots. Reuse `.card`/`.panel-section`/`.section-kicker` styling.
15. **`theme.css`** â add a `.plot-gallery` grid + `.plot-card` + lightbox overlay styles.

---

## Part D â Leaderboard: all 5 metrics + overfitting score + agent reasoning + agent icons

16. **`backend/routers/evaluation.py`** `EvaluationArtifactReader`:
    - In `_index_training_metrics`, keep the full validation metrics dict per model (all 5 / 4 keys) â already passed through as `metrics`.
    - Add reading of `evaluation/overfitting/<model>/overfitting_analysis.json` and attach `overfitting: {is_overfitted, primary_metric, gap}` (gap = `gaps[primary_metric]`) to each leaderboard row in `build_leaderboard`.
    - Pass through `decision_trace` (both `rule_outcomes` and `llm_commentary`) and per-model `llm_flags[]` in the verdict/leaderboard response so the frontend can render full reasoning without a separate call.
17. **`screens/LeaderboardScreen.jsx`** â three layout changes:
    - **Leaderboard table**: replace the fixed `acc`/`f1` columns with **all available metric keys** rendered dynamically (classification 5 / regression 4) plus an **Overfitting** column (gap value; colored red/green by `is_overfitted`). Each row also shows per-model `reasons[]` inline (small text under model name). Add `Icons.download` per-row download button.
    - **[VERY IMPORTANT] Judge Reasoning panel** (replace the current compact "Judge / Reasoning" card with a full panel):
      - `AgentAvatar` for `judge` agent (hue:38) in the panel header, shown prominently.
      - Full `llm_commentary` text in a `.reasoning-block` (scrollable, contrasting bg). Fall back to "Rule-based decision" if null.
      - `rule_outcomes` key-value table (rule name â outcome).
      - Per-model `reasons[]` list for the winner.
      - `llm_flags[]` if present.
    - **SHAP panel**: keep `HBars` but add `AgentAvatar` for feature agent (hue:212) in the header.
18. **`theme.css`** â widen/adjust the `.leaderboard-table` grid (or make it horizontally scrollable) to fit the extra columns. Add `.reasoning-block` style: scrollable pre-wrap text area with contrasting panel background and 12px mono font, max-height ~200px.

---

## Part E â Model download buttons

19. **`backend/routers/evaluation.py`** â add:
    - `GET /api/runs/{session_id}/models/{model_name}/download` â resolve the model's `model_path` from `training_summary.json`, path-traversal-guard it within the session dir (reuse `resolve_plot`-style check), return `FileResponse`.
    - `GET /api/runs/{session_id}/models/download-all` â stream a zip of all completed models' artifacts (`zipfile` + `StreamingResponse`).
20. **`api/client.js`** â add `modelDownloadUrl(sid, modelName)` and `modelsDownloadAllUrl(sid)` URL helpers (plain anchor links, like `activityLogDownloadUrl`).
21. **`screens/LeaderboardScreen.jsx`** â add a per-row download icon button (`Icons.download`) linking to the per-model URL, and a **"Download all models"** button in the leaderboard section header.

---

## Part G â Dataset2Vec outputs

Dataset2Vec (D2V) queries the meta-knowledge store for similar past datasets and produces warm-start model recommendations. The `D2VBridge.query()` exists in `backend/orchestration/d2v_bridge.py` and the store has real data (`backend/agents/dataset2vec/store/`), but the **UI training pipeline (`training_service.py`) never calls it** â it only runs in the CLI. We fix this.

### Backend
24. **`backend/services/pipeline_prep.py`** `run()` â after feature engineering (Step 1), call `D2VBridge(db_dir=...).query(csv_path=engineered_csv, target_column=target_column, task_type=resolved_task)` (non-fatal; `try/except` with logging). If successful, write result as `reports/dataset_prior.json` using `prior.model_dump_json(indent=2)`. Import from `backend.orchestration.d2v_bridge`. The `db_dir` path should come from `config.ini` (add `D2V_DB_DIR=DB` to `[paths]`).
25. **`backend/routers/evaluation.py`** â add `GET /api/runs/{session_id}/d2v-prior` that reads `reports/dataset_prior.json` and returns it. Returns `{status: "pending"}` if not present.
26. **`api/client.js`** â add `fetchD2VPrior(sessionId)` â `/api/runs/{sid}/d2v-prior`.

### Frontend â Feature Engineering page (addition to item 9)
27. Add a **"Similar Datasets / Model Recommendations"** panel at the bottom of `FeatureEngineeringPage.jsx`, loaded via `fetchD2VPrior`:
    - **Neighbors section**: list of similar past datasets â each row shows `dataset_id`, similarity bar (0.0-1.0 using `.bar i`), `best_model` badge, top metric value.
    - **Recommended models section**: `ranked_models[]` as a table â model name, D2V score, expected metric, recommended hyperparameters (truncated key: value list).
    - `cold_start=true`: show an info callout "Cold start â no similar datasets found yet."
    - `caveats[]`: compact callout if non-empty.
    - Omit the panel entirely while status is `"pending"`.

---

## Part H â Additional outputs (validation stats, model config, token usage)

28. **Validation report panel** â add `GET /api/runs/{session_id}/validation` to `evaluation.py` reading `reports/validation_report.json`. Frontend: show on Dashboard or new Feature Engineering page â column count, null density heatmap summary, flagged columns, variance failures. Reuse the existing `.stage-row` list pattern.

29. **Model config panel** â on the Training page (in the existing session bar or below it), read `reports/model_config.json` (candidate model families, count, task type). Show as a `.stage-row` chip list: each selected model family as a `.pill`. Add `fetchModelConfig(sessionId)` â `GET /api/runs/{sid}/model-config`. Backend endpoint in `evaluation.py`.

30. **Token usage panel** â `fetchTokens(sessionId)` already exists (`client.js:112`) but is never shown. Display it on the Leaderboard page: a compact `"Token usage"` card showing total tokens and per-agent breakdown (agent name â input/output tokens). Already has endpoint `GET /api/runs/{sid}/tokens`.

---

## Part F â Agent reasoning CSS + icons (cross-cutting)

22. **`theme.css`** â add:
    - `.reasoning-block`: `background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--radius); padding: 14px 16px; font-family: var(--mono); font-size: 12px; line-height: 1.65; white-space: pre-wrap; overflow-y: auto; max-height: 220px; color: var(--ink-muted);`
    - `.agent-reasoning-header`: flex row with gap, aligns `AgentAvatar` + title + badge horizontally.
    - `.rule-outcomes-table`: compact two-column grid (rule name | outcome) styled like `.leaderboard-head` rows.
    - `.reasoning-panel`: `.card.panel-section` variant with a left color-accent border using the agent's hue CSS var `--agent-accent`.

23. **Agent icons placement summary** (use `AgentAvatar` from `components/AgentAvatar.jsx`, `AGENTS` from `data.js`):
    - Feature Engineering page: `feature` avatar (hue:212, short:FS) in the Agent Reasoning panel header AND inline (size=22) next to the `select_features` step row. `judge` avatar (hue:38, short:JD) in the Agents row.
    - Training analytics section: `judge` avatar (hue:38) in the Judge Reasoning panel header. `hpt` avatar (hue:330, short:HP) in the HPT plots section header.
    - Leaderboard: `judge` avatar (hue:38) in the Judge Reasoning panel header (replaces the existing small avatar). `feature` avatar (hue:212) in the SHAP panel header.

---

## Verification

Backend (use the `PYTHON` from `config.ini` / `.venv`):
- Run an end-to-end pipeline on a sample dataset to populate a session under `.mitra/<session_id>/`.
- `curl` each new/changed endpoint and confirm shape:
  - `/api/runs/{sid}/feature-engineering` â 11 steps with ok/error/elapsed + agents + `reasoning.llm_reasoning` populated from FeatureSelector rationale.
  - `/api/runs/{sid}/leaderboard` â each model has full `metrics` (5/4 keys) + `overfitting` + `decision_trace` with `llm_commentary`.
  - `/api/runs/{sid}/plots` lists `hpt/` PNG and it is non-empty after the HPT trial-history fix.
  - `/api/runs/{sid}/models/{model_name}/download` and `/models/download-all` return files.
- Confirm `<session>/reports/feature_engineering/{execution_log.txt,feature_artifact.json,feature_run.json}` exist.
- Run affected backend tests via spawned test agent: feature_engineering, evaluation router, hpt, overfitting suites.

Frontend (`mitra frontend` / `npm run dev`):
- Feature Engineering tab: step list + **Agent Reasoning panel shows FeatureSelector rationale text** + `AgentAvatar` icons visible + D2V neighbors/recommendations panel.
- Training page (post-completion): Judge Reasoning panel shows full `llm_commentary` + `rule_outcomes` table + `AgentAvatar` for judge + model config chip list.
- Leaderboard: 5 metrics + overfitting column + **Judge Reasoning panel with full commentary** + `AgentAvatar` icons + token usage panel + per-row download buttons.
- Visualization tab: PNG gallery grouped by stage with click-to-enlarge.

## Notes / constraints (per CLAUDE.md)

- Reuse existing components/endpoints (no duplication); imports at top; ASCII only; descriptive names; full typing on new Python; `=>` not arrows in logs; `mkdir -p` for any new output dir; no new paths hardcoded (use `config.ini`); no commits unless asked.
- Adds **no new frontend dependency** (PNG reuse + existing custom widgets + CSS lightbox).
- Agent reasoning display is **VERY IMPORTANT** â must be prominent, not collapsed or hidden behind a toggle. Full text, not truncated on screen. 
