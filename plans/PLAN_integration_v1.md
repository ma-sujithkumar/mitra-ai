# MITRA End-to-End Integration & Restructure Plan

## Context

Four epics were built independently (Epic 1 schema/ingestion, Epic 2 feature
engineering, Epic 3 model selection + training + dataset2Vec, Epic 4
SHAP/overfitting/HPT/judge) plus a FastAPI `backend/` and a React/Vite
`frontend/`. They duplicate code, use inconsistent directory styles, and are only
partially wired (backend integrates Epic 3 training only; Epic 2 and Epic 4 are
not called by the backend at all). The goal is a single cleaned-up, scalable,
self-hosted application with an end-to-end ML training pipeline driven by Google
ADK agents, runnable both via UI and `--cli`.

This plan restructures the repo (no `epic_*` folders), de-duplicates, enforces
the `google-adk`-only LLM constraint, builds the missing orchestrator (parallel
eval + judge feedback loop), wires dataset2Vec warm-start, and integrates the UI
— executed in backend-first phases.

**End-state UX:** the user opens the app, uploads a dataset (+ optional
description) on page 1, and watches the AI agents run the whole AutoML pipeline
live — validation, metadata, feature engineering, model selection (dataset2Vec
warm-start), parallel training, SHAP + overfitting + HPT, and the judge loop —
each agent streaming its status, then a leaderboard with the winning model,
comprehensive on-demand visualizations, and explainability. Nothing else is
required from the user after upload; `--cli` runs the same pipeline headless.

## Locked decisions (from Q&A)

- **LLM client:** repo ships an ADK `LiteLlm` wrapper + a generator; the Epic-1
  setup step **generates `client.py` into the session workspace** from the user's
  provider/model feedback and smoke-tests it there. Codebase stays read-only at
  runtime; all generated/mutable files live under `.mitra/<user_id>/<session_id>/`.
- **`google-adk` only:** migrate Epic 2 (currently OpenAI SDK) and the Judge
  (currently `claude` CLI) onto the shared ADK `LiteLlm` client now.
- **Config:** `config.ini` = env/paths/python binary; `config/config.yaml` = all
  module/pipeline params. Every `config.yaml` param surfaced in UI page-1
  `advanced_settings`; on app invoke, `config.yaml` is copied to the run dir and
  used from there.
- **metadata:** `metadata.json` canonical everywhere (embedded as JSON in prompts).
- **User model:** multi-user now; stub identity until the auth DB lands.
- **Runtime:** cross-platform; portable interpreter resolution; no absolute paths.
- **Encoder:** dataset2Vec inference + DB append only (no online training).
- **Frontend:** keep existing React/Vite (ignore DESIGN_PLAN's Streamlit).
- **Orchestrator:** hybrid — ADK agents + async/process executor (Ray already
  present) for true-parallel SHAP/Optuna/overfitting.
- **Execution:** phased, backend-first.

## Target structure (no epic_* folders)

```
mitra/
  bin/                 mitra (node launcher), setup.sh
  config/config.yaml   all module params (UI-surfaced)
  config.ini           env/paths/python only
  llm/                 adk_client.py (LiteLlm wrapper) + client_generator.py
  DB/                  encoder.pt, *.parquet, index.faiss, optuna.db, mapping
  model_library/       shared MLKit (kept)
  backend/
    main.py, session.py, config_loader.py, services/, routers/, schemas/
    orchestration/     hybrid DAG runner + unified event bus + token counter
    prompts/           ALL prompts (none in .py)
    agents/
      metadata_gen/        (from backend/agents)
      feature_engineering/ (from Epic2/pipeline)
      model_selection/     (from epic_3/model_selection)
      dataset2vec/         (from epic_3/dataset2Vec)
      training/            (epic_3 training + training_orchestrator + ray_wrapper)
      evaluation/{shap,overfitting,hpt,judge}/   (from epic_4)
  frontend/            React/Vite (kept)
  docs/<module>/
  .mitra/<user_id>/<session_id>/   runtime workspace (all generated artifacts)
```

Each migrated module keeps its existing internal package, re-homed under the new
path with imports updated. No logic rewrites during the move (git mv + import
fix), then refactor for dedup.

---

## Phase 0 — Safety, baseline, scaffolding (no behavior change)

1. **Scrub committed secret:** remove the live `sk-ant-...` key from
   `Epic2/config/config.yaml`; route key via `.env` `LLM_API_KEY`. (User to rotate
   the key — it is in git history.)
2. **Fix absolute-path violations:** `.env` `PYTHON=/home/...`, dataset2Vec
   `config.ini` `model_library_root`, `epic_4/SHAP/config/config.ini` Windows
   `[python] PYTHON`. Replace with relative/resolved-at-runtime values.
3. **Unified `requirements.txt`** at root = union of `backend`, `Epic2`,
   `epic_3`, `epic_4/*`, `dataset2Vec`, `model_library` (google-adk, litellm,
   fastapi, uvicorn, ray, pandas, numpy, scikit-learn, torch, xgboost, shap,
   optuna, faiss, pydantic, pyyaml, jsonschema, python-dotenv, certifi, pytest).
4. **Single `config/config.yaml`** consolidating `[pipeline] [ray] [hpt] [judge]
   [overfitting] [shap] [dataset2vec] [model_selection]` plus existing `config.ini`
   pipeline keys; `config.ini` reduced to `[python] [paths] [upload]` + env.
   Delete per-epic `config.ini`/`config.yaml` after migration (TASK-4.1).
5. **Preserve all docs/specs/chats/results (no deletion):** relocate every
   per-epic `SPEC.md`, `*.md`, `chat*.md`, `plan.md`/`PLAN.md`, `*results*.txt`,
   benchmark `.txt`, gaps/impl notes, CARD.md, diagrams (`.pdf/.mmd`) into
   `docs/<module>/` (renamed sensibly), per spec OUTPUT #7. These are **moved, never
   deleted**. Applies to `epic_1/**`, `Epic2/*.txt|*.md|*.pdf|*.mmd`,
   `epic_3/*spec*.md` + module READMEs/CARDs, `epic_4/**/*.md`, root
   `DESIGN_PLAN.md`/`design_plan.pdf`/`SPEC.md`/`INTEGRATION_SPEC.md`, and
   `meeting_transcripts/`. Config consolidation only merges config files; it does
   not touch any chat/spec/results/plan artifact.
6. **Baseline tests:** run every existing test suite (`backend/tests`,
   `epic_3/**/tests`, `epic_4/SHAP/tests`, `dataset2Vec/tests`) and record a green
   baseline before moving anything. (Spawn a test agent.)

## Phase 1 — Restructure + backend stitch + CLI smoke (backend-first)

1. **Move modules** into the target tree (git mv), fix imports, keep behavior.
   Repoint `backend/services/training_service.py` factories
   (`epic_3.training_orchestrator.TrainingOrchestrator`, `epic_3.ray_wrapper.RayExecutor`,
   `epic_3.events.TrainingEventBus`) to new `backend.agents.training.*` /
   `backend.orchestration.*` paths.
2. **Shared LLM client (ADK only):**
   - `llm/adk_client.py`: thin wrapper reusing existing
     `backend/agents/metadata_gen_agent.py::LlmSettingsResolver` + `LiteLlm` (this
     already does ADK-correct OpenAI/Gemini/Anthropic via `api_key`+`api_base`).
   - `llm/client_generator.py`: Epic-1 setup step renders a concrete `client.py`
     into `.mitra/<user_id>/<session_id>/llm/client.py` from chosen provider/model
     and runs the existing smoke test (`backend/agents/llm_smoke_test.py` /
     `LlmSmokeTester`).
   - **Migrate Epic 2** `Epic2/pipeline/openai_llm.py::OpenAICompatibleLlm` →
     call the shared ADK client in `feature_engineering` (selector tool is the
     only LLM consumer). **Migrate Judge** `claude_adk_llm.py`/
     `custom_anthropic_client.py` → shared ADK client. Re-test both.
3. **Pre-training bridge** (`backend/services/pipeline_prep.py`, tasks.md §1):
   - Run `FeatureEngineerOrchestrator` (Epic 2) on session `data/data.csv` →
     `engineered_dataset.csv` + `feature_artifact.json`.
   - Adapter `feature_artifact.json` → `feature_selection.json`
     (`keep/drop/engineered/rationale`) matching
     `model_selection/schemas.py::FeatureSelectionInput`.
   - Train/test split (ratio from config) → `data/train.csv`, `data/test.csv`.
   - `select_models(...)` (`model_selection/selector.py`) with `metadata.json` +
     `feature_selection.json` + `mini_data.csv` → `model_config.json`
     (`max_models` from config/UI, default 10).
   - Hook into `TrainingService.start()` (tasks.md §1.5): generate missing
     artifacts before Ray. Reuse existing `TrainingOrchestrator.prepare_and_execute_*`.
4. **Per-agent CLI + reproducible command logging** (REQ #13): standard
   `python -m backend.agents.<module>.cli` entrypoints (most already exist:
   model_selection `cli.py`, training `cli.py`, judge `run_judge.py`); each logs
   the exact command to rerun into the session log.
5. **Unified `--cli` runner** (`backend/orchestration/run_pipeline.py`): runs the
   full DAG headless against a dataset path, writing all artifacts to the session
   dir. This is the Phase-1 smoke-test driver.
6. **Smoke test (gate):** end-to-end CLI run on `Epic2/test data/.../train.csv`
   (regression) and `epic_3/model_selection/fixtures/iris_*` (classification);
   assert metadata.json → feature_selection.json → model_config.json → training_summary.json
   → judge_decision.json all produced. (Spawn a test agent; `backend/tests/test_e2e_pipeline.py`, tasks.md §5.1.)

## Phase 2 — Orchestrator (parallel + judge loop), dataset2Vec, tokens

1. **Hybrid DAG orchestrator** (`backend/orchestration/`), config-driven (JSON DAG
   map, no if-else ladders) implementing REQ #19:
   `model_selection → training → (SHAP ∥ overfitting ∥ Optuna/HPT) → judge → loop`.
   - Parallel branch via existing Ray (`RayExecutor`) / async tasks (REQ #12).
   - Reuse: `SHAPService` (evaluation/shap), `OverfittingAnalyzer`
     (evaluation/overfitting — consolidate the duplicate in HPT, TASK-4.2),
     `HyperparameterTuningAgent` (evaluation/hpt), `JudgeAgent.judge()`.
   - Build `JudgeInput` via `judge/adapter.py` (`adapt_from_hpt_results`,
     `build_shap_summary_from_csv`).
2. **Judge → model-selection feedback loop** (REQ #18/#19): if the judge rejects
   most candidates, re-invoke `select_models` excluding rejected models;
   `max_turns` (default 3) from `config.yaml` / UI advanced_settings.
3. **dataset2Vec warm-start bridge** (tasks.md §2.4): at invocation, load
   `DB/encoder.pt` in background; on a new dataset, embed via `query.py`
   (`embed_query_dataset`) + `MetaKnowledgeStore.search(...)` →
   `DatasetPrior.ranked_models`; feed as priors into
   `model_selection/agents.py` ranking. After the run, append the new dataset
   embedding + final leaderboard back to `DB/` (`write_embeddings`,
   `write_leaderboard_record`, `build_meta_kb`) — REQ #16/#17. Inference + append
   only; concurrency-guarded write.
4. **Unify event streams:** fold the in-memory `JobRegistry` (validate/metadata)
   into the `TrainingEventBus` so the whole pipeline emits one SSE stream with
   per-agent status; add resume support (REQ #14) by persisting agent
   status/inputs/outputs in the session dir and allowing re-entry at any node.
5. **Token counting** (REQ #20) + **flow logging** (REQ #21): wrap the ADK client
   to accumulate per-agent token counts into
   `.mitra/<user_id>/<session_id>/token_usage.json`; structured flow log in the
   session dir.

## Phase 3 — UI integration, visualizations, launcher

1. **Evaluation API** (`backend/routers/evaluation.py`, tasks.md §3.1):
   `GET /api/runs/{session_id}/leaderboard | /verdict | /shap` + static PNG
   serving from the session `plots/` dir.
2. **Wire React screens** (tasks.md §3.2): replace mock `LeaderboardScreen`/
   `PipelineScreen` data with live API + the unified SSE stream; show real
   per-agent status; resume-from-agent controls.
3. **Comprehensive visualizations on demand** (REQ #9): every stage dumps plots to
   the session `plots/<stage>/` dir; UI shows them only on button click (popup),
   never by default. Reuse existing dumps and add the rest. Full set:
   - **Data/EDA (feature_engineering):** missingness heatmap, per-feature
     histograms/distributions, correlation heatmap (pearson + spearman), outlier
     boxplots, target distribution / class-balance bar, mutual-info + RF-importance
     + mRMR ranking bars, PCA variance-explained + 2D scatter, feature
     count before/after selection.
   - **Model selection / dataset2Vec:** candidate ranking bar, dataset2Vec
     neighbor-similarity bar, 2D embedding projection (t-SNE/UMAP) of the new
     dataset vs the 120 corpus datasets.
   - **Training / leaderboard:** metric leaderboard bar per model, learning/loss
     curves (iterative models), confusion matrix + ROC + PR curves
     (classification), predicted-vs-actual + residual plots (regression),
     per-fold CV score boxplots.
   - **Overfitting:** train-vs-CV gap bar per model, learning curve vs sample size.
   - **HPT / Optuna:** optimization history, hyperparameter importance/sensitivity,
     parallel-coordinate + slice plots (Optuna built-ins).
   - **SHAP (exists, extend):** beeswarm/summary, bar importance, dependence plots,
     per-sample force/waterfall.
   - **Judge:** ranked-models verdict chart with score breakdown
     (performance / 1-overfitting / 1-complexity weights).
   A small shared `plotting` util (matplotlib, Agg backend, config-driven
   DPI/format/max-features) writes all of these so each agent only declares what to
   plot. Guardrails: SHAP/EDA sampling caps + timeouts (DESIGN_PLAN R14).
4. **Advanced settings (page 1):** render every `config.yaml` param; persist the
   copy into the run dir on invoke (REQ #10). Includes `max_models` (default 10),
   judge `max_turns`, split ratio, parallelism.
5. **Launcher:** `bin/mitra` (node) — invokable in any dir, creates `mitra/`
   workspace there, resolves the Python interpreter portably (no absolute paths),
   starts backend + frontend (or `--cli`). `bin/setup.sh` installs requirements.
6. **UI-interfaceable classes** (REQ #8/#10/#11): ensure each module exposes a
   clean Python entry the orchestrator/CLI and API call (most already do:
   `select_models`, `FeatureEngineerOrchestrator`, `SHAPService`, `JudgeAgent`,
   `HyperparameterTuningAgent`).

### UI <-> backend binding map (every component attached)

| UI screen / component | Backend binding | Status |
|---|---|---|
| `UploadScreen` | `POST /api/upload`, `POST /api/validate`+`/api/validate/events` SSE, `POST /api/metadata`+`/api/metadata/events` SSE | exists |
| `Settings` + `ByomFields` | `GET /api/health`, `GET /api/config/public`, `POST /api/llm/smoke-test` | exists |
| `Settings` advanced panel | NEW `GET/PUT /api/config/advanced` (full config.yaml; copied to run dir on invoke) | new |
| `TrainingPage` + `training/*` | `POST /api/training/start`, `GET /api/training/status/{id}`, `/api/training/events` SSE | exists |
| `PipelineScreen` + `AgentAvatar`/`StatusPill` | NEW unified `GET /api/pipeline/events?session_id=` (per-agent status) + `POST /api/pipeline/start` + `POST /api/pipeline/resume {from_agent}` | new (replaces data.js mock) |
| `LeaderboardScreen` + `HBars` | NEW `GET /api/runs/{id}/leaderboard | /verdict | /shap` | new |
| Plot popups (REQ #9) | NEW `GET /api/runs/{id}/plots` + static `GET /api/runs/{id}/plots/{name}` | new |
| Token usage (REQ #20) | NEW `GET /api/runs/{id}/tokens` (token_usage.json) | new |
| dataset2Vec warm-start | surfaced via `/leaderboard` (neighbors/priors) | new |
| `Dashboard` + `Sparkline` | `GET /api/runs`, `GET /api/runs/stats` (fix hardcoded 0) | exists |

Live per-component binding is powered by the Phase-2 unified event bus + a single
session-dir state file: every agent emits status to one SSE stream, so
`PipelineScreen` renders each agent card live and offers resume-from-any-agent
(REQ #14). New endpoints land in `backend/routers/{pipeline,evaluation,config}.py`,
registered in `backend/main.py`; new React callers added to
`frontend/src/api/{client.js,events.js}` and `App.jsx` passes `activeSessionId`
to every screen.

---

## Reuse map (do not rewrite)

- LLM/ADK: `backend/agents/metadata_gen_agent.py::{LlmSettingsResolver, LiteLlm}`,
  `LlmSmokeTester`.
- Pre-train: `Epic2/pipeline/orchestrator.py::FeatureEngineerOrchestrator`,
  `epic_3/model_selection/selector.py::select_models`.
- Train: `epic_3/training_orchestrator::TrainingOrchestrator`,
  `epic_3/ray_wrapper::RayExecutor`, `epic_3/events::TrainingEventBus`,
  `model_library/ml_kit.py::{MODEL_REGISTRY, MLKit}`.
- Eval: `epic_4/SHAP/.../shap_service.py::SHAPService`,
  `epic_4/overfitting_analysis_tool::OverfittingAnalyzer`,
  `epic_4/hyperparameter_tuning_agent::HyperparameterTuningAgent`,
  `epic_4/judge_agent::{JudgeAgent, UpstreamAdapter}`.
- Warm-start: `dataset2Vec/d2v_core/store.py::MetaKnowledgeStore`,
  `dataset2Vec/query.py`.
- Backend: `SessionManager`, `DatasetNormalizer` (mini_data), `DataValidator`,
  `ConfigLoader`.

## Risks

- ADK migration of Epic 2 / Judge may shift LLM behavior → re-run their suites.
- True parallelism (SHAP/Optuna/overfitting) under Ray needs resource limits to
  avoid OOM (DESIGN_PLAN R14: SHAP ≤1000 rows, timeout).
- dataset2Vec DB write-back concurrency → single-writer lock.
- Surfacing every config param in UI is a large surface; exclude secrets/paths.
- Resume/context-management adds state-store complexity; keep one session-dir
  state file as the single source of truth.
- Big restructure can break imports broadly → move in small, test-gated steps.

## Verification

- Phase 0: all existing suites green pre-move (recorded baseline).
- Phase 1: `python -m backend.orchestration.run_pipeline --dataset <csv> --cli`
  produces every artifact; `backend/tests/test_e2e_pipeline.py` passes for iris +
  regression fixtures. Spawn a test agent per CLAUDE.md.
- Phase 2: forced-rejection input exercises the judge loop (≤ max_turns);
  dataset2Vec query returns neighbors and appends to `DB/`; `token_usage.json`
  populated.
- Phase 3: `bin/mitra` in a fresh dir starts the app; upload→pipeline→leaderboard
  works in the browser; plots pop up on click; `--cli` runs headless.
- After approval, also copy this plan to `plans/integration_<timestamp>.md` per
  repo convention (rule 25); each phase's tests run via a spawned test agent
  (rule 26).
