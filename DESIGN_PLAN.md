# MITRA v2 — Design Plan (Reference for Agents)

> Source: `design_plan.pdf` ("MITRA v2 — Improved Design Plan: Self-Hosted Agentic AutoML",
> 30 pages, internal draft). This file is a faithful text transcription/summary of that PDF so
> other agents do not have to re-read the binary. Where the PDF gives exact code, schemas, or
> thresholds, they are reproduced verbatim.

## 0. One-paragraph summary

MITRA v2 is a **self-hosted, agentic AutoML platform**. A user uploads a tabular CSV (or an image
ZIP), and a pipeline of agents validates it, infers metadata, engineers features, selects and trains
several ML models on a Ray cluster, judges them for overfit/underfit, optionally tunes
hyperparameters, and publishes a leaderboard. The UI is Streamlit (3 pages) fed by a Server-Sent
Events (SSE) bus. The agent harness is **Google ADK (adk-python)**. Models are bring-your-own
(BYOM) via a **LiteLLM factory** (OpenAI / Anthropic / Gemini). Everything runs in a single
self-hosted Docker container.

| Field | Value |
|---|---|
| Version | v2 (improves on v1) |
| Team | 8 people, 2 hrs/day, 14 working days (28 hrs/person) |
| UI | Streamlit (3 pages) + SSE event bus |
| Agent framework | Google ADK (adk-python) |
| Agents | 8, one per person: DataValidator, MetadataGen, FeatureSelection, ModelSelection, Classification, Regression/USL, Judge, HPT |
| Compute | Ray (local head + workers) |
| BYOM | LiteLLM factory -> OpenAI / Anthropic / Gemini |
| Server | FastAPI + Uvicorn (self-hosted, Docker image) |

The four v2 themes: (1) every component has a low-level spec, (2) all inter-agent data contracts are
explicitly typed, (3) 14 risks have concrete code-level fixes, (4) the 8-person sprint is planned at
task-hour granularity so each engineer owns one agent end-to-end.

---

## 1. What changed from v1 (betterments)

- **B1** Added **DataValidator** as the very first stage (fail fast on nulls >80%, zero-variance cols, wrong format).
- **B2** Defined the **SSE Event Bus** as an `asyncio.Queue` per session with a typed event schema.
- **B3** Specified inter-agent **I/O contracts**: exact file names, JSON schemas, who writes vs reads.
- **B4** Gave the **Template Library** a concrete structure: `train.py.jinja2`, `hp_space.yaml`, `resources.yaml` per family.
- **B5** Made the **Judge Agent decision tree** explicit (gap threshold, floor threshold, max-retries, SHAP OOM guard).
- **B6** Fully specified the **USL Agent** (K-Means / DBSCAN / Isolation Forest; eval via silhouette / Davies-Bouldin).
- **B7** **14-risk matrix** with code-level mitigations (not policy statements).
- **B8** **8-person sprint** at task-hour granularity, 1 agent per person, parallel tracks.
- **B9** **Integration test plan** with two canonical fixtures (`iris.csv`, `cats-dogs-10.zip`).
- **B10** `metadata.json` strict schema with JSON-Schema validation at write time.

---

## 2. Resolved ambiguities (key decisions)

- **USL feature columns**: `metadata.json` always includes `output_cols: []` for unsupervised. The agent uses all non-dropped columns as features. If `output_cols` is non-empty, USL is never chosen.
- **Ports**: Streamlit 8501, FastAPI 8001, Ray head 6379, Ray dashboard 8265. Only 8501 and 8001 are mapped by default.
- **Concurrent Ray training**: each `model_NNN/` subdir is created atomically before `ray.remote()`. Leaderboard is re-sorted on every `GET /leaderboard` by reading all `metrics.json`.
- **Error/Debug loop limit**: max 3 self-correction attempts per script; on the 4th failure verdict=abort, model skipped. Limit is in session config, tunable.
- **Feature Selection no-orphan rule**: outputs `feature_selection.json` with keep/drop arrays validated against `data_encoded.csv` headers before the scaler runs. Mismatch = validation error + abort.
- **Unknown model name**: `TemplateResolver` returns a fallback flag; the family agent invokes the write-from-scratch path with the LLM, using the closest template as a few-shot example.
- **Streamlit SSE**: P2 uses `st.empty()` + a `requests.get(..., stream=True)` loop in an `st.fragment` with a 500ms re-run interval.
- **`mini_data.csv`**: `pandas.describe(include='all')` + dtype per column + null_count + unique_count + `is_pii_suspect` (bool from column-name heuristic). Capped at 1000 rows. Always generated before any LLM call.

---

## 3. System architecture (refined)

Five-tier structure preserved from v1. Three refinements: (1) the SSE Event Bus is a first-class
component; (2) DataValidator is the first stage before any LLM touch; (3) the **LiteLLM client
factory is the only path to any LLM** — no agent makes a direct HTTP call.

Everything runs inside **one Docker container**:
- **Browser/User** -> **Streamlit UI :8501** (P1 Upload+Validation, P2 Live Pipeline status, P3 Leaderboard+Explain)
- **FastAPI/Uvicorn :8001** (Session Router + SSE Publisher; Event Bus = asyncio.Queue)
- **Google ADK Runtime**: Root Orchestrator (SequentialAgent) -> stages: DataValidator, Training Orchestrator, Preprocessor (Encoder+Scaler), HPT Agent, MetadataGen, FeatureSelection, ModelSelection (LLM call), Judge (+ Error/Debug agent on error path)
- **Ray Cluster :6379** (Ray Head + workers) for training/eval
- **LiteLLM Gateway** (env-configured) -> OpenAI / Anthropic / Gemini
- **Local Disk** `.mitra/<sid>` (data, config, models, logs)

---

## 4. End-to-end pipeline with data contracts

Every arrow carries a **named artifact**. No stage reads a file it did not explicitly receive.
DataValidator gates everything and is the cheapest place to fail.

Pipeline (Figure 2): `User Upload (data_file, description, metadata?)` -> **Data Validator**
(schema/null check / format detect) -> (invalid -> Validation Error -> P1 banner + fix hint) /
(valid -> **Metadata Gen Agent** LLM call -> unified schema) -> `metadata.json`, `mini_data.csv` ->
**Categorical Encoder** (chunked, try-float heuristic) -> `data_encoded.csv`, `encoder_map.json` ->
**Scaler** (Std | MinMax | Log per column) -> `data_scaled.csv`, `scaler_map.json` -> **Feature
Selection Agent** (Spearman/χ²/MI/RF importance) -> `feature_selection.json` (keep, drop,
engineered) -> **Model Selection LLM Call** (one pass, few-shot) -> `model_config.json`
[{name, family, hp_space, rationale}] -> **Training Orchestrator** (routes to family agent) ->
`problem_type?` -> Classification / Regression / USL agent -> **Template Resolver** (library hit ->
fill params; miss -> write from scratch) -> `train_script.py`, `resources.yaml` -> **Ray Wrapper**
(`ray.remote()` + log stream) -> `model.pkl|pt`, `train_metrics.json` -> **Evaluator** (holdout
metrics + SHAP sample) -> `eval_metrics.json`, `shap_values.npy` -> **Judge Agent** (bias-var
analysis + verdict) -> `verdict?` -> accept (-> Leaderboard, update top-10) / abort (max 3 retries
-> log + skip) / overfit|underfit (-> **HPT Agent** random search ~20 trials -> new HP -> retrain).

### 4.1 Stage-by-stage data contract table

| Stage | Reads | Writes | Owner | Type |
|---|---|---|---|---|
| Data Validator | data_file, type_hint | validation_report.json | P3 | Python |
| Metadata Gen | mini_data.csv, description, metadata?(raw) | metadata.json, mini_data.csv | P4 | **LLM Agent** |
| Encoder | data.csv (chunked), metadata.json | data_encoded.csv, encoder_map.json | P3 | **Python** |
| Scaler | data_encoded.csv, metadata.json | data_scaled.csv, scaler_map.json | P3 | **Python** |
| Feature Selection | mini_data.csv, metadata.json, data_encoded header | feature_selection.json | P5 | **LLM Agent** |
| Model Selection | metadata.json, feature_selection.json, mini_data | model_config.json | P6 | **LLM Call** |
| Training Orch | model_config.json | (routes to family agent) | P6 | LLM Agent |
| Family Agent | model_config.json[i], templates/ | train_script.py, resources.yaml | P6 | LLM Agent |
| Ray Wrapper | train_script.py, data_scaled.csv | model.pkl\|pt, train_metrics.json | P1 | **Python** |
| Evaluator | model.pkl\|pt, data_scaled.csv (holdout) | eval_metrics.json, shap_values.npy | P1 | **Python** |
| Judge Agent | train+eval metrics, shap_values | verdict.json | P7 | LLM Agent |
| HPT Agent | model_config[i], hp_space.yaml, verdict.json | new_hp.json | P7 | **Python** |
| Error+Debug | error_log, failed_script | patched_script, error_analysis | P2 | LLM Agent |

> **Design intent on LLM usage** (important for simplification questions): encoding, scaling,
> training, evaluation, HPT, and validation are **deterministic Python — no LLM**. LLM calls are
> reserved for MetadataGen (1 call), FeatureSelection (agent), ModelSelection (1 call, no loop),
> family-agent template fallback (only on cache miss), Judge, and the Error/Debug path. Most stages
> are pure code.

### 4.2 `metadata.json` — full schema (single source of truth)

Validated with `jsonschema.validate()` immediately after MetadataGen writes it.

```json
{"$schema":"...", "type":"object",
 "required":["problem_type","output_cols","input_cols","drop_cols","col_types","data_format","row_count","col_count"],
 "properties": {
   "problem_type": {"enum":["classification","regression","unsupervised"]},
   "output_cols": {"type":"array","items":{"type":"string"}},
   "input_cols": {"type":"array","items":{"type":"string"}},
   "drop_cols": {"type":"array","items":{"type":"string"}},
   "col_types": {"type":"object","additionalProperties":{"enum":["numeric","categorical","text","datetime","image_path"]}},
   "data_format": {"enum":["tabular","image"]},
   "row_count": {"type":"integer","minimum":10},
   "col_count": {"type":"integer","minimum":1},
   "user_description": {"type":"string","minLength":20},
   "class_balance": {"type":"object"}
 }}
```

---

## 5. SSE Event Bus — low-level design

Flow (Figure 4): any ADK agent or Python stage calls `emit_event(stage, msg, pct)` (ADK tool) ->
per-session `asyncio.Queue` -> FastAPI SSE generator -> Streamlit P2 `st.empty()` polling loop.

### 5.1 Event schema
```json
{"session_id":"", "stage":"feature_selection",
 "level":"info|warn|error", "msg":"chi-square on 12 cols",
 "pct":42, "ts":"2026-05-24T10:32:11Z"}
```

### 5.2 FastAPI SSE endpoint
```python
async def event_stream(session_id: str):
    q = session_queues[session_id]  # asyncio.Queue, created at POST /session
    while True:
        event = await asyncio.wait_for(q.get(), timeout=300)
        if event is SENTINEL: break
        yield f"data: {json.dumps(event)}\n\n"

@app.get("/session/{sid}/events")
async def stream(sid: str):
    return StreamingResponse(event_stream(sid), media_type="text/event-stream")
```

### 5.3 `emit_event()` ADK tool (agent side)
```python
def emit_event(stage:str, msg:str, pct:int=0, level:str="info") -> str:
    event = {"session_id": ctx.session_id, "stage": stage, "level": level,
             "msg": msg, "pct": pct, "ts": utcnow()}
    asyncio.get_event_loop().call_soon_threadsafe(q.put_nowait, event)
    return "ok"
```

> **Risk R4 addressed**: queue is unbounded (`maxsize=0`). If Streamlit disconnects, the generator
> exits and the SENTINEL drains the queue. Ray workers are unaffected.

---

## 6. Agent I/O contracts (all 8 agents)

Each boundary is a named file validated by the receiving agent (Figure 3). Inputs (purple) -> agent -> outputs (green):

- **DataValidator**: `data_file, type_hint` -> `validation_report.json {ok, errors[], warnings[]}`
- **MetadataGen**: `mini_data.csv, user_description, user_metadata?(raw)` -> `metadata.json {problem_type, output_cols, input_cols, drop_cols, col_types, row_count}`
- **FeatureSelection**: `mini_data.csv, metadata.json, data_encoded.csv (header only)` -> `feature_selection.json {keep[], drop[], engineered[], rationale{}}`
- **ModelSelection**: `metadata.json, feature_selection.json, mini_data.csv` -> `model_config.json [{name, family, hp_space, rationale}]`
- **FamilyAgent**: `model_config.json[i], data_scaled.csv (path only), template_library/` -> `train_script.py, resources.yaml`
- **JudgeAgent**: `train_metrics.json, eval_metrics.json, shap_values.npy, model_config.json[i]` -> `verdict.json {verdict, reason, next_hp_hint}`
- **HPTAgent**: `model_config.json[i], hp_space.yaml, verdict.json` -> `new_hp.json {params:{}}`
- **ErrorDebug**: `error_log.txt, failed_script.py` -> `patched_script.py, error_analysis.txt`

---

## 7. Data Validator Agent

- Owner P3. **Python (deterministic, no LLM).** Tools: read_file, write_file.
- Inputs: `data_file (path)`, `type_hint: tabular|image`.
- Outputs: `validation_report.json -> {ok:bool, errors:[], warnings:[], detected_format:str, row_count:int, col_count:int}`. On `ok=false`: error surfaced to P1, pipeline halts.
- Guardrails: never loads full file into RAM (chunked `pd.read_csv(chunksize=10000)`). Error enums: `NULL_EXCESS | ZERO_VARIANCE | BAD_FORMAT | ROW_COUNT_TOO_LOW | IMAGE_LAYOUT_INVALID`. Image mode walks the ZIP, checks every subfolder has >=5 images, rejects flat structure. Tabular mode rejects if any column >80% nulls or `row_count < 50`.
- File: `mitra/agents/data_validator/validator.py`. **R1 mitigation**: validator itself must not load the full file.

---

## 8. Metadata Gen Agent

- Owner P4. **LLM Agent (single LLM call + schema validation).** Tools: read_file, write_file.
- Inputs: `mini_data.csv` (always present), `user_description` (>=20 words), `user_metadata.json` (optional).
- Outputs: `metadata.json` validated against §4.2 schema. On schema failure: one retry with the validation error appended; second failure -> abort with structured error.
- Guardrails: validated with `jsonschema.validate()` right after the LLM returns. User metadata is a hint (cannot override row_count/col_count). Min description length enforced at UI (20 words) and here. **No tool access to `data.csv` — only `mini_data.csv`.**
- Prompt rules (from `prompt.md`): never hallucinate column names; only use columns from mini_data; map a user-mentioned target into `output_cols`; drop PII columns (phone, aadhaar, SSN, email, id) into `drop_cols`; output ONLY the JSON, no fences.

---

## 9. Feature Selection Agent

- Owner P5. **LLM Agent (tools + stats).** Tools: read_file, write_file, python_exec.
- Inputs: `mini_data.csv`, `metadata.json`, `data_encoded.csv (header row only — not data)`.
- Outputs: `feature_selection.json {keep:[], drop:[], engineered:[{name, formula}], rationale:{col:reason}}`. Validated: `keep ∪ drop == all non-drop_cols` from metadata (no orphan columns).
- Guardrails: **HARD GUARDRAIL** — `python_exec` has the `data_encoded.csv` path REMOVED from its context; only the header row is provided as a string. Agent must not call `pd.read_csv` on the full file (tool returns `PermissionError`). Spearman/Pearson/χ² are computed by a Python helper on mini_data — the agent calls the helper, not raw pandas. Engineered features limited to degree-2 polynomial and ratio features (no arbitrary lambda). Every column in `keep` must exist in the `data_encoded.csv` header.

---

## 10. Model Selection Call

- Owner P6. **Single LLM call (no tools, no loop).**
- Inputs: `metadata.json`, `feature_selection.json`, `mini_data.csv`.
- Output: `model_config.json: [{name, family, rationale, hp_space:{param:range}, priority}]` sorted by expected fit.
- Guardrails: single pass — response parsed immediately; if JSON invalid, one retry. Output list capped at 8 models (prompt: "return between 3 and 8 models"). `family` must be one of: `xgboost | random_forest | logistic_reg | svm | mlp | cnn_image | kmeans | dbscan | isolation_forest`. `hp_space` values are ranges/lists (not single values) — they feed directly into Optuna's `suggest_*` API.

---

## 11. Family Agents (Classification / Regression / USL)

| Agent | Algorithms | Eval Metrics | Key constraints |
|---|---|---|---|
| Classification | XGBoost, Random Forest, Logistic Reg, SVM, MLP, CNN (images) | Accuracy, F1-macro, ROC-AUC, Confusion Matrix | Class imbalance: stratified split; if imbalance >5:1, force `class_weight='balanced'` |
| Regression | XGBoost-Reg, Random Forest-Reg, Ridge, Lasso, MLP-Reg | RMSE, MAE, R², residual plot | Outlier rows (\|z\|>3) flagged in eval report but NOT auto-dropped |
| USL | K-Means (k=2..√n), DBSCAN (eps from nearest-neighbor), Isolation Forest | Silhouette, Davies-Bouldin, Calinski-Harabasz; Isolation Forest -> anomaly_ratio | `output_cols` must be empty in metadata.json; else orchestrator rejects USL routing |

### 11.1 Template Resolution Protocol (all family agents)
1. Read `model_config.json[i].family`.
2. Look up `mitra/templates/{family}/train.py.jinja2`. Exists -> Step 3, else -> Step 5.
3. Read `hp_space.yaml` from same folder; merge with `model_config[i].hp_space` (config overrides template defaults).
4. Render Jinja2 template with `{n_classes, input_dim, output_cols, hp_dict}`. Write `train_script.py`. Done.
5. (fallback) Call LLM with closest template as a few-shot example. LLM writes the full script. Run syntax check (`py_compile`). On fail -> retry once.

> **R3 mitigation**: Step 2 means the LLM is only called for genuinely novel architectures. All
> standard models (XGBoost, RF, MLP, CNN) always hit Step 4.

---

## 12. Template Library

Each family folder ships the same files (Figure 5). `TemplateResolver` does a name->folder lookup, then a Jinja2 render.

```
mitra/templates/xgboost/
  train.py.jinja2     # Jinja2 with {{n_estimators}}, {{max_depth}}, {{lr}}, {{output_cols}}
  hp_space.yaml       # Optuna search space defaults
  resources.yaml      # {cpu: 2, gpu: 0, mem_gb: 4}
  test_smoke.py       # pytest fixture: trains on 50 rows of iris, asserts accuracy > 0.5
```

Folders: `xgboost/`, `random_forest/`, `mlp/`, `cnn_image/`, `svm/`, `kmeans/`, `logistic_reg/` (each ships the four files above).

### 12.2 `hp_space.yaml` format
```yaml
# xgboost/hp_space.yaml
n_estimators: {type: int, low: 50, high: 500, step: 50}
max_depth:    {type: int, low: 3, high: 10}
learning_rate:{type: float, low: 0.01, high: 0.3, log: true}
subsample:    {type: float, low: 0.6, high: 1.0}
colsample:    {type: float, low: 0.6, high: 1.0}
```
HPT Agent reads this and calls `trial.suggest_int()`, `trial.suggest_float()` accordingly. No hard-coded search spaces in agent prompts.

---

## 13. Judge Agent

Full decision tree (Figure 7), thresholds are config-driven:
1. Compute `gap = train_score - val_score`.
2. `gap > OVERFIT_THRESHOLD (0.15)`? yes -> overfit branch.
3. else `val_score < FLOOR (e.g. <0.55 acc)`? yes -> underfit branch.
4. else -> run SHAP (stratified sample <=1000 rows) -> **verdict: accept** (write to leaderboard).
5. overfit/underfit -> `retries >= MAX_RETRIES (3)`? yes -> **verdict: abort** (reason: max retries exhausted); no -> **verdict: overfit** (next_hp_hint: reduce complexity ↓depth, ↓estimators, +L2) or **verdict: underfit** (next_hp_hint: increase capacity ↑depth, ↑estimators, ↑layers).

### 13.1 Configurable thresholds (session config)
```
OVERFIT_THRESHOLD: 0.15   # train_score - val_score > this -> overfit
FLOOR_SCORE: 0.55         # val_score < this (classification) -> underfit
FLOOR_R2: 0.10            # R² < this (regression) -> underfit
MAX_RETRIES: 3            # per model before abort
SHAP_MAX_ROWS: 1000       # cap to avoid OOM
SHAP_TIMEOUT_SEC: 120     # kill SHAP if exceeded
```
> **R14 mitigation**: SHAP runs in a Ray remote task with a memory limit. On timeout (120s), the verdict is still computed from metrics only and SHAP is marked unavailable.

### 13.2 `verdict.json` schema
```json
{"verdict":"accept|overfit|underfit|abort", "gap":0.18, "val_score":0.81,
 "retries_used":1, "reason":"train_acc=0.97 val_acc=0.79, gap=0.18 > threshold 0.15",
 "next_hp_hint":{"reduce":["max_depth","n_estimators"], "increase":["subsample"]},
 "shap_available":true}
```

---

## 14. HPT Agent

- Owner P7. **Python (Optuna, no LLM in loop).** Tools: optuna_run, ray_submit.
- Inputs: `model_config.json[i]`, `hp_space.yaml`, `verdict.json (with next_hp_hint)`.
- Outputs: `new_hp.json {trial_id, params:{}, expected_improvement}`, `hp_history.json` (all trials, for P3).
- Guardrails: no LLM in loop (cost). `next_hp_hint` biases the Optuna sampler (suggested 'reduce' params capped at current value, 'increase' floored). Hard budget: max 20 trials OR 5 minutes wall-clock per model. Stratified sampling for class-imbalanced data. Each trial runs as `ray.remote()`.
```python
def objective(trial):
    hp = sample_from_space(trial, hp_space, hint=verdict.next_hp_hint)
    ref = ray_submit(train_script, hp, data_path); metrics = ray.get(ref)
    return metrics['val_score']
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, timeout=300, callbacks=[MaxRetriesCallback(3)])
```

---

## 15. Error + Debug Agent

- Owner P2. **LLM Agent.** Tools: read_file, write_file, bash, python_exec (restricted).
- Inputs: `error_log.txt (last 200 lines)`, `failed_script.py`, `resources.yaml`.
- Outputs: `patched_script.py`, `error_analysis.txt {error_type, root_cause, fix_applied, confidence}`.
- Guardrails: may ONLY write files under `.mitra/<session-id>/` (MITRA source tree mounted read-only). Max 3 self-correction attempts; 4th -> abort event, skip model. `bash` restricted to: python, `pip install` (whitelist only), cat, head. No curl/wget/git. pip whitelist: scikit-learn, xgboost, lightgbm, torch, shap, optuna, pandas, numpy.

---

## 16. Ray Wrapper

Deterministic Python module (not an agent). Four responsibilities:
- **Start**: `ray.init(address='auto', ignore_reinit_error=True)`. If Ray not running, spawn a local head with `ray.init(num_cpus=os.cpu_count()-1)`. Reports to `/health`.
- **Submit**: `@ray.remote(num_cpus=N, num_gpus=G)`-decorated `run()`; N/G from `resources.yaml`. Falls back to CPU-only if no GPU.
- **Stream logs**: `ray.get()` is blocking; logs tailed by a background thread emitting SSE every 2s.
- **Teardown**: on `DELETE /session`, `ray.cancel(ref, force=True)` for active refs. Cluster stays alive for the next session.
> **R5 mitigation**: `/health` checks `ray.is_initialized()` on startup; P1 blocks session creation if `ray_ok=false`.

---

## 17. Risk Register v2 — 14 risks with code-level mitigations

| ID | Risk | Sev | Code-level mitigation |
|---|---|---|---|
| R1 | Agent reads full dataset | Crit | TemplateResolver & all agent tool contexts get ONLY `mini_data.csv` path. `data_scaled.csv` path injected only into the Ray worker subprocess, never an ADK context. |
| R2 | ModelSel circles (agent loop) | Crit | Model Selection is a FunctionCall, not an Agent. ADK runner invokes it as a single turn. No tools -> no loop. |
| R3 | Template regen every session | High | `TemplateResolver.resolve(family)` checks `templates/` before any LLM call. Cache miss ~0% for standard families. |
| R4 | SSE asyncio.Queue deadlock | High | Queue unbounded (maxsize=0). FastAPI SSE generator has 300s timeout. On disconnect, generator exits and SENTINEL drains pending events. |
| R5 | Ray cluster fails to start | High | `/health` polls `ray.is_initialized()` every 5s. P1 blocks session start with a banner if `ray_ok=false`. Docker `--shm-size=2g` to prevent Ray shared-memory OOM. |
| R6 | LLM key misconfigured | High | Startup smoke test: `client.complete([{role:user, content:'Reply OK'}])`. Fails with exit code 1 and prints the upstream HTTP error. Container does not serve requests. |
| R7 | Judge thresholds too tight | High | `OVERFIT_THRESHOLD`, `FLOOR_SCORE` in session_config (not hard-coded). Conservative defaults (0.15, 0.55). Overridable via `?overfit_thresh=0.2` query param. |
| R8 | Optuna runs unbounded | Med | `study.optimize(n_trials=20, timeout=300)`. MaxRetriesCallback halts after 3 failed trials. Budget logged as SSE event. |
| R9 | metadata.json schema drift | High | `jsonschema.validate(metadata, SCHEMA)` right after MetadataGen writes it. Pipeline halts on mismatch. Schema version embedded in JSON. |
| R10 | Error agent edits platform code | High | MITRA source mounted read-only. bash tool whitelist: only `.mitra/`. `os.path.abspath` check before every write. |
| R11 | Image ZIP bad layout | Med | DataValidator walks ZIP tree; any image file in root (not a subfolder) -> `IMAGE_LAYOUT_INVALID` + one-line fix in P1. |
| R12 | Token cost spike | Med | Per-session token counter; agents report usage via emit_event. At 80% of `TOKEN_BUDGET` (default 100K) a P2 warning banner appears. At 100%, pipeline pauses and asks. |
| R13 | USL target leakage | High | Training Orchestrator: if `problem_type=='unsupervised'` AND `output_cols!=[]`, reject with `USL_TARGET_LEAKAGE` before any model runs. |
| R14 | SHAP OOM on large model | Med | SHAP runs in `ray.remote(memory=2*1024**3)`. If it exceeds 2GB or 120s, Ray kills the task and verdict is metrics-only. `shap_available=false`. |

Risk matrix (Figure 9): Critical (red) = R1 (agent reads full dataset), R2 (ModelSel loop). High
(orange) = R3, R4, R5, R6, R9, R13, R10. Medium (yellow) = R7?, R8, R11, R12, R14, plus SHAP OOM.

---

## 18. 8-Person Sprint Plan (2 hrs/day × 14 days = 28 hrs/person)

Each person owns one agent end-to-end (prompt, schema, tests, integration). P1 (Infra) and P2
(API+SSE) are the backbone; their first 4 days must finish before any agent can emit events. All
agent owners start with their JSON schema and unit tests on Day 1.

### 18.1 Agent ownership matrix

| Person | Primary Agent | Secondary tasks | Mid-sprint deliverable (D8) |
|---|---|---|---|
| P1 — Infra | — | Docker, bin/mitra, Ray, template library skeleton | Ray starts cleanly; /health green; 7 template folders created |
| P2 — API+SSE | — | FastAPI router, asyncio Event Bus, SSE endpoint | POST /session creates queue; GET /events streams from stub agent |
| P3 — Validator+Pre | DataValidator | Chunked encoder, scaler, mini_data generator | DataValidator passes iris.csv; data_scaled.csv produced |
| P4 — Metadata | MetadataGen Agent | JSON-Schema validator, merge logic | iris.csv + 30-word description -> valid metadata.json |
| P5 — Features | FeatureSelection Agent | Spearman/χ² helpers, PII few-shot prompts | feature_selection.json produced from iris metadata |
| P6 — ModelSel+Train | ModelSelection + Classification | Regression+USL prompts, training orch routing | One XGBoost model trained on Ray; metrics.json written |
| P7 — Judge+HPT | Judge Agent + HPT Agent | SHAP wrapper, Optuna wrapper, hp_space loader | Judge produces verdict.json for the D8 XGBoost model |
| P8 — UI+IntTest | — | Streamlit P1/P2/P3, E2E test fixtures | P1 upload works; P2 shows SSE events from stub |

### 18.2 Day-by-day milestones (16h/day across 8 people)

| Day | Milestone | Acceptance |
|---|---|---|
| D2 | Infra alive | bin/mitra boots; /health green; SSE queue per session |
| D4 | P1 UI + Validator | Upload CSV in P1; DataValidator returns validation_report.json |
| D6 | Metadata + Preprocessing | mini_data.csv + metadata.json + data_scaled.csv for iris.csv |
| D8 | Mid-sprint: one model trained | FeatureSel -> ModelSel -> XGBoost train on Ray -> metrics.json |
| D10 | Judge + HPT loop | Judge produces verdict -> HPT runs 5 trials -> retrain on Ray |
| D12 | All agents integrated | Full pipeline: iris.csv -> leaderboard (3+ models) |
| D13 | P3 UI + SHAP plots | Leaderboard renders; SHAP bar chart for top model in P3 |
| D14 | Image path + E2E tests | cats-dogs-10.zip -> CNN model -> leaderboard; both pytest suites pass |

---

## 19. Integration Test Plan

### 19.1 Fixture 1 — `iris.csv` (tabular classification)
150 rows, 4 features, 3 classes (small so CI < 5 min). Expected: metadata problem_type=classification,
output_cols=[species]; feature_selection keeps all 4; model_config includes xgboost + random_forest;
leaderboard top-1 accuracy >= 0.90; verdict=accept (no overfitting).

### 19.2 Fixture 2 — `cats-dogs-10.zip` (image classification)
10 images per class (cats/, dogs/), no GPU needed. Expected: DataValidator accepts ZIP;
model_config includes cnn_image; CNN trains, accuracy >= 0.60; shap_available=false (CNN SHAP slow, falls back).

### 19.3 Negative test cases (P8 owns)
| Test | Input | Expected behaviour |
|---|---|---|
| T-NEG-1 | CSV with 80% nulls in column A | DataValidator: NULL_EXCESS; P1 shows fix hint |
| T-NEG-2 | ZIP with flat structure | DataValidator: IMAGE_LAYOUT_INVALID; clear message |
| T-NEG-3 | description = 'hello' (2 words) | P1 rejects before POST /session (client-side validation) |
| T-NEG-4 | TYPE=openai but wrong API key | Startup smoke test fails; container exits code 1; no requests served |
| T-NEG-5 | USL dataset with output_cols=['target'] | Training Orchestrator: USL_TARGET_LEAKAGE; model skipped |
| T-NEG-6 | LLM returns family='mystery_net' | TemplateResolver: fallback to write-from-scratch; LLM writes script; smoke test runs |

---

## 20. Acceptance Criteria v2 (all must pass before tagging the Docker image stable)

| Tier | Criterion | Measured by |
|---|---|---|
| Functional | iris.csv -> 3+ trained models, top-1 acc >= 0.90 | pytest test_iris_e2e.py |
| Functional | cats-dogs-10.zip -> CNN, acc >= 0.60 | pytest test_image_e2e.py |
| Robustness | Empty metadata.json + 30-word description -> pipeline completes | pytest test_no_metadata.py |
| BYOM | TYPE=openai -> TYPE=anthropic in .env -> pipeline completes unchanged | Manual / pytest parametrize |
| BYOM | TYPE=openai with wrong key -> container exits code 1 in < 5s | pytest test_smoke_fail.py |
| Self-hosting | docker run mitra:latest on clean Ubuntu 22.04 VM -> P1 loads | Manual |
| Cost | iris.csv end-to-end consumes < 50K tokens | Token counter in SSE logs |
| SSE | P2 progress bars update without page refresh | Playwright E2E |
| Reliability | Kill LLM endpoint mid-run -> P2 error banner, no orphaned Ray jobs | pytest test_llm_failure.py + ray.available_resources() |
| Perf | iris.csv full pipeline (train 3 models) < 4 min on 4-core laptop | pytest + time assertion |
| Security | Error/Debug agent cannot write to mitra/ source tree | pytest test_agent_sandbox.py (PermissionError) |
| Data | Feature selection output validated against data_encoded.csv header — no orphan columns | pytest test_feature_schema.py |

---

## Appendix — How Epic2 (this repo's `Epic2/`) relates to this plan

`Epic2/` implements the **feature-engineering slice** as a standalone ADK pipeline
(`profile_data -> infer_types -> handle_missing -> handle_outliers -> create_features_pre ->
encode_features -> create_features_post -> scale_features -> select_features -> validate_features ->
write_report`). It is **not** a 1:1 match to the design above:

- The design keeps **Encoder and Scaler as deterministic Python (no LLM)**; Epic2 currently routes
  scaling (and missing/outlier/feature-creation/selection/report) through the LLM. This is the main
  source of the "LLM calls across many tools" cost and is the primary simplification target.
- The design's LLM budget is roughly: MetadataGen (1 call), FeatureSelection (agent),
  ModelSelection (1 call), Judge, family-template fallback (cache miss only), Error/Debug. Everything
  else is pure code.
- Epic2's LLM transport is an **OpenAI-compatible client** (`pipeline/openai_llm.py`), so an
  Anthropic key must use Anthropic's OpenAI-compatible endpoint
  (`base_url: https://api.anthropic.com/v1/`, plain model id e.g. `claude-sonnet-4-6`).
