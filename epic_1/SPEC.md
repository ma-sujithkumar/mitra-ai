# SPEC: Epic 1 - Upload, Validate & Metadata Generation (Page 1)

**Assignees:** Vidhi Kant Gupta, Sebin Francis Kannampuzha
**Tech Stack:** ReactJS (frontend), Python/FastAPI (backend), Google ADK (agent), LiteLLM (LLM routing)

---

## 1. GOAL

Implement Page 1 of MITRA, the entry point where users upload a dataset, provide
minimal metadata hints, configure their LLM credentials, and trigger an automated
two-phase validation + metadata extraction before the pipeline begins.

Contraints:
1. User needs to upload a csv file with the data
2. Need a browse file option from frontend
3. Need validations of file format in both frontend and backend
4. Once the validation is done first we need to create a .mitra folder and save the uploaded file locally

The two artifacts produced by Epic 1 flow into all downstream epics:
- `validation_report.json` — deterministic data quality gate.
- `metadata.json` — structured contract read by every subsequent agent.

---

## 2. SCOPE 

### In scope only confied to epic-1 changes
- Global app shell: sidebar navigation, top bar, routing between pages (Dashboard,
  New Run, Pipeline, Leaderboard).
- Dashboard page (overview cards, recent runs table, agent roster, accuracy trend).
- New Run page (Page 1): file upload, dataset picker, metadata form, BYOM credentials.
- `.env` configuration schema and LLM smoke-test on startup.
- `config.ini` — single global config file for all controllable parameters. Should contain defaults for all necessary configs if user does not specify
- Client of anthropic, open ai, gemini compatible APIs to connect to the BYOM LLM (no LLM involved in this step)
- Backend API (Python/FastAPI) for file upload and session workspace management.
- Data Validator (Python, deterministic, no LLM): 6 checks, produces
  `validation_report.json`.
- Metadata Generation Agent (LLM via Google ADK): reads mini-data statistics + the user input for data description + system level prompt on what needs to be acheived,
  produces `metadata.json` against a fixed JSON schema.
- Session workspace: `.mitra/<session_id>/` directory layout.

### Out of scope (handled in later epics)
- Page 2: preprocessing pipeline (encoding, scaling, feature selection, model
  selection, training on Ray).
- Page 3: leaderboard, SHAP report.
- HPT loop, judge agent, drift monitor.
- Docker packaging.

---

## 3. APPLICATION CONTEXT

```
User browser (ReactJS)
    |
    | HTTP / SSE
    v
FastAPI backend (Python)
    |-- /api/upload        => saves dataset, creates session workspace
    |-- /api/validate      => runs Data Validator, streams progress via SSE
    |-- /api/metadata      => runs Metadata Gen Agent (ADK), streams progress
    |-- /api/runs          => returns recent run history (Dashboard)
    |
    +--> .mitra/<session_id>/
            data/
                data.csv           (raw upload, copied as-is)
                mini_data.csv      (pandas describe(include="all") stats, Drop columns the user explicitly listed in their description (parse for "exclude X" or "ignore column Y" intent), chunk-sampled)
            reports/
                validation_report.json
                metadata.json
```

The `.env` file is read at server startup. A smoke-test LLM call is made before
any agent is allowed to run. If the smoke test fails the backend returns an error
on the first `/api/validate` or `/api/metadata` request.

---

## 4. TECHNOLOGY STACK

| Layer          | Technology                                      |
|----------------|-------------------------------------------------|
| Frontend       | ReactJS + Vite; design tokens from theme.css    |
| Backend        | Python 3.10+, FastAPI, uvicorn                  |
| Agent runtime  | Google ADK (Agent Development Kit)              |
| LLM routing    | LiteLLM (every agent routes through this)       |
| LLM providers  | OpenAI, Gemini, Anthropic (user-supplied key)   |
| Data processing| pandas (chunk-based), json, jsonschema          |
| Config         | config.ini (one file, multiple sections)        |
| Credentials    | .env (never surfaced in the GUI)                |

---

## 5. .ENV CONFIGURATION

The `.env` file lives at the repo root and is sourced before launching the backend.
Users create it from `.env.example`. It is added to `.gitignore`.

```ini
# .env  (DO NOT COMMIT — added to .gitignore)
LLM_TYPE=anthropic          # one of: openai | gemini | anthropic
LLM_API_KEY=sk-ant-...      # API key from your provider account
LLM_GATEWAY_URL=            # optional: LiteLLM proxy / company gateway URL
                            # leave blank to use the provider default endpoint
```

### IMPORTANT NOTE: The GUI should hide the API key once user enters it. Use masking. It should not be visible anywhere in the DOM or console or UI.

### LLM smoke test

On FastAPI startup (`lifespan` handler), send a minimal test prompt to the
configured LLM endpoint. If the call fails or returns an HTTP error, log a
prominent warning. Any subsequent agent call that requires the LLM returns
`HTTP 503` with body `{"error": "LLM_SMOKE_TEST_FAILED"}` until the server is
restarted with a valid `.env`.

```
Smoke-test prompt: "Reply with the single word OK."
Success condition: HTTP 200 and response body contains "OK" (case-insensitive).
```

---

## 6. GLOBAL CONFIG (config.ini)

Single `config.ini` at the repo root. Sections are added per epic.
Epic 1 owns the `[python]`, `[paths]`, `[pipeline]`, and `[upload]` sections.

```ini
[python]
PYTHON=/path/to/venv/bin/python

[paths]
WORKSPACE_ROOT=.mitra
SESSION_LOG_DIR=.mitra/logs

[upload]
MAX_FILE_SIZE_MB=2000

ALLOWED_EXTENSIONS=.csv,.xls,.xlsx,.zip
MINI_DATA_SAMPLE_ROWS=1000
CHUNK_SIZE_ROWS=50000

[pipeline]
TRAIN_TEST_SPLIT=0.8
MAX_ML_MODELS=10
MAX_HPT_TRIALS=5
```

All backend code reads these values via a `ConfigLoader` class (reuse
`model_library/core/config_loader.py` if interface matches, else write a thin
wrapper that calls `configparser`).

---

## 7. FRONTEND — APP SHELL

### 7.1 Layout

```
+--248px sidebar--+--------------- main (flex col) -----------------------+
| MITRA AI logo   | TopBar (sticky, h=64px, blur backdrop)                |
| WORKSPACE label |                                                        |
| Dashboard       | <page content>                                         |
| New Run         |                                                        |
| Pipeline        |                                                        |
| Leaderboard     |                                                        |
| [spacer]        |                                                        |
| Settings        |                                                        |
| Course Team     |                                                        |
+-----------------+--------------------------------------------------------+
```

Grid: `248px 1fr`. Sidebar bg: `#fff`, border-right: `1px solid var(--line)`.
Design tokens are defined in `wireframes/MITRA AI/mitra/theme.css` — implement
them as CSS custom properties in the React app's global stylesheet.

### 7.2 Sidebar nav items

| Key         | Label       | Icon   | Badge condition                  |
|-------------|-------------|--------|----------------------------------|
| dashboard   | Dashboard   | grid   | none                             |
| upload      | New Run     | upload | none                             |
| pipeline    | Pipeline    | flow   | spinner if run state = running   |
| leaderboard | Leaderboard | trophy | none                             |

Active item: `background: var(--accent-soft); color: var(--accent-ink)`.
Settings item is rendered below the spacer but its route is out of Epic 1 scope.

### 7.3 TopBar

Height 64 px, sticky top, `rgba(255,255,255,0.8)` + `backdrop-filter: blur(10px)`.
Shows page icon, title, subtitle (from `ROUTE_META` map). Right slot: shows
"Run in progress" spinner button when a run is active and user is not on pipeline
page.

### 7.4 Routing

Client-side route state held in React `useState`. No URL router needed for v1.
Routes: `dashboard | upload | pipeline | leaderboard`.

---

## 8. FRONTEND — DASHBOARD PAGE

### 8.1 Hero card

Gradient background (`linear-gradient(120deg, #fff 0%, #faf9ff 55%, #f3efff 100%)`),
accent radial glow (right side). Contains:
- H1 headline: "A team of agents, one optimized model."
- Muted body paragraph (product description).
- Two CTAs: "Start a new run" (primary) => navigates to upload; "View last
  leaderboard" (secondary) => navigates to leaderboard.

### 8.2 Stat tiles (4-column grid)

| Label           | Icon   | Accent |
|-----------------|--------|--------|
| Total runs      | layers | no     |
| Models trained  | cpu    | yes    |
| Best accuracy   | target | yes    |
| Avg run time    | gauge  | no     |

Data is fetched from `GET /api/runs/stats`. In v1, the backend can return
hardcoded/mock values. Schema: `{total_runs, models_trained, best_accuracy, avg_run_time_min}`.

### 8.3 Recent runs table

Columns: Run ID (mono), Dataset, Task (tag), Best model, Accuracy (right-aligned
mono), Drift, arrow icon. Clicking a row navigates to leaderboard.
Data from `GET /api/runs?limit=5`.

Drift states: `stable` (green dot), `watch` (amber dot), `—` (grey dot).

### 8.4 Right column

- Accuracy trend card: sparkline SVG, best accuracy value, "+N pts last 6 runs".
- Agent roster card: list of 8 agents, each with avatar (colored monogram),
  name, role, type tag, and green live dot.

Agent roster is static configuration read from `config/agents.json` (see
Section 10.2).

---

## 9. FRONTEND — NEW RUN PAGE (Page 1)

### 9.1 Layout

Two-column grid: `1.35fr 1fr` gap `18px`.

Left column: dropzone card + fixture dataset picker.
Right column: metadata form card.

Below both columns (full width): validation report card (hidden until
"Validate & Review" is clicked).

### 9.2 Dropzone

Dashed border (`2px dashed var(--accent-line)`), centred upload icon, label
"Drop a dataset to begin", sub-label "CSV or image .zip - up to 200 MB -
processed locally", "Browse files" secondary button.

Accepted MIME / extensions: `.csv`, `.xls`, `.xlsx`, `.zip`.
Max size: read from `config.ini [upload] MAX_FILE_SIZE_MB`.
On drop or browse: validate extension and size client-side; show inline error if
invalid before uploading.

### 9.3 Fixture dataset picker

Section header: "OR PICK A FIXTURE" (mono, faint, uppercase).
List of sample datasets; each row is a selectable card with: doc icon, filename
(mono), row/col/size metadata, task type tag, checkmark when selected.
Fixture data is hardcoded in the frontend (no API call):
- `iris.csv` — 150 rows, 5 cols, 4.5 KB, Classification
- `housing.csv` — 20,640 rows, 9 cols, 1.4 MB, Regression
- `cats-dogs-10.zip` — 2,000 rows, img cols, 18.2 MB, Image / CNN

Selecting a fixture populates the metadata form with sensible defaults and resets
validation state to idle.

### 9.4 Metadata form (right card)

Header: "Metadata", sub: "Minimal hints — agents infer the rest into metadata.json".

| Field           | Widget          | Validation                          | Notes                            |
|-----------------|-----------------|-------------------------------------|----------------------------------|
| Problem type    | Segmented ctrl  | required                            | Auto-detect / Classify / Regress / Cluster |
| Target column   | text input      | optional for unsupervised           | hint: "leave blank for unsupervised" |
| Description     | textarea (4 rows)| min 20 chars, required             | hint: ">= 20 chars - guides feature & model agents" |
| Data type       | Segmented ctrl  | required                            | CSV / Excel / Image (routes backend preprocessing) |
| Provider (BYOM) | Segmented ctrl  | required                            | Anthropic / OpenAI / Gemini      |
| API Key         | password input  | required                            | placeholder varies by provider   |
| Gateway URL     | url input       | optional                            | placeholder: "Where is your model running?" |

"Validate & Review" primary button (full width): disabled while validating.
On click: POST to `/api/upload` (if a file is staged but not yet uploaded),
then POST to `/api/validate`.
While validating: button label = "Validating your data..." + spinner icon.

### 9.5 Validation report card (shown after clicking Validate)

Appears below the two-column section with fade-up animation.
Header row: Data Validator agent avatar (DV, hue 14), name, artifact filename
(`validation_report.json`), status pill (Passed / Validating...).

Check grid (2-column, checks reveal one by one with 340 ms interval):

| Key       | Label                    | Detail on pass                          | Warn condition                     |
|-----------|--------------------------|-----------------------------------------|------------------------------------|
| format    | File format & encoding   | utf-8, comma-delimited, N columns       | unsupported encoding               |
| rows      | Row count                | N rows, above minimum (10)              | fewer than 10 rows                 |
| nulls     | Null density             | N columns exceed 80% threshold          | any column > 80% null              |
| variance  | Zero-variance scan       | No constant columns detected            | any column has zero variance       |
| pii       | PII heuristic            | No PII-suspect column names             | column names match PII patterns    |
| target    | Target separability      | target_col, N balanced classes          | mild class overlap / imbalance     |

Footer row: summary text + "Run pipeline" primary button (disabled until all
checks pass, or pass+warn — no blocker checks failed).

---

## 10. BACKEND API

### 10.1 Endpoints

| Method | Path                   | Description                                                |
|--------|------------------------|------------------------------------------------------------|
| POST   | /api/upload            | Accept multipart file; save to `.mitra/<sid>/data/data.csv`; generate mini_data.csv; return session_id |
| POST   | /api/validate          | Body: `{session_id}`. Run DataValidator; stream SSE progress; write validation_report.json |
| POST   | /api/metadata          | Body: `{session_id, description, target_col, problem_type}`. Run MetadataGenAgent; stream SSE; write metadata.json |
| GET    | /api/runs              | Return list of recent run summaries from `.mitra/` session dirs |
| GET    | /api/runs/stats        | Return aggregate stats: total_runs, models_trained, best_accuracy, avg_run_time_min |
| GET    | /api/health            | LLM smoke-test status + server uptime                      |

SSE event format (for /api/validate and /api/metadata):
```
data: {"type": "check", "key": "format", "status": "pass", "detail": "..."}
data: {"type": "check", "key": "rows", "status": "warn", "detail": "..."}
data: {"type": "done", "artifact": "validation_report.json"}
data: {"type": "error", "message": "..."}
```

### 10.2 Session workspace layout

```
.mitra/
  <session_id>/          # UUID4, created by /api/upload
    data/
      data.csv           # raw uploaded file (never modified after save)
      mini_data.csv      # pandas describe() on a 1000-row sample (chunked read)
    reports/
      validation_report.json
      metadata.json
  logs/
    <session_id>.log
```

`session_id` is returned by `/api/upload` and echoed back in all subsequent calls.
`mkdir -p` is used for all directories.

### 10.3 mini_data.csv generation

Generated during `/api/upload` immediately after saving `data.csv`.
Uses chunked reading (chunk size from `config.ini [upload] CHUNK_SIZE_ROWS`).
Sample at most `MINI_DATA_SAMPLE_ROWS` rows, then run `pandas.describe(include="all")`.
Write the transposed describe output to `mini_data.csv`.
This file is the ONLY data the Metadata Gen Agent is allowed to read from disk.

---

## 11. DATA VALIDATOR (Python, deterministic)

Class: `DataValidator` in `epic_1/backend/validator.py`.
Called by the `/api/validate` endpoint. No LLM involved.
Yields `ValidationCheckResult` objects one by one (streamed as SSE).

### 11.1 Checks (in execution order)

```python
CHECK_ORDER = ["format", "rows", "nulls", "variance", "pii", "target"]
```

| Check     | Logic                                                                          | Pass   | Warn                        | Fail (blocker)     |
|-----------|--------------------------------------------------------------------------------|--------|-----------------------------|--------------------|
| format    | Sniff delimiter and encoding of data.csv; verify columns parseable             | utf-8 + delimiter detected | unusual encoding | binary / unparseable |
| rows      | Count rows via chunk iteration; compare to MIN_ROWS from config.ini           | >= min | none                        | < min rows         |
| nulls     | Per column: null_count / total_rows; compare to NULL_THRESHOLD (0.8)          | 0 cols exceed threshold | none            | any col > threshold |
| variance  | Per numeric column: check std != 0 after chunk aggregation                    | no zero-variance cols | none            | any constant col   |
| pii       | Column name regex match against PII_PATTERNS from config.ini (json array)     | no matches | names match patterns  | none (warn only)   |
| target    | If target_col provided: count unique values; check class balance               | balanced | mild imbalance          | target col missing |

Blockers (format/rows/nulls/variance/target-missing) set `passed=False` on the
report. Warn-only checks (pii, mild imbalance) do not block the pipeline.

### 11.2 Output: validation_report.json

```json
{
  "session_id": "uuid4",
  "passed": true,
  "blocker_count": 0,
  "warn_count": 1,
  "checks": [
    {
      "key": "format",
      "label": "File format & encoding",
      "status": "pass",
      "detail": "utf-8, comma-delimited, 5 columns"
    },
    {
      "key": "target",
      "label": "Target separability",
      "status": "warn",
      "detail": "species - 3 balanced classes",
      "warn_message": "Mild class overlap on sepal width"
    }
  ]
}
```

---

## 12. METADATA GENERATION AGENT (LLM via Google ADK)

Class: `MetadataGenAgent` in `epic_1/backend/agents/metadata_gen_agent.py`.
Invoked by `/api/metadata` after validation passes.
Routes all LLM calls through LiteLLM (never direct provider SDK calls).

### 12.1 Inputs

- `mini_data.csv` from the session workspace (statistical summary only).
- User-provided `description` (free-text, minimum 20 chars).
- User-provided `target_col` (may be empty string for unsupervised).
- User-provided `problem_type` hint (may be "auto").
- Optional: user-uploaded metadata file path (if provided during upload).

**Hard guardrail in system prompt:** "You must NOT read `data.csv`. You have
access ONLY to `mini_data.csv`. Do not call any tool that reads the full dataset."

### 12.2 System prompt (agents/prompts/metadata_gen.md)

The system prompt instructs the agent to:
1. Read `mini_data.csv` using pandas describe(include="all").
2. Infer column types: if a column has <= N unique values relative to row count,
   classify as `categorical`; else `numeric`. N is configurable.
3. Determine `problem_type`:
   - If user hint is not "auto", use it directly.
   - Else: if target column has <= CLASSIFICATION_UNIQUE_THRESHOLD unique values
     (relative to row count), classify as `classification`; else `regression`.
     If no target column, classify as `unsupervised`.
4. Extract input columns = all columns except the target column.
5. Drop columns the user explicitly listed in their description (parse for
   "exclude X" or "ignore column Y" intent). Figure out PII columns and call it out in the metadata.json
6. Produce `metadata.json` strictly conforming to the JSON Schema in Section 12.4.

### 12.3 Agent tools

The agent is given exactly two tools:
- `read_mini_data(session_id: str) -> str` — reads mini_data.csv, returns
  its content as a string.
- `write_metadata(session_id: str, metadata: dict) -> None` — validates
  the dict against the JSON Schema and writes metadata.json. Raises
  `ValidationError` if schema is violated (agent must fix and retry).

No other file read or shell access is granted.

### 12.4 metadata.json schema (fixed — all agents downstream read this)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": [
    "session_id", "problem_type", "target_col", "target_col_type",
    "input_cols", "cols_to_drop", "statistics"
  ],
  "properties": {
    "session_id": { "type": "string" },
    "problem_type": { "type": "string", "enum": ["classification", "regression", "unsupervised"] },
    "target_col": { "type": ["string", "null"] },
    "target_col_type": { "type": ["string", "null"], "enum": ["categorical", "numeric", null] },
    "input_cols": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "col_type"],
        "properties": {
          "name": { "type": "string" },
          "col_type": { "type": "string", "enum": ["categorical", "numeric"] }
        }
      }
    },
    "cols_to_drop": {
      "type": "array",
      "items": { "type": "string" }
    },
    "statistics": {
      "type": "object",
      "description": "Per-column stats from pandas describe(). Keys are column names.",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "count": { "type": "number" },
          "mean": { "type": ["number", "null"] },
          "std": { "type": ["number", "null"] },
          "min": { "type": ["number", "null"] },
          "25%": { "type": ["number", "null"] },
          "50%": { "type": ["number", "null"] },
          "75%": { "type": ["number", "null"] },
          "max": { "type": ["number", "null"] },
          "top": { "type": ["string", "null"] },
          "freq": { "type": ["number", "null"] }
        }
      }
    }
  }
}
```

The schema is stored at `epic_1/backend/schemas/metadata_schema.json` and is
loaded by both the agent's `write_metadata` tool and any downstream agent that
reads `metadata.json`.

---

## 13. CONFIG CONTROLLABLES (config.ini additions for Epic 1)

```ini
[upload]
MAX_FILE_SIZE_MB=200
ALLOWED_EXTENSIONS=.csv,.xls,.xlsx,.zip
MINI_DATA_SAMPLE_ROWS=1000
CHUNK_SIZE_ROWS=50000
MIN_ROWS=10
NULL_THRESHOLD=0.8
# JSON array of regex patterns for PII column name detection
PII_PATTERNS=["(?i)aadhaar","(?i)pan_","(?i)mobile","(?i)phone","(?i)email","(?i)ssn","(?i)passport"]

[metadata_agent]
CLASSIFICATION_UNIQUE_THRESHOLD=0.05
# if unique_count / row_count <= this, column treated as categorical
CATEGORICAL_UNIQUE_RATIO=0.05
LLM_MAX_RETRIES=3
```

---

## 14. DEVELOPMENT OUTPUTS

```
epic_1/
  SPEC.md                                     (this file)
  task.txt
  backend/
    main.py                                   (FastAPI app, lifespan smoke-test)
    config_loader.py                          (thin wrapper around configparser)
    session.py                                (session workspace creation, UUID)
    mini_data.py                              (chunked mini_data.csv generator)
    validator.py                              (DataValidator class)
    agents/
      metadata_gen_agent.py                   (MetadataGenAgent, ADK-based)
      tools.py                                (read_mini_data, write_metadata tools)
      prompts/
        metadata_gen.md                       (system prompt for the agent)
    schemas/
      metadata_schema.json                    (JSON Schema for metadata.json)
      validation_report_schema.json           (JSON Schema for validation_report.json)
    routers/
      upload.py                               (/api/upload endpoint)
      validate.py                             (/api/validate SSE endpoint)
      metadata.py                             (/api/metadata SSE endpoint)
      runs.py                                 (/api/runs, /api/runs/stats)
      health.py                               (/api/health)
  frontend/
    src/
      App.jsx                                 (app shell, routing, run state)
      components/
        Sidebar.jsx
        TopBar.jsx
        AgentAvatar.jsx
        StatusPill.jsx
        Segmented.jsx
        Sparkline.jsx
        HBars.jsx
        Stat.jsx
        FormField.jsx
        ByomFields.jsx
      screens/
        Dashboard.jsx
        UploadScreen.jsx
      theme.css                               (design tokens, per wireframe)
      icons.jsx                               (SVG icon set)
      data.js                                 (AGENTS roster, ROUTE_META)
  config/
    agents.json                               (8 agent definitions: id, name, short, hue, type, role, owner)
.env.example                                  (template; .env in .gitignore)
config.ini                                    (global config, owned here; extended by later epics)
requirements.txt                              (fastapi, uvicorn, google-adk, litellm, pandas, jsonschema, python-dotenv)
```

---

## 15. ACCEPTANCE CRITERIA

1. Server starts and prints "LLM smoke-test: OK" (or fails fast with a clear error)
   when `.env` is correctly configured.
2. Uploading `iris.csv` via the dropzone creates a session workspace and `mini_data.csv`
   within 2 seconds (chunk-based, no full in-memory load).
3. Clicking "Validate & Review" streams all 6 checks to the UI, each revealing with
   the animated 340 ms interval. Final status pill reads "Passed - ready to run".
4. On a dataset with a null-heavy column (> 80%), the null-density check shows
   status "fail" and the "Run pipeline" button remains disabled.
5. Metadata Gen Agent produces a `metadata.json` that validates against
   `metadata_schema.json` without errors.
6. `metadata.json` correctly identifies `species` as the target column with
   `col_type: categorical` and `problem_type: classification` for `iris.csv`.
7. The agent never reads `data.csv` directly (enforced by tool access restriction).
8. All config values (thresholds, limits) are read from `config.ini`; none are
   hardcoded in Python or JSX.
9. Re-running validation for the same session_id overwrites `validation_report.json`
   cleanly (no duplicate session directories).
10. Dashboard renders recent runs table and agent roster without errors when the
    `.mitra/` workspace has at least one completed session.

---

## 16. OPEN ITEMS

- Image dataset (ZIP) support in the validator: detecting folder-per-class structure
  and generating image-count mini_data is non-trivial. Deferred to a follow-up
  task within Epic 1 if time allows; current scope targets CSV/XLS.
- The `agents.json` roster currently lists 8 agents mapped to 8 team members.
  Ownership assignment finalization is pending confirmation from the team.
- Google ADK version pinning: confirm the exact version that supports the
  agent-to-agent protocol before locking `requirements.txt`.
