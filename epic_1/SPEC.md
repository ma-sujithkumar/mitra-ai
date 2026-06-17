# SPEC: Epic 1 - Upload, Validate & Metadata Generation (Page 1)

**Assignees:** Vidhi Kant Gupta, Sebin Francis Kannampuzha
**Tech Stack:** ReactJS (frontend), Python/FastAPI (backend), Google ADK (agent), LiteLLM (LLM routing)

---

## 1. GOAL

Implement Page 1 of MITRA, the entry point where users upload a tabular dataset,
provide minimal metadata hints, configure or confirm their LLM credentials, and
trigger an automated two-phase validation + metadata extraction before the
pipeline begins.

Constraints:
1. User needs to upload a CSV or Excel file with the data.
2. Need a browse file option from frontend
3. Need validations of file format in both frontend and backend
4. Once upload starts, create a `.mitra` session workspace and save the uploaded
   file locally. Excel files are converted to canonical `data.csv` while keeping
   the original source file for audit/debugging.
5. ZIP/image support is visible as deferred, but not accepted in Epic 1.
6. The validation and metadata generation section in new run page should only appear after clicking on Validate and review button , move LLM settings of configuring model , provider , api gateway , api key  
  in a more compact way and i want this section not to be part of new run page but of settings page, new run page should only have a reference of what model , provider is selected for the particular run
7. Validate and review should be enabled only after mandatory fields - file uploaded , LLM smoke test successful are provided

The two artifacts produced by Epic 1 flow into all downstream epics:
- `validation_report.json` — deterministic data quality gate.
- `metadata.json` — structured contract read by every subsequent agent.

---

## 2. SCOPE 

### In scope for Epic 1
- Global app shell: sidebar navigation, top bar, routing between pages (Dashboard,
  New Run, Pipeline, Leaderboard).
- Dashboard page (overview cards, recent runs table, agent roster, accuracy trend).
- New Run page (Page 1): file upload, latest uploaded dataset picker, metadata
  form, optional metadata file upload, validation split, BYOM credentials, and
  model selection.
- Pipeline and Leaderboard pages implemented as polished frontend prototype
  screens using the Claude design handoff. Their real backend orchestration,
  training, SHAP, and leaderboard persistence are out of scope for Epic 1.
- Hybrid BYOM configuration: `.env` defaults plus optional per-run UI override.
- `.env` configuration schema and LLM smoke-test status on startup.
- `config.ini` — single global config file for all controllable parameters.
  It should contain defaults for all necessary configs if user does not specify.
- Backend API (Python/FastAPI) for file upload and session workspace management.
- Data Validator (Python, deterministic, no LLM): 6 checks, produces
  `validation_report.json`.
- Metadata Generation Agent (LLM via Google ADK): reads mini-data statistics + the user input for data description + system level prompt on what needs to be achieved,
  produces `metadata.json` against a fixed JSON schema.
- Session workspace: `.mitra/<session_id>/` directory layout.
- Session-level `run_config.json` containing user pipeline preferences such as
  validation split. This file is separate from `metadata.json`.

### Out of scope (handled in later epics)
- Page 2 backend: preprocessing pipeline (encoding, scaling, feature selection,
  model selection, training on Ray).
- Page 3 backend: real leaderboard, SHAP report generation and persistence.
- HPT loop, judge agent, drift monitor.
- ZIP/image validator support.
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
    |-- /api/validate      => starts Data Validator job
    |-- /api/validate/events => streams validation progress via SSE
    |-- /api/metadata      => starts Metadata Gen Agent (ADK) job
    |-- /api/metadata/events => streams metadata progress via SSE
    |-- /api/uploads/recent => returns latest uploaded datasets
    |-- /api/runs          => returns recent run history (Dashboard)
    |
    +--> .mitra/<session_id>/
            data/
                data.csv           (canonical CSV used by all agents)
                source.<ext>       (only for Excel uploads; original file)
                mini_data.csv      (pandas describe(include="all") stats, chunk-sampled)
                user_metadata.<ext> (optional user-uploaded metadata file)
            reports/
                validation_report.json
                metadata.json
                run_config.json
            session.json           (session_id, timestamp, original filename)
```

The `.env` file is read at server startup. A smoke-test LLM call is attempted
when `.env` has enough credentials. Startup never hard-fails because a user can
provide BYOM credentials per run from the frontend. `/api/health` reports `.env`
LLM readiness. Metadata generation uses per-run credentials first; if they are
absent, it falls back to `.env`. If neither credential path passes a smoke test,
metadata generation returns `HTTP 503`.

---

## 4. TECHNOLOGY STACK

| Layer          | Technology                                      |
|----------------|-------------------------------------------------|
| Frontend       | ReactJS + Vite; design tokens from theme.css    |
| Backend        | Python 3.10+, FastAPI, uvicorn                  |
| Agent runtime  | Google ADK (Agent Development Kit)              |
| LLM routing    | LiteLLM (every agent routes through this)       |
| LLM providers  | OpenAI, Gemini, Anthropic (.env or per-run BYOM) |
| Data processing| pandas (chunk-based), json, jsonschema          |
| Config         | config.ini (one file, multiple sections)        |
| Credentials    | Hybrid: `.env` defaults + optional per-run BYOM |

---

## 5. .ENV CONFIGURATION

The `.env` file lives at the repo root and is sourced before launching the backend.
Users create it from `.env.example`. It is added to `.gitignore`.

```ini
# .env  (DO NOT COMMIT — added to .gitignore)
LLM_TYPE=anthropic          # one of: openai | gemini | anthropic
LLM_API_KEY=sk-ant-...      # API key from your provider account
LLM_MODEL=                  # optional; blank uses config.ini provider base model
LLM_GATEWAY_URL=            # optional: LiteLLM proxy / company gateway URL
                            # leave blank to use the provider default endpoint
```

### IMPORTANT NOTE: The GUI should mask the API key. There is no reveal toggle.
The key is never logged, never stored in local/session storage, never written to
`.mitra`, and never echoed back from the backend. It is held only long enough to
start the metadata job. The frontend should use an uncontrolled password input
and must not mirror the key into React state, page text, data attributes, console
logs, or request/response logs.

### Hybrid BYOM behavior

1. Backend reads `.env` at startup and reports the smoke-test result through
   `/api/health`.
2. The New Run page always shows a BYOM section with provider, optional model,
   API key, and optional gateway URL.
3. If the user supplies a per-run API key, that key overrides `.env` for the
   metadata job only.
4. If the user leaves the per-run API key blank, backend uses `.env` credentials.
5. If the model field is blank, backend uses the provider base model from
   `config.ini [llm_models]`.
6. For custom gateway/hosted models, the UI displays a note that model names must
   follow LiteLLM provider-style routing for the configured `LLM_TYPE`, for
   example `openai/<model>`, `anthropic/<model>`, or `gemini/<model>`.

### LLM smoke test

On FastAPI startup (`lifespan` handler), send a minimal test prompt to the
configured `.env` LLM endpoint if enough `.env` values exist. If the call fails
or returns an HTTP error, log a prominent warning and report the failed state via
`/api/health`; the server still starts. Per-run BYOM credentials are smoke-tested
when metadata generation starts. If both the per-run credentials and `.env`
fallback are unavailable or fail, `/api/metadata` returns `HTTP 503` with body
`{"error": "LLM_SMOKE_TEST_FAILED"}`.

```
Smoke-test prompt: "Reply with the single word OK."
Success condition: HTTP 200 and response body contains "OK" (case-insensitive).
```

---

## 6. GLOBAL CONFIG (config.ini)

Single `config.ini` at the repo root. Sections are added per epic.
Epic 1 owns the `[python]`, `[paths]`, `[pipeline]`, `[upload]`,
`[llm_models]`, and `[metadata_agent]` sections.

```ini
[python]
PYTHON=/path/to/venv/bin/python

[paths]
WORKSPACE_ROOT=.mitra
SESSION_LOG_DIR=.mitra/logs

[upload]
MAX_FILE_SIZE_MB=200
ALLOWED_EXTENSIONS=.csv,.xls,.xlsx
MINI_DATA_SAMPLE_ROWS=1000
CHUNK_SIZE_ROWS=50000
RECENT_UPLOAD_LIMIT=5

[pipeline]
TRAIN_TEST_SPLIT=0.8
MAX_ML_MODELS=10
MAX_HPT_TRIALS=5

[llm_models]
OPENAI_BASE_MODEL=openai/gpt-5.1
ANTHROPIC_BASE_MODEL=anthropic/claude-sonnet-4-5-20250929
GEMINI_BASE_MODEL=gemini/gemini-3-pro
```

All backend code reads these values via a root app `ConfigLoader` class. The
existing `model_library/core/config_loader.py` is model-library-specific, so Epic
1 should reuse its style and validation approach but implement a thin
`configparser` wrapper for the root app.

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
Settings item is rendered below the spacer. Epic 1 may include a lightweight
frontend-only Settings page that shows LLM/provider defaults and health status,
but persistent settings management is out of scope.

### 7.3 TopBar

Height 64 px, sticky top, `rgba(255,255,255,0.8)` + `backdrop-filter: blur(10px)`.
Shows page icon, title, subtitle (from `ROUTE_META` map). Right slot: shows
"Run in progress" spinner button when a run is active and user is not on pipeline
page.

### 7.4 Routing

Client-side route state held in React `useState`. No URL router needed for v1.
Routes: `dashboard | upload | pipeline | leaderboard | settings`.

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

Data is fetched from `GET /api/runs/stats`. In Epic 1, values are derived from
`.mitra` where available; model metrics can remain mocked until later epics write
real leaderboard artifacts. Schema: `{total_runs, models_trained, best_accuracy, avg_run_time_min}`.

### 8.3 Recent runs table

Columns: Run ID (mono), Dataset, Task (tag), Best model, Accuracy (right-aligned
mono), Drift, arrow icon. Clicking a row navigates to leaderboard.
Data from `GET /api/runs?limit=5`.

Drift states: `stable` (green dot), `watch` (amber dot), `—` (grey dot).

### 8.4 Right column

- Accuracy trend card: sparkline SVG, best accuracy value, "+N pts last 6 runs".
- Agent roster card: list of 8 agents, each with avatar (colored monogram),
  name, role, type tag, and green live dot.

Agent roster is static configuration read from `config/agents.json`.

---

## 9. FRONTEND — NEW RUN PAGE (Page 1)

### 9.1 Layout

Two-column grid: `1.35fr 1fr` gap `18px`.

Left column: dropzone card + latest uploaded dataset picker.
Right column: metadata form card.

Below both columns (full width): validation report card and metadata generation
status card (hidden until "Validate & Review" is clicked).

### 9.2 Dropzone

Dashed border (`2px dashed var(--accent-line)`), centred upload icon, label
"Drop a dataset to begin", sub-label "CSV or Excel - up to 200 MB - processed
locally", "Browse files" secondary button.

Accepted MIME / extensions: `.csv`, `.xls`, `.xlsx`.
Max size: read from `config.ini [upload] MAX_FILE_SIZE_MB`.
On drop or browse: validate extension and size client-side; show inline error if
invalid before uploading.
ZIP/image upload is shown as a deferred capability and is not accepted by the
file picker in Epic 1.

### 9.3 Latest uploaded dataset picker

Section header: "RECENT UPLOADS" (mono, faint, uppercase).
Data is fetched from `GET /api/uploads/recent?limit=5`.
List only sessions that already have `.mitra/<session_id>/data/data.csv`.
Each row is a selectable card with: doc icon, original filename (mono), row/col/
size metadata when available, task type tag when known, upload timestamp, and
checkmark when selected.

Selecting a recent upload reuses the existing `session_id` and resets validation
state to idle. It does not copy data into a new session. If no recent uploads
exist, show an empty state and require the user to browse/drop a new file.

### 9.4 Metadata form (right card)

Header: "Metadata", sub: "Minimal hints — agents infer the rest into metadata.json".

| Field           | Widget          | Validation                          | Notes                            |
|-----------------|-----------------|-------------------------------------|----------------------------------|
| Problem type    | Segmented ctrl  | required                            | Auto-detect / Classify / Regress / Cluster |
| Target column   | text input      | optional for unsupervised           | hint: "leave blank for unsupervised" |
| Validation split| number input    | optional, 0.05 to 0.5               | saved to `run_config.json`; default from `config.ini` |
| Description     | textarea (4 rows)| min 20 chars, required             | hint: ">= 20 chars - guides feature & model agents" |
| Data type       | Segmented ctrl  | required                            | CSV / Excel; Image is shown as deferred |
| Metadata file   | file input      | optional                            | `.csv` or `.json`; combines with description for metadata context |
| Provider (BYOM) | Segmented ctrl  | required                            | Anthropic / OpenAI / Gemini      |
| Model           | text input      | optional                            | blank uses provider base model from `config.ini` |
| API Key         | password input  | required only if `.env` is not valid| no reveal toggle, never persisted |
| Gateway URL     | url input       | optional                            | placeholder: "Where is your model running?" |

"Validate & Review" primary button (full width): disabled while validating.
On click: POST to `/api/upload` (if a file is staged but not yet uploaded),
or reuse the selected recent `session_id`, then POST to `/api/validate`.
While validating: button label = "Validating your data..." + spinner icon.
If validation passes with no blocker checks, frontend automatically starts
metadata generation with `POST /api/metadata` and streams metadata events. The
"Run pipeline" button remains disabled until `metadata.json` is generated and
schema-valid.

The validation split is written to `.mitra/<session_id>/reports/run_config.json`
and is not added to `metadata.json`.

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

After validation completes, show a dataset summary below the checks: row count,
column count, file size, inferred data type, null-heavy column count, target
column status, and optional metadata file status.

Footer row: summary text + "Run pipeline" primary button (disabled until all
checks pass or pass+warn with no blocker checks failed, and metadata generation
has completed successfully).

---

## 10. BACKEND API

### 10.1 Endpoints

| Method | Path                   | Description                                                |
|--------|------------------------|------------------------------------------------------------|
| POST   | /api/upload            | Accept multipart dataset file and optional metadata file; create session; normalize to `.mitra/<sid>/data/data.csv`; generate mini_data.csv; return session_id |
| GET    | /api/uploads/recent    | Return latest uploaded datasets from `.mitra`, default `limit=5` |
| POST   | /api/validate          | Body: `{session_id, target_col, validation_split}`. Start DataValidator job; write/update `run_config.json`; return accepted status |
| GET    | /api/validate/events   | Query: `session_id`. Stream validator SSE events; write validation_report.json |
| POST   | /api/metadata          | Body: `{session_id, description, target_col, problem_type, provider, model, api_key, gateway_url}`. Start MetadataGenAgent job |
| GET    | /api/metadata/events   | Query: `session_id`. Stream metadata agent SSE events; write metadata.json |
| GET    | /api/runs              | Return list of recent run summaries from `.mitra/` session dirs |
| GET    | /api/runs/stats        | Return aggregate stats: total_runs, models_trained, best_accuracy, avg_run_time_min |
| GET    | /api/health            | LLM smoke-test status + server uptime                      |
| GET    | /api/config/public     | Return non-secret frontend config: upload limits, allowed extensions, provider defaults, base model names |

SSE event format (for `/api/validate/events` and `/api/metadata/events`):
```
data: {"type": "check", "key": "format", "status": "pass", "detail": "..."}
data: {"type": "check", "key": "rows", "status": "warn", "detail": "..."}
data: {"type": "done", "artifact": "validation_report.json"}
data: {"type": "error", "message": "..."}
```

The frontend uses `POST` to start jobs and browser `EventSource` over `GET` to
stream events. Epic 1 supports one active validation job and one active metadata
job per `session_id`.

### 10.2 Session workspace layout

```
.mitra/
  <session_id>/          # YYYYMMDD_HHMMSS_dataset_slug_uuid8
    session.json         # uploaded_at, original filename, normalized filename, summary
    data/
      data.csv           # canonical CSV for all downstream agents
      source.xlsx        # original Excel file, if uploaded
      source.xls         # original Excel file, if uploaded
      user_metadata.csv  # optional metadata file, if uploaded
      user_metadata.json # optional metadata file, if uploaded
      mini_data.csv      # pandas describe() on a 1000-row sample (chunked read)
    reports/
      validation_report.json
      metadata.json
      run_config.json
  logs/
    <session_id>.log
```

`session_id` is returned by `/api/upload` and echoed back in all subsequent calls.
Session IDs include timestamp, sanitized dataset slug, and an 8-character UUID
suffix, for example `20260613_142530_iris_a1b2c3d4`.
`mkdir -p` is used for all directories.

### 10.3 mini_data.csv generation

Generated during `/api/upload` immediately after saving canonical `data.csv`.
For CSV uploads, copy the source into canonical `data.csv`. For Excel uploads,
preserve the original as `source.xlsx` or `source.xls`, convert the first sheet
to canonical `data.csv`, and use `data.csv` for all downstream work.

Uses chunked reading (chunk size from `config.ini [upload] CHUNK_SIZE_ROWS`) for
CSV-based profiling.
Sample at most `MINI_DATA_SAMPLE_ROWS` rows, then run `pandas.describe(include="all")`.
Write the transposed describe output to `mini_data.csv`.
This file is the ONLY data the Metadata Gen Agent is allowed to read from disk.

---

## 11. DATA VALIDATOR (Python, deterministic)

Class: `DataValidator` in `backend/validator.py`.
Started by the `/api/validate` endpoint and streamed by
`/api/validate/events`. No LLM involved.
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
  "session_id": "20260613_142530_iris_a1b2c3d4",
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

Class: `MetadataGenAgent` in `backend/agents/metadata_gen_agent.py`.
Invoked by `/api/metadata` after validation passes, and streamed by
`/api/metadata/events`.

The implementation uses Google ADK `LlmAgent` with ADK's `LiteLlm` connector.
All LLM calls route through LiteLLM (never direct provider SDK calls).

### 12.1 Inputs

- `mini_data.csv` from the session workspace (statistical summary only).
- User-provided `description` (free-text, minimum 20 chars).
- User-provided `target_col` (may be empty string for unsupervised).
- User-provided `problem_type` hint (may be "auto").
- Optional: user-uploaded metadata file content (`user_metadata.csv` or
  `user_metadata.json`) if provided during upload. The backend reads this file
  and passes bounded text/context to the agent; the agent is not given a generic
  file-read tool.
- LLM settings resolved from per-run BYOM fields first, then `.env`: provider,
  model, API key, and optional gateway URL. Blank model uses
  `config.ini [llm_models]` provider base model.

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

No other file read, full dataset read, shell access, or direct provider SDK
access is granted.

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

The schema is stored at `backend/schemas/metadata_schema.json` and is
loaded by both the agent's `write_metadata` tool and any downstream agent that
reads `metadata.json`.

---

## 13. CONFIG CONTROLLABLES (config.ini additions for Epic 1)

```ini
[upload]
MAX_FILE_SIZE_MB=200
ALLOWED_EXTENSIONS=.csv,.xls,.xlsx
MINI_DATA_SAMPLE_ROWS=1000
CHUNK_SIZE_ROWS=50000
RECENT_UPLOAD_LIMIT=5
MIN_ROWS=10
NULL_THRESHOLD=0.8
# JSON array of regex patterns for PII column name detection
PII_PATTERNS=["(?i)aadhaar","(?i)pan_","(?i)mobile","(?i)phone","(?i)email","(?i)ssn","(?i)passport"]

[pipeline]
TRAIN_TEST_SPLIT=0.8
MAX_ML_MODELS=10
MAX_HPT_TRIALS=5

[llm_models]
OPENAI_BASE_MODEL=openai/gpt-5.1
ANTHROPIC_BASE_MODEL=anthropic/claude-sonnet-4-5-20250929
GEMINI_BASE_MODEL=gemini/gemini-3-pro

[metadata_agent]
CLASSIFICATION_UNIQUE_THRESHOLD=0.05
# if unique_count / row_count <= this, column treated as categorical
CATEGORICAL_UNIQUE_RATIO=0.05
LLM_MAX_RETRIES=3
```

---

## 14. DEVELOPMENT OUTPUTS

```
backend/
  main.py                                     (FastAPI app, lifespan smoke-test)
  config_loader.py                            (thin wrapper around configparser)
  session.py                                  (session workspace creation, timestamped IDs)
  mini_data.py                                (CSV/Excel normalization + chunked mini_data.csv generator)
  validator.py                                (DataValidator class)
  jobs.py                                     (in-memory validation/metadata job event registry)
  agents/
    metadata_gen_agent.py                     (MetadataGenAgent, ADK + LiteLlm)
    tools.py                                  (read_mini_data, write_metadata tools)
    prompts/
      metadata_gen.md                         (system prompt for the agent)
  schemas/
    metadata_schema.json                      (JSON Schema for metadata.json)
    validation_report_schema.json             (JSON Schema for validation_report.json)
  routers/
    upload.py                                 (/api/upload, /api/uploads/recent)
    validate.py                               (/api/validate, /api/validate/events)
    metadata.py                               (/api/metadata, /api/metadata/events)
    runs.py                                   (/api/runs, /api/runs/stats)
    health.py                                 (/api/health)
    config.py                                 (/api/config/public)
frontend/
  package.json                                (Vite + React)
  vite.config.js
  src/
    App.jsx                                   (app shell, routing, run state)
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
      PipelineScreen.jsx                      (frontend prototype, mocked data)
      LeaderboardScreen.jsx                   (frontend prototype, mocked data)
      Settings.jsx                            (lightweight health/config view)
    theme.css                                 (design tokens, per handoff)
    icons.jsx                                 (SVG icon set)
    data.js                                   (AGENTS roster, ROUTE_META, mocked Page 2/3 data)
config/
  agents.json                                 (8 agent definitions: id, name, short, hue, type, role, owner)
epic_1/
  SPEC.md                                     (this file)
  task.txt
.env.example                                  (template; .env in .gitignore)
config.ini                                    (global config, owned here; extended by later epics)
requirements.txt                              (fastapi, uvicorn, google-adk, litellm, pandas, openpyxl, xlrd, jsonschema, python-dotenv)
```

---

## 15. ACCEPTANCE CRITERIA

1. Server starts even when `.env` is missing or invalid, logs a prominent warning,
   and reports LLM readiness from `/api/health`. When `.env` is correctly
   configured, health shows the smoke-test as OK.
2. Uploading a CSV via the dropzone creates a timestamped session workspace
   (`YYYYMMDD_HHMMSS_dataset_slug_uuid8`) and `mini_data.csv` within 2 seconds
   for small files (chunk-based, no full in-memory load for CSV profiling).
3. Uploading `.xlsx` or `.xls` preserves the original as `source.xlsx` or
   `source.xls`, converts the first sheet to canonical `data.csv`, and generates
   `mini_data.csv` from the canonical CSV.
4. The latest uploads picker shows only the latest 5 sessions that contain
   `.mitra/<session_id>/data/data.csv`; selecting one reuses its session_id.
5. Optional user metadata file upload accepts `.csv` or `.json`, stores it under
   `.mitra/<session_id>/data/`, and includes bounded content in metadata agent
   context while the description remains mandatory.
6. Clicking "Validate & Review" starts validation with `POST /api/validate`,
   streams all 6 checks through `GET /api/validate/events`, and reveals checks in
   the UI. Final status pill reads "Passed - ready to generate metadata" when no
   blocker checks fail.
7. On a dataset with a null-heavy column (> 80%), the null-density check shows
   status "fail" and the "Run pipeline" button remains disabled.
8. After validation passes, frontend automatically starts metadata generation
   with `POST /api/metadata` and streams events through
   `GET /api/metadata/events`.
9. Metadata Gen Agent produces a `metadata.json` that validates against
   `metadata_schema.json` without errors.
10. `metadata.json` correctly identifies `species` as the target column with
   `col_type: categorical` and `problem_type: classification` for `iris.csv`.
11. The agent never reads `data.csv` directly (enforced by tool access restriction).
12. Per-run BYOM credentials override `.env`, are never persisted, and the API
    key field has no reveal toggle.
13. Blank model input uses provider base model from `config.ini [llm_models]`.
14. Validation split is saved to `reports/run_config.json` and is not added to
    `metadata.json`.
15. All config values (thresholds, limits, allowed extensions, recent upload
    limit, and default model names) are read from `config.ini`; none are
    hardcoded in Python or JSX.
16. Re-running validation for the same session_id overwrites `validation_report.json`
   cleanly (no duplicate session directories).
17. Dashboard renders recent runs table and agent roster without errors when the
    `.mitra/` workspace has at least one completed session.
18. Pipeline and Leaderboard frontend routes render the Claude handoff design as
    mocked/prototype pages without requiring real Epic 2/3 backend work.

---

## 16. OPEN ITEMS

- Image dataset (ZIP) support in the validator: detecting folder-per-class structure
  and generating image-count mini_data is non-trivial. Deferred to a follow-up
  task; current scope targets CSV/XLS/XLSX only.
- The `agents.json` roster currently lists 8 agents mapped to 8 team members.
  Ownership assignment finalization is pending confirmation from the team.
- Google ADK and LiteLLM version pinning: confirm/install versions that support
  ADK `LlmAgent` with the `LiteLlm` model connector before locking
  `requirements.txt`.
- Pipeline and Leaderboard backend APIs are intentionally deferred beyond Epic 1.
