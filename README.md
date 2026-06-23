<a id="readme-top"></a>

<div align="center">

<img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/9be4d344-6782-461a-b5a6-32a07bf7b34e" width="600" alt="animated hello">

# <img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="35" /> MITRA <img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="35" />

### An agent-driven, self-hosted AutoML platform

Upload a dataset, describe the target, and watch a pipeline of Google ADK
agents handle validation, feature engineering, model selection,
dataset2Vec-warm-started training, SHAP/overfitting/HPT evaluation, and an
LLM judge loop end-to-end — in the browser or headless via `--cli`.

[Report Bug](../../issues) ·
[Request Feature](../../issues)

</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#architecture">Architecture</a>
      <ul>
        <li><a href="#pipeline-stages">Pipeline stages</a></li>
        <li><a href="#repository-layout">Repository layout</a></li>
        <li><a href="#runtime-workspace">Runtime workspace</a></li>
      </ul>
    </li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#configuration">Configuration</a>
      <ul>
        <li><a href="#configini-environment--paths"><code>config.ini</code> — environment &amp; paths</a></li>
        <li><a href="#env-llm-credentials"><code>.env</code> — LLM credentials</a></li>
        <li><a href="#advanced-pipeline-settings-ui">Advanced pipeline settings (UI)</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#run-the-full-app-backend--frontend">Run the full app (backend + frontend)</a></li>
        <li><a href="#run-backend-or-frontend-only">Run backend or frontend only</a></li>
        <li><a href="#run-headless-via---cli">Run headless via <code>--cli</code></a></li>
      </ul>
    </li>
    <li><a href="#api-surface">API Surface</a></li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#team">Team</a></li>
  </ol>
</details>

---

## About The Project

MITRA turns a raw dataset into a deployed, explained model with no manual
ML-engineering steps in between. A single FastAPI backend orchestrates a DAG
of agents — each one independently invocable as its own CLI — and a React/Vite
frontend streams every stage live over SSE. The same DAG runs headless for
batch/CI use via `backend/orchestration/run_pipeline.py --cli`.

Design goals:

- **One config surface.** `config.ini` holds env/paths/python only; every
  tunable pipeline parameter is in one place and surfaced in the UI's
  Advanced Settings panel.
- **No duplicated agents.** Each ML concern (feature engineering, model
  selection, training, SHAP, overfitting, HPT, judge, dataset2Vec) is a single
  module under `backend/agents/`, reused by both the API and the CLI.
- **Google ADK only.** Every LLM call goes through the shared `LiteLlm`
  wrapper — no other LLM client is used anywhere in the codebase.
- **Read-only codebase at runtime.** Every artifact a run produces — uploads,
  reports, evaluation output, plots, token counts — is written under
  `.mitra/<user_id>/<session_id>/`, never into the source tree.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Architecture

```
                upload dataset + (optional) description
                                 |
                                 v
 ┌──────────────────────────── FastAPI (backend/) ─────────────────────────────┐
 │                                                                              │
 │  validate ──▶ metadata ──▶ feature engineering ──▶ model selection          │
 │  (Epic 1)     (LLM agent)   (Epic 2, ADK)            (Epic 3 + dataset2Vec  │
 │                                                        warm-start)          │
 │                                  │                                          │
 │                                  v                                          │
 │                          parallel training (Ray / local)                    │
 │                                  │                                          │
 │                                  v                                          │
 │              ┌──────────── parallel evaluation ────────────┐               │
 │              │   SHAP   ||   overfitting   ||   HPT/Optuna  │               │
 │              └──────────────────────────────────────────────┘               │
 │                                  │                                          │
 │                                  v                                          │
 │                    judge (LLM) ──▶ feedback loop (≤ max_turns)             │
 │                                  │                                          │
 │                                  v                                          │
 │           plots/<stage>/*.png  +  dataset2Vec DB write-back                │
 │                                                                              │
 │  every stage emits SSE events on a single unified event bus                │
 └──────────────────────────────────────────────────────────────────────────┬─┘
                                                                              │
                                  SSE + REST                                 │
                                                                              v
                                                          React / Vite (frontend/)
                                              Upload → Live Training → Leaderboard
```

The exact same stage sequence is run by `PipelineRunner` in
`backend/orchestration/run_pipeline.py` for `--cli` mode and by
`TrainingService` for the browser flow — neither re-implements the other;
the browser flow calls the same agent classes.

### Pipeline stages

| # | Stage | Module | Output |
|---|-------|--------|--------|
| 1 | Metadata generation | `backend/agents/metadata_gen_agent.py` | `reports/metadata.json` |
| 2 | Feature engineering | `backend/agents/feature_engineering/` | `data/engineered_dataset.csv` |
| 3 | Model selection (dataset2Vec warm-start) | `backend/agents/model_selection/`, `backend/agents/dataset2vec/` | `reports/model_config.json` |
| 4 | Parallel training | `backend/agents/training/`, `backend/agents/training_orchestrator/` | `reports/training_summary.json` |
| 5 | Parallel evaluation | `backend/agents/evaluation/{shap,overfitting,hpt}/` | `evaluation/{shap,overfitting,hpt}/...` |
| 6 | Judge + feedback loop | `backend/agents/evaluation/judge/`, `backend/orchestration/judge_loop.py` | `reports/judge_decision.json` |
| 7 | Visualizations | `backend/orchestration/plotting.py` | `plots/<stage>/*.png` |
| 8 | dataset2Vec write-back | `backend/orchestration/d2v_bridge.py` | `DB/*.parquet`, `DB/index.faiss` |

### Repository layout

```
mitra/
├── bin/                    mitra (launcher), setup.sh
├── config.ini              env / paths / python interpreter (only)
├── requirements.txt        unified backend + agent dependencies
├── DB/                      dataset2Vec encoder + corpus + leaderboard DB
├── model_library/           shared MLKit model registry
├── backend/
│   ├── main.py               FastAPI app factory (create_app)
│   ├── config_loader.py      single ConfigLoader for all of config.ini
│   ├── session.py            session workspace resolution (.mitra/<id>/)
│   ├── routers/               upload, validate, metadata, training,
│   │                          evaluation, config, runs, health, llm
│   ├── services/               TrainingService (browser flow orchestration)
│   ├── orchestration/          run_pipeline (--cli), eval_runner, judge_loop,
│   │                           d2v_bridge, plotting, token_counter, events
│   └── agents/
│       ├── feature_engineering/
│       ├── model_selection/
│       ├── dataset2vec/
│       ├── training/, training_orchestrator/, ray_wrapper/
│       └── evaluation/{shap,overfitting,hpt,judge}/
├── frontend/                React / Vite UI
│   └── src/{screens,components,api}/
├── docs/<module>/           preserved specs, plans, design notes
└── .mitra/<user_id>/<session_id>/   runtime workspace (gitignored)
```

### Runtime workspace

Every run is fully isolated under `WORKSPACE_ROOT` (default `.mitra/`,
configurable in `config.ini`):

```
.mitra/<user_id>/<session_id>/
├── data/                 uploaded + engineered + train/test CSVs
├── reports/              metadata.json, model_config.json,
│                         training_summary.json, judge_decision.json
├── evaluation/           shap/, overfitting/, hpt/ per-model artifacts
├── plots/                <stage>/*.png on-demand visualizations
├── token_usage.json      per-agent LLM token accounting
└── config_overrides.json per-session Advanced Settings overrides
```

The codebase itself is never written to at runtime — only this directory.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="50" title="Python" />
  <img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="50" title="React" />
  <img src="https://user-images.githubusercontent.com/74038190/212257463-4d082cb4-7483-4eaf-bc25-6dde2628aabd.gif" width="50" title="JavaScript" />
  <img src="https://user-images.githubusercontent.com/74038190/212281763-e6ecd7ef-c4aa-45b6-a97c-f33f6bb592bd.gif" width="50" title="HTML5" />
  <img src="https://user-images.githubusercontent.com/74038190/212281775-b468df30-4edc-4bf8-a4ee-f52e1aaddc86.gif" width="50" title="CSS3" />
  <img src="https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif" width="50" title="Git" />
  <img src="https://user-images.githubusercontent.com/74038190/212281756-450d3ffa-9335-4b98-a965-db8a18fee927.gif" width="50" title="Markdown" />
</div>

<br />

* [![FastAPI][fastapi-shield]][fastapi-url]
* [![React][react-shield]][react-url]
* [![Vite][vite-shield]][vite-url]
* Google ADK (`google-adk[extensions]`) + LiteLLM — the only LLM client
* Ray — distributed/parallel training and evaluation
* scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch — `model_library/` registry
* SHAP, Optuna — explainability and hyperparameter tuning
* FAISS — dataset2Vec nearest-neighbour warm-start

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

* **Python 3.12+** with a virtual environment (the venv does **not** need to
  live inside the repo — see [Configuration](#configini-environment--paths)).
* **Node.js 18+** and npm, for the frontend.
* An API key for at least one LLM provider (OpenAI, Anthropic, or Gemini).
* **Windows users:** the run commands below are given for **PowerShell**.
  `bin/setup.sh` is a Bash script, so installing via it requires **Git Bash**
  (ships with Git for Windows) or WSL; otherwise use the manual `pip` / `npm`
  install shown for Windows below.

### Installation

#### Linux / macOS

1. Clone the repo and create a virtual environment (anywhere on disk):
   ```sh
   git clone git@github.com:Deeplearning1227/deeplearning-repo.git mitra
   cd mitra
   python3 -m venv ~/venv
   ```
2. Point `config.ini` at that interpreter — this is the **one** thing the
   launcher needs to find your environment (details below):
   ```ini
   [python]
   PYTHON=~/venv/bin/python
   ```
3. Run the setup script — it installs backend (`pip`) and frontend (`npm`)
   dependencies using the interpreter from step 2:
   ```sh
   bin/setup.sh
   ```
4. Copy the LLM credentials template and fill in a provider key:
   ```sh
   cp .env.example .env
   # edit .env: LLM_TYPE=anthropic / openai / gemini, LLM_API_KEY=...
   ```
5. Start the app:
   ```sh
   bin/mitra up
   ```

#### Windows (PowerShell)

1. Clone the repo and create a virtual environment (anywhere on disk):
   ```powershell
   git clone git@github.com:Deeplearning1227/deeplearning-repo.git mitra
   cd mitra
   python -m venv C:\venvs\mitra
   ```
2. Point `config.ini` at that interpreter using a **native Windows path with
   backslashes** (the launcher does not understand Git-Bash-style `/d/...`
   paths — see [Configuration](#configini-environment--paths)):
   ```ini
   [python]
   PYTHON=C:\venvs\mitra\Scripts\python.exe
   ```
3. Install dependencies. Either run the setup script from **Git Bash**
   (`bin/setup.sh`), or install manually from PowerShell:
   ```powershell
   & "C:\venvs\mitra\Scripts\python.exe" -m pip install -r requirements.txt
   cd frontend; npm install; cd ..
   ```
4. Copy the LLM credentials template and fill in a provider key:
   ```powershell
   Copy-Item .env.example .env
   # edit .env: LLM_TYPE=anthropic / openai / gemini, LLM_API_KEY=...
   ```
5. Start the app — see [Usage](#run-the-full-app-backend--frontend) for the
   two-terminal Windows flow (the bundled `bin/mitra up` launcher does not yet
   start the frontend on Windows).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration

MITRA deliberately has **two** configuration files, each with one job. Do not
add a third.

### `config.ini` — environment & paths

Location: repo root. This is the **only** file the launcher and backend read
for environment wiring (interpreter, workspace paths, upload limits, and
default pipeline parameters). It is plain `ConfigParser` INI, read once by
`backend/config_loader.py::ConfigLoader`.

> [!IMPORTANT]
> `[python] PYTHON=` is how every entrypoint (`bin/mitra`, `bin/setup.sh`,
> the CLI) finds your Python. Get this wrong and nothing else works — see
> [Troubleshooting](#troubleshooting).

```ini
[python]
# Absolute or ~-relative path to the interpreter, OR a bare command on PATH.
# Leave blank to let the launcher auto-detect: <repo>/.venv, <repo>/venv,
# then "python3"/"python" on PATH (in that order).
PYTHON=~/venv/bin/python

[paths]
# Everything generated at runtime lives here. Never inside the repo tree.
WORKSPACE_ROOT=.mitra
SESSION_LOG_DIR=.mitra/logs

[upload]
MAX_FILE_SIZE_MB=200
ALLOWED_EXTENSIONS=.csv,.xls,.xlsx
...

[pipeline]
TRAIN_TEST_SPLIT=0.8
MAX_ML_MODELS=10
MAX_HPT_TRIALS=5
RUN_POST_TRAINING_EVAL=true   # run SHAP+overfitting+HPT+judge after training
MAX_JUDGE_TURNS=3             # judge <-> model-selection feedback loop length

[training_api]
DEFAULT_EXECUTION_MODE=ray    # "ray" or "local"
MAX_CONCURRENT_RUNS=2

[hpt]
OVERFITTING_GAP_THRESHOLD=0.10
VAL_SPLIT_RATIO=0.2
OPTUNA_SEED=42
```

Resolution rules that matter when configuring `[python] PYTHON`:

| Value | Resolved as |
|---|---|
| *(blank)* | `<repo>/.venv/bin/python` → `<repo>/venv/bin/python` → `python3`/`python` on `PATH` |
| `~/venv/bin/python` | Expanded against `$HOME`, then checked for existence |
| `/abs/path/to/python` | Used directly |
| `relative/path` | Resolved relative to the repo root |
| `some-command` (not a path) | Passed straight to the OS as a command on `PATH` |

> [!IMPORTANT]
> **On Windows**, `[python] PYTHON` must be a **native Windows path** (e.g.
> `C:\venvs\mitra\Scripts\python.exe`) or a bare command on `PATH` (e.g.
> `python`). The launcher resolves the venv interpreter at
> `<venv>\Scripts\python.exe`. Git-Bash-style paths such as
> `/d/conda/envs/mitra311/python.exe` are **not** understood by the Node
> launcher and cause the backend to fail with `ENOENT` — use `D:\conda\...`
> instead.

`config.ini` never contains secrets, absolute machine-specific defaults baked
into code, or LLM API keys — those live in `.env`.

### `.env` — LLM credentials

Location: repo root, copied from `.env.example` (gitignored). This is the
**only** place LLM secrets are configured for backend startup defaults:

```env
LLM_TYPE=anthropic
LLM_API_KEY=sk-...
LLM_MODEL=
LLM_GATEWAY_URL=
LLM_CA_BUNDLE=
```

Per-run, the UI's **Settings → Run Configuration** panel lets a user "bring
your own model" (provider/model/key/gateway) which overrides `.env` for that
run only, after a required connection smoke-test.

### Advanced pipeline settings (UI)

Every tunable in `[pipeline]`, `[training_api]`, and `[hpt]` is also surfaced
in **Settings → Advanced Settings** in the browser, backed by:

```
GET /api/config/advanced?session_id=<id>     # effective value: override > config.ini default
PUT /api/config/advanced?session_id=<id>     # validates type + range, persists override
```

Saved overrides are written to `.mitra/<user_id>/<session_id>/config_overrides.json`
and are read by the pipeline at invoke time — they apply to that session
only and never mutate `config.ini`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

All commands below assume you are in the repo root. `bin/mitra` resolves its
own location, so it can also be invoked with a full path from anywhere.

### Run the full app (backend + frontend)

* Backend: `http://127.0.0.1:8000` (override with `MITRA_HOST`/`MITRA_PORT`)
* Frontend: `http://127.0.0.1:5173` (Vite picks the next free port if busy)

Open the frontend URL, upload a dataset on the first screen, and the Live
Training page takes over automatically.

**Linux / macOS** — one command starts both; `Ctrl-C` stops both cleanly:

```sh
bin/mitra up
```

**Windows (PowerShell)** — the bundled launcher cannot spawn `npm` on Windows
(a Node `spawn EINVAL` limitation, see [Troubleshooting](#troubleshooting)),
so start the two services in **separate terminals**:

```powershell
# Terminal 1 - backend (uses the interpreter from config.ini [python] PYTHON)
& "C:\venvs\mitra\Scripts\python.exe" -m uvicorn backend.main:create_app --factory --host 127.0.0.1 --port 8000

# Terminal 2 - frontend
cd frontend; npm run dev
```

### Run backend or frontend only

**Linux / macOS:**

```sh
bin/mitra backend     # uvicorn only, same host/port rules as above
bin/mitra frontend    # vite dev server only
```

**Windows (PowerShell):**

```powershell
# Backend only
& "C:\venvs\mitra\Scripts\python.exe" -m uvicorn backend.main:create_app --factory --host 127.0.0.1 --port 8000

# Frontend only
cd frontend; npm run dev
```

### Run headless via `--cli`

Runs the identical agent DAG without a browser — useful for batch jobs, CI,
or scripted experiments.

**Linux / macOS:**

```sh
bin/mitra cli -- \
  --dataset path/to/train.csv \
  --target  target_column \
  --session-id my_run_001 \
  --provider anthropic \
  --model    claude-sonnet-4-6 \
  --mode     local \
  --max-models 10 \
  -v
```

Equivalent direct form:

```sh
"$PYTHON" -m backend.orchestration.run_pipeline --dataset train.csv --target label
```

**Windows (PowerShell)** — invoke the pipeline module directly with your
configured interpreter:

```powershell
& "C:\venvs\mitra\Scripts\python.exe" -m backend.orchestration.run_pipeline `
  --dataset path\to\train.csv `
  --target target_column `
  --session-id my_run_001 `
  --provider anthropic `
  --model claude-sonnet-4-6 `
  --mode local `
  --max-models 10 `
  -v
```

Artifacts land in `.mitra/<user_id>/<session_id>/` exactly as described in
[Runtime workspace](#runtime-workspace); the command printed to
`pipeline_command.txt` in that directory reproduces the run.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## API Surface

| Concern | Endpoints |
|---|---|
| Upload / validate / metadata | `POST /api/upload`, `POST /api/validate`(+`/events`), `POST /api/metadata`(+`/events`) |
| Training | `POST /api/training/start`, `GET /api/training/status/{id}`, `GET /api/training/events` (SSE) |
| Leaderboard / evaluation | `GET /api/runs/{id}/leaderboard`, `/verdict`, `/shap`, `/tokens` |
| Visualizations | `GET /api/runs/{id}/plots`, `GET /api/runs/{id}/plots/{path}` |
| Configuration | `GET /api/config/public`, `GET`/`PUT /api/config/advanced` |
| LLM | `POST /api/llm/smoke-test` |
| Runs / health | `GET /api/runs`, `GET /api/runs/stats`, `GET /api/health` |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Testing

**Linux / macOS:**

```sh
# Backend (pytest)
"$PYTHON" -m pytest backend/tests -q

# Frontend (node:test + vite build)
cd frontend
npm test
npm run build
```

**Windows (PowerShell):**

```powershell
# Backend (pytest)
& "C:\venvs\mitra\Scripts\python.exe" -m pytest backend/tests -q

# Frontend (node:test + vite build)
cd frontend
npm test
npm run build
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Troubleshooting

**`No module named uvicorn` when running `bin/mitra up`**

The launcher fell back to a system Python that doesn't have the project's
dependencies installed. Fix `[python] PYTHON=` in `config.ini` to point at
the interpreter where you ran `bin/setup.sh` / `pip install -r requirements.txt`,
then re-run `bin/mitra up`. Confirm the interpreter directly:

```sh
"$PYTHON" -c "import uvicorn; print(uvicorn.__version__)"
```

**Leaderboard stays empty after training finishes**

Check `[pipeline] RUN_POST_TRAINING_EVAL=true` in `config.ini` (or the
session's Advanced Settings override) — the browser flow only runs
SHAP/overfitting/HPT/judge after training when this is enabled.

**"Test connection" fails in Settings**

Confirm `.env` has a valid `LLM_API_KEY` for the selected `LLM_TYPE`, or that
the per-run BYOM fields in the UI are filled in and smoke-tested before
starting a run.

**(Windows) `bin/mitra up` crashes with `Error: spawn EINVAL`**

The Node launcher cannot spawn `npm.cmd` on Windows (a Node security
restriction on spawning `.cmd`/`.bat` files without a shell). Start the
backend and frontend in two separate terminals instead — see
[Run the full app](#run-the-full-app-backend--frontend). Confirm the
interpreter on Windows:

```powershell
& "C:\venvs\mitra\Scripts\python.exe" -c "import uvicorn; print(uvicorn.__version__)"
```

**(Windows) Backend fails with `ENOENT` / interpreter not found**

`[python] PYTHON` in `config.ini` is set to a Git-Bash-style path such as
`/d/conda/envs/mitra311/python.exe`. The Node launcher needs a native Windows
path — change it to `D:\conda\envs\mitra311\python.exe` (backslashes), or to a
bare `python` that resolves on `PATH`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Backend-first restructure (no `epic_*` folders, ADK-only LLM client)
- [x] Hybrid orchestrator: parallel SHAP/overfitting/HPT + judge feedback loop
- [x] dataset2Vec warm-start + corpus write-back
- [x] Live leaderboard, verdict, SHAP, and on-demand plot API + UI wiring
- [x] Advanced settings panel + portable launcher (`bin/mitra`, `bin/setup.sh`)
- [ ] Resume-from-any-agent in the live pipeline view

See [open issues](../../issues) for the full list.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Team

1. Sujithkumar M A — Texas Instruments
2. Avinash Bhargav — Siemens
3. Shiva Priya — Bosch
4. Meena M — Bosch
5. Sebin Francis — Cisco
6. Onkar Shamsunder Biyani — SMILe
7. Subhasis Mahana — Samsung
8. Vidhi Kant Gupta — NPCI

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[fastapi-shield]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[fastapi-url]: https://fastapi.tiangolo.com/
[react-shield]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[react-url]: https://react.dev/
[vite-shield]: https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white
[vite-url]: https://vitejs.dev/
