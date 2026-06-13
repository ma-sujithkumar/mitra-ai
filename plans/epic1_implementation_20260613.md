# Epic 1 Implementation Plan — MITRA AI
**Date:** 2026-06-13  
**Branch:** epic1  
**Assignees:** Vidhi Kant Gupta, Sebin Francis Kannampuzha

---

## Goal

Implement Page 1 of MITRA AI: file upload, deterministic data validation (6 checks, no LLM), and LLM-driven metadata generation (Google ADK + LiteLLM). Produces `validation_report.json` and `metadata.json` consumed by all downstream epics.

---

## Monorepo Layout

```
deeplearning-repo/
  mitra-ui/            Vite + React frontend
  mitra-backend/       FastAPI backend
  model_library/       existing ML library (unchanged)
  config.ini           global config (all epics share this)
  .env.example         LLM credentials template
  epic_1/config/agents.json   8 agent definitions
```

---

## Phase 1: Global Config

- `config.ini` at repo root — sections: python, paths, upload, metadata_agent, pipeline
- `.env.example` — LLM_TYPE, LLM_API_KEY, LLM_GATEWAY_URL
- Add `.env`, `.mitra/`, `mitra-ui/node_modules/`, `mitra-ui/dist/` to `.gitignore`

---

## Phase 2: Backend (mitra-backend/)

### Files
| File | Purpose |
|------|---------|
| `main.py` | FastAPI app + lifespan LLM smoke-test |
| `config_loader.py` | configparser wrapper (NOT the YAML one from model_library) |
| `session.py` | SessionManager: UUID4, mkdir -p workspace |
| `mini_data.py` | MiniDataGenerator: chunked pandas describe() |
| `validator.py` | DataValidator: 6 checks, yields ValidationCheckResult |
| `agents/metadata_gen_agent.py` | MetadataGenAgent (Google ADK + LiteLLM) |
| `agents/tools.py` | read_mini_data(), write_metadata() ADK tools |
| `agents/prompts/metadata_gen.md` | System prompt (hard guardrail: only mini_data.csv) |
| `schemas/metadata_schema.json` | Fixed JSON Schema for metadata.json |
| `schemas/validation_report_schema.json` | JSON Schema for validation_report.json |
| `routers/upload.py` | POST /api/upload |
| `routers/validate.py` | POST /api/validate (SSE) |
| `routers/metadata.py` | POST /api/metadata (SSE) |
| `routers/runs.py` | GET /api/runs, GET /api/runs/stats |
| `routers/health.py` | GET /api/health |

### Key design decisions
- `config_loader.py`: uses `configparser`, reads `../config.ini` (repo root)
- Validation checks order: `["format", "rows", "nulls", "variance", "pii", "target"]`
- SSE format: `data: {"type":"check","key":"...","status":"...","detail":"..."}\n\n`
- LLM smoke-test on startup; returns 503 if failed until server restart

---

## Phase 3: Frontend (mitra-ui/)

### Scaffold
```bash
npm create vite@latest mitra-ui -- --template react
```

### Source structure
```
mitra-ui/src/
  App.jsx               app shell, route state (useState, no router lib)
  main.jsx
  theme.css             adapted from epic_1/mitra-ai/project/mitra/theme.css
  icons.jsx             adapted from prototype icons.jsx
  data.js               AGENTS, ROUTE_META, SAMPLE_DATASETS
  api.js                uploadFile(), streamValidation(), streamMetadata(), getRuns(), getRunStats()
  components/           Sidebar, TopBar, AgentAvatar, StatusPill, Segmented, Sparkline, HBars, Stat, FormField, ByomFields
  screens/              Dashboard.jsx, UploadScreen.jsx
```

### Prototype adaptation
All prototype files at `epic_1/mitra-ai/project/mitra/` are CDN-based JSX. Convert to ES modules:
- Remove `window.X = X` global assignments
- Add `import React, { useState, useEffect, useRef } from 'react'`
- Add named/default exports
- Replace mock `validate()` with real API calls from `api.js`
- Add real `<input type="file">` handler in UploadScreen

---

## Phase 4: Integration

- Backend CORS: allow `localhost:5173` (Vite dev) and `localhost:4173` (Vite preview)
- Start backend: `cd mitra-backend && uvicorn main:app --reload --port 8000`
- Start frontend: `cd mitra-ui && npm run dev`

---

## Verification (10 acceptance criteria from SPEC)

1. Backend starts with "LLM smoke-test: OK"
2. Upload iris.csv creates session workspace within 2 seconds
3. All 6 checks stream to UI with 340ms animation
4. Null-heavy CSV blocks "Run pipeline" button
5. metadata.json validates against metadata_schema.json
6. iris.csv => species = categorical, problem_type = classification
7. Agent never reads data.csv (tool access restriction)
8. All thresholds come from config.ini (no hardcoding)
9. Re-running same session_id overwrites cleanly
10. Dashboard shows recent runs and agent roster
