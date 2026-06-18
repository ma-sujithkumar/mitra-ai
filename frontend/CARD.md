---
name: frontend
path: frontend
purpose: React/Vite client application providing a modern, interactive dashboard for MITRA AutoML pipeline configuration, execution monitoring, and model evaluation.
interfaces:
  inputs:
    - name: API Responses
      format: JSON
      upstream: backend / routers
      description: Returns configuration details, validation reports, metadata status, and training run progress.
    - name: Event stream
      format: text/event-stream (SSE)
      upstream: backend / routers (runs)
      description: Live log updates and execution progress percentage from active pipeline agents.
  outputs:
    - name: user upload requests
      format: multipart/form-data
      downstream: backend (POST /api/upload)
      description: Uploads tabular dataset (CSV/XLSX) and metadata overrides.
    - name: session run commands
      format: JSON REST requests
      downstream: backend (POST /api/validate, POST /api/metadata, POST /api/runs)
      description: Triggers sequential stages of validation, metadata analysis, model selection, and training.
entry_points:
  - name: App.jsx
    type: React Root Component
    description: Manages routing and application layout state (Sidebar, Topbar, active Screen).
  - name: screens/UploadScreen.jsx
    type: React Component
    description: Initial file upload screen handling validation reports and metadata editing.
  - name: screens/PipelineScreen.jsx
    type: React Component
    description: Renders real-time agent execution cards using Server-Sent Events (SSE) events.
  - name: screens/LeaderboardScreen.jsx
    type: React Component
    description: Leaderboard visualizer displaying performance metrics, training status, and SHAP feature importances.
dependencies:
  - react
  - vite
  - tailwindcss (optionally configured in UI templates)
  - lucide-react (icons)
---

# Technical Architecture: Frontend

## Overview
The `frontend` submodule is a Single Page Application (SPA) built on React and Vite. It provides an agentic pipeline dashboard allowing users to:
1. Upload and preview datasets.
2. Interactively inspect validation warnings/blockers.
3. Review and refine LLM-generated metadata (e.g. target column selection, column types, dropped columns).
4. Run/monitor the multi-agent execution pipeline in real-time.
5. Inspect leaderboard performance (e.g. F1, RMSE, residuals) and explainability plots (SHAP).

## Core Component Walkthrough
1. **`api/client.js`**: Reusable axios-like wrapper handles calling backend REST endpoints `/api/upload`, `/api/validate`, `/api/metadata`, and `/api/runs`.
2. **`api/events.js`**: Sets up `EventSource` to listen to `/api/session/{sid}/events`, broadcasting logs and stage progress to updates state.
3. **`screens/UploadScreen.jsx`**: Handles file selection, drag-and-drop, displays tabular preview, shows validation rules results, and lets users change target columns/data types.
4. **`screens/PipelineScreen.jsx`**: Displays list of agents (DataValidator, MetadataGen, FeatureSelection, ModelSelection, etc.) and their live status, progress bar, and logs via SSE.
5. **`screens/LeaderboardScreen.jsx`**: Displays a table of all trained models, their status (completed, failed, Acceptance), metric breakdowns, and a SHAP feature importance horizontal bar chart.

## Interfacing Guide
- **Upstream Integration:** Interfaced directly with user mouse/keyboard actions.
- **Downstream Integration:** Performs REST calls to backend FastAPI server running on port `8001` or proxy-configured port. Emits requests with CORS credentials.

## Suggested Cleanup/Refactoring
- **State Management:** Currently relies on standard React state variables in `App.jsx` passed down to children. Refactoring to a unified context hook (`SessionContext`) will avoid prop drilling.
- **SSE Fallback:** Add automatic reconnection logic inside `api/events.js` if the connection drops.
