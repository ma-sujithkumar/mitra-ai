# Plan to Commit and Push Changes in Phases

This plan outlines the structure of 9 distinct commits to stage and push all current tracked and untracked modifications to the dev branch, keeping them organized by logical subsystem.

## Commit Phase Schedule

### Commit 1: Global Configuration and System Initialization
- **Files**:
  - `config.ini`
  - `backend/config_loader.py`
  - `backend/main.py`
- **Message**: `config: update python path, add D2V_DB_DIR and feature engineering API settings`

### Commit 2: Evaluation Agents Refactoring and Fixes
- **Files**:
  - `backend/agents/evaluation/hpt/agent.py`
  - `backend/agents/evaluation/hpt/config_loader.py`
  - `backend/agents/evaluation/judge/judge_agent.py`
  - `backend/agents/evaluation/shap/runner.py`
- **Message**: `agents: update HPT trial history, judge agent runner, and SHAP schema validation`

### Commit 3: Feature Engineering Backend Logic Refactoring
- **Files**:
  - `backend/agents/feature_engineering/config.py`
  - `backend/agents/feature_engineering/orchestrator.py`
- **Message**: `feature_engineering: refactor orchestrator to resolve LLM settings natively`

### Commit 4: Feature Engineering Router and Status Service
- **Files**:
  - `backend/routers/feature_engineering.py`
  - `backend/services/feature_status.py`
- **Message**: `backend: add feature engineering router and status reader service`

### Commit 5: Pipeline Preparation and Evaluation Router Extensions
- **Files**:
  - `backend/services/pipeline_prep.py`
  - `backend/routers/evaluation.py`
- **Message**: `backend: update pipeline prep with D2V query and extend evaluation router endpoints`

### Commit 6: Frontend Core Config, Styling, and API Integration
- **Files**:
  - `frontend/src/api/client.js`
  - `frontend/src/data.js`
  - `frontend/src/theme.css`
  - `frontend/src/App.jsx`
- **Message**: `frontend: integrate base API clients, styling, and main App routes`

### Commit 7: Existing Frontend Screens Updates
- **Files**:
  - `frontend/src/screens/UploadScreen.jsx`
  - `frontend/src/screens/TrainingPage.jsx`
  - `frontend/src/screens/LeaderboardScreen.jsx`
- **Message**: `frontend: refactor upload, training, and leaderboard screens`

### Commit 8: New Feature Engineering and Visualization Screens
- **Files**:
  - `frontend/src/screens/FeatureEngineeringPage.jsx`
  - `frontend/src/screens/VisualizationPage.jsx`
- **Message**: `frontend: implement feature engineering and data visualization pages`

### Commit 9: Database and Documentation
- **Files**:
  - `auth.db`
  - `plans/fix_auth_db_issue_20260620.md`
  - `plans/commit_and_push_20260621.md`
- **Message**: `db/docs: update auth.db and include diagnostic plan and commit plan`

## Execution Plan
1. Reset any existing staged changes.
2. For each commit:
   - Run `git add` for the specified files.
   - Run `git commit -m "<message>"` (no "co-authored-by" trailers).
3. Push the commits to the remote `dev` branch: `git push origin dev`.
