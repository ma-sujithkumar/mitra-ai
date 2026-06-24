# Plan: Rich Judge Agent Inputs (Metrics, SHAP, Overfitting) & Transcripts Visualizer

## Objective
Enable the Judge Agent to receive detailed validation and holdout metrics, detailed SHAP values, and comprehensive overfitting results (including training, validation/test metrics, and K-Fold cross validation outcomes). Also, allow users to view the entire raw prompt and response transcripts of the Judge LLM on both the Training Page and Leaderboard screens.

---

## 1. Backend Changes

### A. Extend `OverfittingInfo` Schema
- **File**: `backend/agents/evaluation/judge/schemas.py`
- **Changes**: Add optional fields to `OverfittingInfo` so that the rule-independent context data passed to the judge contains the rich overfitting details:
  - `train_metrics: Optional[Dict[str, float]] = None`
  - `test_metrics: Optional[Dict[str, float]] = None`
  - `cv_results: Optional[Dict[str, Any]] = None`

### B. Enrich Upstream Adapter Maps
- **File**: `backend/agents/evaluation/judge/adapter.py`
- **Changes**:
  - Update `adapt_overfitting` to extract `train_metrics`, `test_metrics`, and `cv_results` from the overfitting analyzer json.
  - Update `adapt_from_hpt_results` to map `train_metrics` and `val_metrics` (acting as test/validation metrics) from the HPT results dictionary.

### C. Update Prompt Template
- **File**: `backend/agents/evaluation/judge/prompts/judge_prompt.jinja2`
- **Changes**: Formulate a detailed breakdown in the prompt template for overfitting details per model:
  - Output train metrics if present.
  - Output test/validation metrics if present.
  - Output cross-validation results if present.

---

## 2. Frontend Changes

### A. Display Judge Agent Transcript on Training Page
- **File**: `frontend/src/screens/TrainingPage.jsx`
- **Changes**: Under the "Agent Reasoning" card, if `verdictData?.decision_trace?.transcript` is available, render a styled `<details>` block summarizing "View Full LLM Prompt Transcript".
- **Style**: Use a collapsible component styling with standard CSS classes matching the app's theme. Show the transcript in a scrollable monospace `<pre className="reasoning-block">` block.

### B. Display Judge Agent Transcript on Leaderboard Screen
- **File**: `frontend/src/screens/LeaderboardScreen.jsx`
- **Changes**: Similarly, add an expandable `<details>` section inside the "Agent Reasoning" card on the Leaderboard page.

---

## 3. Verification & Execution
- Run `pytest` to confirm all existing tests (including judge tests) still pass.
- Run frontend development server to inspect visual changes.
