# UI, Backend Integration & Epic 4 Judge Agent Fixes Specification

## Objective

Fix broken UI workflows, missing frontend-backend integrations, SSE streaming issues, and improve Epic 4 Judge Agent robustness to detect model bias, shortcut learning, and degenerate model behavior.

**Critical Requirement**

For every fix below:

* Verify frontend components are correctly wired to backend APIs.
* Verify API responses are properly handled in the frontend.
* Verify loading, success, and failure states are displayed.
* Verify SSE/event streams are functioning end-to-end.
* Verify all functionality works after pulling the latest `dev` branch.
* Do not leave any frontend component disconnected from the backend.
* Validate all fixes through end-to-end testing.

---

# Task 0: Sync with Latest Development Branch

## Requirement

Pull and integrate the latest changes from the `dev` branch before starting any implementation.

## Acceptance Criteria

* Latest `dev` branch changes are merged.
* No merge conflicts remain unresolved.
* Application builds successfully.
* Frontend and backend start without errors.
* Existing functionality regression tested.

---

# Task 1: Visualization Page Trigger

## Problem

The Visualization page currently has no mechanism to trigger plot generation.

Users cannot generate or refresh visualizations.

## Required Fix

Add a dedicated button:

### Button

Label:

```text
Generate Visualizations
```

or

```text
Refresh Visualizations
```

### Behavior

When clicked:

1. Call backend visualization API.
2. Generate all visualization plots.
3. Re-populate all graphs.
4. Replace existing plots if already present.
5. Show loading state while processing.

### UI Requirements

#### Loading State

Display:

```text
Generating visualizations...
```

Disable button while request is active.

#### Success State

Display:

```text
Visualizations generated successfully
```

#### Failure State

Display backend error message.

### Backend Validation

Verify:

* Visualization endpoint exists.
* Endpoint is reachable.
* Frontend correctly invokes endpoint.
* Response payload matches frontend expectations.

### Acceptance Criteria

* Button visible on Visualization page.
* Clicking button generates plots.
* Clicking again refreshes plots.
* Existing plots update correctly.
* Loading and error states function.

---

# Task 2: Training Page Missing from UI

## Problem

Training page is no longer visible and appears broken after a recent commit.

## Required Fix

Debug the regression.

### Investigation

Identify:

* Route removal
* Navigation issue
* Conditional rendering issue
* State management issue
* Backend dependency failure

### Required Outcome

Restore:

* Training page visibility
* Navigation access
* Functional training workflow

### Validation

Verify:

* Route registration
* Menu visibility
* Navigation links
* Backend API integration

### Acceptance Criteria

* Training page appears in UI.
* User can navigate to page.
* Training workflow functions correctly.
* No console errors.
* No backend integration failures.

---

# Task 3: Hyperparameter Tuning (HPT) Status & Streaming

## Problem

Current HPT status bar is non-functional.

No trigger exists to start HPT.

Users cannot monitor trial execution progress.

## Required Fix

### HPT Trigger

Add trigger mechanism to start HPT process.

Possible actions:

* Start HPT button
* Existing workflow integration

Verify backend endpoint invocation.

### HPT Status Bar

Display:

* Current trial number
* Total trials
* Completion percentage
* Current best score

Example:

```text
Trial 12 / 50
Best Score: 0.924
Progress: 24%
```

### SSE Streaming

Implement Server-Sent Events (SSE).

Stream:

* Trial started
* Trial completed
* Trial metrics
* Best model updates
* HPT completion

### Leaderboard Integration

Create live event stream panel.

Example:

```text
[10:31:05] Trial 1 started
[10:31:12] Trial 1 completed | Accuracy=0.88
[10:31:13] New Best Model Found
[10:31:20] Trial 2 started
```

### UI Components

Add:

#### HPT Progress Section

* Progress bar
* Trial counter
* Best score

#### HPT Event Stream

Scrollable live log viewer.

Auto-scroll to latest event.

### Acceptance Criteria

* HPT process can be started.
* Status bar updates in real time.
* SSE connection established.
* Trial events streamed live.
* Leaderboard reflects updates.
* Stream survives long-running jobs.

---

# Task 4: Evaluation Pipeline SSE Stream

## Problem

Evaluation pipeline execution is not visible during Live Training.

Users have no visibility into evaluation progress.

## Required Fix

Implement SSE-based event streaming for evaluation pipeline.

### Stream Location

Page:

```text
Live Training
```

### Events to Stream

Examples:

```text
Evaluation Started
Loading Validation Dataset
Running Metrics
Generating SHAP Values
Computing Drift Metrics
Evaluation Completed
```

### UI Component

Create:

#### Evaluation Event Stream Panel

Features:

* Live updates
* Timestamped events
* Auto-scroll
* Error highlighting

Example:

```text
[10:42:01] Evaluation Started
[10:42:03] Dataset Loaded
[10:42:07] Accuracy Calculated
[10:42:11] SHAP Analysis Running
[10:42:20] Evaluation Complete
```

### Status Indicators

Display:

* Running
* Completed
* Failed

### Acceptance Criteria

* SSE connection established.
* Evaluation events stream live.
* Events appear without page refresh.
* Failure events displayed clearly.
* Stream terminates cleanly on completion.

---

# Task 5: Epic 4 Judge Agent Robustness & Model Bias Detection

## Owner

**Kompalli Avinash Bhargav**

## Priority

**HIGH**

---

## Problem Statement

Using the IPL dataset (`matches.csv`), the trained model learned a degenerate shortcut where it predicts that `team_2` wins most matches.

This results in artificially good metrics while providing poor real-world predictive capability.

The current Epic 4 Judge Agent does not detect this failure mode and incorrectly approves the model.

---

## Required Fix

Enhance the Epic 4 Judge Agent to detect:

* Model bias
* Shortcut learning
* Label leakage
* Degenerate prediction behavior
* Majority-class exploitation

The Judge Agent should never rely solely on overall accuracy.

It must reason about model behavior and determine whether the model has learned meaningful predictive patterns.

---

## Example Failure Case

Dataset:

```text
matches.csv (IPL Matches)
```

Observed behavior:

```text
Model predicts team_2 wins most matches.
```

Even if reported accuracy is acceptable, this should be considered suspicious.

The Judge Agent must identify:

* Severe class prediction imbalance
* Dominant prediction of a single class
* Shortcut learning
* Potential data leakage
* Lack of meaningful feature utilization

---

## Judge Agent Responsibilities

### Prediction Distribution Analysis

Example:

```text
Class 0 Predictions: 95%
Class 1 Predictions: 5%
```

or

```text
team_2 wins: 92%
team_1 wins: 8%
```

Flag as suspicious.

---

### Baseline Comparison

Compare against:

* Majority-class baseline
* Random baseline
* Previous best model

Questions:

* Is the model meaningfully better than majority-class prediction?
* Is the model learning useful signal?
* Is the gain statistically meaningful?

---

### Feature Dependence Analysis

Inspect:

* SHAP values
* Feature importance
* Dominant feature concentration

Detect:

* Single feature dominance
* Leakage features
* Trivial predictors

---

### Class-Level Metrics

Mandatory metrics:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Balanced Accuracy

Require per-class analysis.

---

### Prediction Entropy Analysis

Detect:

* Near-identical predictions
* Extremely low prediction diversity
* Overconfident prediction behavior

Flag low-diversity outputs.

---

### Bias Detection Rules

Flag model when:

* One class dominates predictions
* Prediction distribution significantly differs from training distribution
* Majority-class baseline performs similarly
* Feature importance suggests leakage
* Confidence levels are unrealistically high

---

## Recovery Strategy

When Judge Agent rejects a model:

### Required Actions

1. Explain why model was rejected.
2. Identify likely root cause.
3. Recommend corrective action.
4. Trigger alternative model exploration.

Example:

```text
Current model appears biased toward predicting team_2.

Reason:
92% of predictions belong to one class.

Action:
Trying alternative model families.
```

---

### Alternative Model Search

Judge Agent should automatically evaluate:

* XGBoost
* Random Forest
* LightGBM
* CatBoost
* Logistic Regression
* Neural Networks

Compare all candidates and select the most robust model.

---

### Automatic Retry Loop

```text
Reject Model
\u2192 Explain Findings
\u2192 Train Alternative Model
\u2192 Re-evaluate
\u2192 Compare Against Baselines
\u2192 Select Best Robust Model
```

---

## UI Requirements

### Model Evaluation Summary

Display:

```text
Accuracy: 72%

Prediction Distribution:
team_1 wins: 48%
team_2 wins: 52%

Bias Risk:
LOW
```

or

```text
Accuracy: 78%

Prediction Distribution:
team_1 wins: 7%
team_2 wins: 93%

Bias Risk:
HIGH

Model Rejected
```

---

### Judge Agent Findings Panel

Expose reasoning summary to users.

Example:

```text
Judge Agent Findings

\u2713 Accuracy acceptable

\u2717 Prediction distribution highly skewed

\u2717 Majority-class baseline nearly identical

\u2717 Evidence of shortcut learning

Decision:
Reject Model
```

Do not expose internal chain-of-thought.

Expose only concise findings and rationale.

---

## System Prompt Updates for Epic 4 Judge Agent

Add the following mandatory rules:

* Never approve a model based solely on accuracy.
* Always analyze prediction distribution.
* Compare against majority-class baseline.
* Detect class imbalance exploitation.
* Detect shortcut learning patterns.
* Detect label leakage indicators.
* Evaluate feature importance concentration.
* Evaluate class-wise metrics.
* Reject degenerate models even when overall accuracy is high.
* If a model is rejected, automatically evaluate alternative model families.
* Prefer robust, generalizable models over models with inflated metrics.
* Provide explicit approval or rejection rationale.
* Surface findings in the UI.
* Explain why a model is accepted or rejected.
* Recommend corrective actions when issues are found.

---

# Backend Integration Requirements

Mandatory for all tasks.

## Verify API Wiring

For every UI component:

* Endpoint exists.
* Request reaches backend.
* Response handled correctly.
* Error handling implemented.

## Verify SSE Wiring

For every stream:

* SSE endpoint reachable.
* Connection lifecycle handled.
* Reconnect strategy implemented.
* Stream cleanup implemented.

## Error Handling

Display actionable errors.

Example:

```text
Unable to connect to HPT stream.
Retry Connection
```

Avoid silent failures.

## Logging

Add sufficient frontend and backend logs for:

* Trigger execution
* API calls
* SSE connection events
* Stream failures
* Retry attempts

---

# Final Validation Checklist

* [ ] Latest dev branch merged
* [ ] Visualization button added
* [ ] Visualization generation works
* [ ] Training page restored
* [ ] Training workflow functional
* [ ] HPT trigger implemented
* [ ] HPT status bar functional
* [ ] HPT SSE stream functional
* [ ] HPT leaderboard updates live
* [ ] Evaluation SSE stream functional
* [ ] Epic 4 Judge Agent bias detection implemented
* [ ] Prediction distribution analysis implemented
* [ ] Majority-class baseline comparison implemented
* [ ] Alternative model retry loop implemented
* [ ] Judge Agent findings visible in UI
* [ ] Frontend-backend wiring verified
* [ ] Loading states implemented
* [ ] Error states implemented
* [ ] Regression testing completed
* [ ] No console errors
* [ ] No backend exceptions
* [ ] End-to-end validation completed
