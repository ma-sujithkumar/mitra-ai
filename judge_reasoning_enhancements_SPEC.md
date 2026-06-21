## Leaderboard Judge Reasoning Enhancements

### Problem

The current leaderboard only displays scores and rankings.

Users cannot understand:

* Why a model was selected.
* Why a model was rejected.
* Why one model ranked above another.
* Whether the Judge Agent found bias, leakage, overfitting, or robustness issues.

This makes the decision process opaque and difficult to trust.

---

### Required Fix

For every model evaluated by Epic 4, the Judge Agent must generate a detailed reasoning summary.

The reasoning should be visible directly from the leaderboard.

The leaderboard should not behave as a simple metric table.

It should behave as a model governance and decision dashboard.

---

### Model Decision Card

Each model entry should contain:

#### Model Summary

```text
Model: XGBoost
Accuracy: 78.2%
F1 Score: 76.4%
ROC-AUC: 81.2%

Decision:
APPROVED
```

---

#### Judge Reasoning

Example:

```text
Judge Findings

\u2713 Balanced prediction distribution

\u2713 Outperforms majority-class baseline by 18%

\u2713 Strong feature utilization across 12 features

\u2713 Stable performance across folds

\u2713 No evidence of label leakage

\u2713 Consistent SHAP explanations

Decision:
Approved
```

---

### Rejected Model Example

```text
Model: Random Forest

Accuracy: 79.1%

Decision:
REJECTED
```

Reasoning:

```text
Judge Findings

\u2717 93% of predictions belong to a single class

\u2717 Accuracy only 1.2% better than majority baseline

\u2717 Feature importance concentrated on one feature

\u2717 Potential shortcut learning detected

\u2717 Poor minority-class recall

Decision:
Rejected
```

---

### Mandatory Evaluation Dimensions

The Judge Agent must explicitly evaluate and comment on:

#### Predictive Quality

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Balanced Accuracy

#### Class Balance

* Prediction distribution
* Minority-class performance
* Class-wise recall

#### Generalization

* Cross-validation consistency
* Train vs validation gap
* Signs of overfitting

#### Feature Utilization

* SHAP analysis
* Feature importance diversity
* Dominant feature detection

#### Data Leakage

* Leakage indicators
* Suspicious correlations
* Shortcut learning patterns

#### Robustness

* Stability across folds
* Stability across seeds
* Sensitivity to data changes

---

### Ranking Explanation

For every approved model, provide ranking justification.

Example:

```text
Why Ranked #1

Model achieved the highest balanced accuracy.

Performance remained stable across folds.

Prediction distribution closely matched the training distribution.

No leakage indicators were detected.

SHAP analysis showed meaningful utilization of multiple features.

Model demonstrated the best balance between accuracy and robustness.
```

---

### Comparison Explanation

For top-ranked models, explain why one was selected over another.

Example:

```text
Why XGBoost Beat Random Forest

Both models achieved similar accuracy.

However:

- XGBoost had higher minority-class recall.
- XGBoost showed lower variance across folds.
- Random Forest exhibited class prediction bias.
- XGBoost demonstrated better feature diversity.

Final Decision:
XGBoost Selected.
```

---

### SSE Streaming of Judge Reasoning

As models are evaluated, stream Judge Agent findings live to the leaderboard.

Example:

```text
[Judge] Evaluating Random Forest

[Judge] Accuracy acceptable

[Judge] Prediction distribution highly skewed

[Judge] Majority baseline nearly identical

[Judge] Potential shortcut learning detected

[Judge] Rejecting model
```

Example:

```text
[Judge] Evaluating XGBoost

[Judge] Strong balanced accuracy

[Judge] No leakage indicators

[Judge] Stable cross-validation results

[Judge] Approving model
```

---

### Explainability Requirement

A user should never have to guess why a model was selected or rejected.

Every leaderboard entry must answer:

1. Why was this model approved?
2. Why was this model rejected?
3. What strengths were detected?
4. What weaknesses were detected?
5. Why did it rank where it did?
6. Why was another model preferred?

---

### Acceptance Criteria

* Every model has a Judge Reasoning section.
* Every rejection includes explicit reasons.
* Every approval includes explicit reasons.
* Ranking explanations are available.
* Model comparison explanations are available.
* Reasoning is streamed live through SSE.
* Users can understand model selection decisions without inspecting logs.
* Leaderboard acts as a transparent model governance dashboard rather than a simple metrics table.
