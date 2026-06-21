# Plan: Relax Judge Gating Thresholds and Tune LLM Prompt Bar

This plan relaxes the deterministic rule engine's hard-gates and updates the Judge Agent's prompt guidelines to accommodate datasets with many classes (such as multiclass classification with 15 classes on `matches.csv`), where absolute accuracy and recall scores are naturally lower.

## 1. Proposed Changes
### A. Relax Gating Floors in `config.yaml`
In [backend/agents/evaluation/judge/config/config.yaml](file:///home/sujithma/mitra/backend/agents/evaluation/judge/config/config.yaml):
- Lower `accuracy_floor` from `0.60` to `0.30`.
- Lower `r2_floor` from `0.40` to `0.20`.
- Lower `macro_recall_floor` (under `findings`) from `0.50` to `0.30`.

### B. Tune the Evaluation Bar in the LLM Prompt
In [backend/agents/evaluation/judge/prompts/judge_prompt.jinja2](file:///home/sujithma/mitra/backend/agents/evaluation/judge/prompts/judge_prompt.jinja2), add a new mandatory rule:
```jinja2
- Account for class count: For multiclass classification with many classes (e.g., 5+ classes), absolute accuracy and recall metrics are naturally lower. Evaluate performance relative to a random baseline (1 / num_classes) rather than expecting high absolute scores, and avoid flagging issues unnecessarily if the model performs significantly better than random guessing.
```

## 2. Verification Plan
- Run judge agent unit tests to ensure rule engine logic works correctly:
  `~/venv/bin/pytest backend/agents/evaluation/judge/tests/`
