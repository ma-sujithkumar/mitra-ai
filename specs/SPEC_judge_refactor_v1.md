# Judge Agent Refactor Specification

This document details the refactoring of the **Judge Agent** in the MITRA pipeline, comparing the newly introduced architecture with the previous design in the latest commit history.

---

## 1. Executive Summary
The Judge Agent has been refactored from a **passive commentary generator** to an **active ranking and selection system**. In the previous design, the rule engine decided model selection and ranking authoritatively, and the LLM only provided qualitative commentary and flags. 

In the refactored design, the LLM Judge acts as an active ranker that reorders candidate models using performance metrics, overfitting, complexity, and SHAP correlations with domain-reasoning context. This active ranking is constrained by deterministic safety guardrails and finalized by deterministic selection rules.

---

## 2. Structural Differences at a Glance

| Feature / Aspect | Previous Design (Latest Commit) | Refactored Design (Present) |
| :--- | :--- | :--- |
| **Primary LLM Role** | Generate overall text commentary and flags on a fixed, pre-ranked list. | Active ranking (reordering) of surviving candidates. |
| **Dataset Context Inputs** | `metadata.json` and `mini_data.csv` summaries only. | Adds `domain_reasoning.json` containing semantic column annotations. |
| **New Tools Added** | None. | `get_domain_reasoning()` added to `JudgeTools`. |
| **LLM Output Schema** | JSON object with `llm_commentary` and `model_flags`. | Structured list of models with `rank`, `reasoning`, `shap_domain_correlation`, `flags`, plus `overall_commentary`. |
| **Reorder Guardrails** | None (LLM was not allowed to reorder). | Deterministic bubble-pass accuracy clamp (`llm_ranking_max_accuracy_drop`). |
| **Final Selection** | Determined by the rule engine score prior to LLM execution. | Deterministic post-ranking selection (`apply_selection`) using `selection_top_pct` and `selection_min_count`. |
| **Verdict Enums** | `select` (APPROVED) and `reject` (REJECTED). | `select` (APPROVED), `rank_only` (RANKED), and `reject` (REJECTED). |
| **Error / Status Handling** | Silent fallback to rule-only decisions on LLM failures/timeouts. | Surface trace details (`llm_ranking_status`, `llm_ranking_error`) and retryable exceptions. |

---

## 3. Detailed Workflow Comparison

### Previous Flow
```
[Raw Candidate Models]
          │
          ▼
[Rule Engine Gates & Scores] ──► Sets Verdicts ('select' or 'reject') & Ranks
          │
          ▼
[LLM Commentary Call]        ──► Receives Fixed Order; Adds commentary/flags
          │
          ▼
[Merge & Return]
```

### Refactored Flow
```
[Raw Candidate Models]
          │
          ▼
[Rule Engine Gates]           ──► Filters rejects & produces provisional ranking of survivors
          │
          ▼
[LLM Active Ranking Call]     ──► LLM reorders survivors using SHAP & Domain Reasoning tools
          │
          ▼
[Accuracy Guardrail Clamp]    ──► Swaps back models exceeding max accuracy drop threshold
          │
          ▼
[Deterministic Selection]     ──► Selects top-N% (min-count floor); sets others to 'rank_only'
          │
          ▼
[Merge & Return]
```

---

## 4. Schema Changes (`schemas.py`)

1. **`JudgeInput`**:
   * Added `domain_reasoning: Optional[Dict[str, Any]]` field to pass column explanations, task summary, and leakage flags.
2. **`RankedModel`**:
   * Expanded description for `verdict` to support `"select"`, `"rank_only"`, and `"reject"`.
   * Expanded description for `decision` to map verdicts to `"APPROVED"`, `"RANKED"`, and `"REJECTED"`.
   * Added `llm_ranking_reasoning: Optional[str]` to capture the LLM's explanation of model ranking.
3. **`DecisionTrace`**:
   * Added `llm_ranking_status: Optional[str]` (enum: `applied`, `failed`, `skipped`).
   * Added `llm_ranking_error: Optional[str]` to capture exception details when status is `failed`.
4. **`JudgeDecision`**:
   * Added `selected_models: List[str]` to hold the list of selected models.
   * Marked `selected_model: Optional[str]` as **Deprecated** (kept for backward compatibility, mapped to `selected_models[0]`).

---

## 5. Rule Engine Refactoring (`rule_engine.py`)

### A. Accuracy Reorder Guardrail (`enforce_accuracy_reorder_guardrail`)
To prevent the LLM from prioritizing a model with "cleaner" SHAP profiles over a model with vastly superior predictive power, a deterministic safety guardrail was implemented.
* **Mechanism**: A bubble-pass algorithm runs over the LLM-reordered survivors. If model $A$ is ranked below model $B$ but its primary performance metric (accuracy/R2) is better by more than `llm_ranking_max_accuracy_drop` (e.g. `0.10`), the guardrail swaps their ranks back.
* **Configuration**: Configured via `llm_ranking_max_accuracy_drop` under the `[pipeline]` section of `config.ini`.

### B. Deterministic Top-N% Selection (`apply_selection`)
Selection is now decoupled from ranking and happens as the final step.
* **Rule**: Selects the top percentage of eligible models based on:
  $$\text{selected\_count} = \min\left(\text{eligible\_count}, \max\left(\text{selection\_min\_count}, \lceil \text{selection\_top\_pct} \times \text{eligible\_count} \rceil \right)\right)$$
* **Configuration**: Uses `selection_top_pct` and `selection_min_count` from config.
* **Status Mapping**:
  * Selected models in the top list $\rightarrow$ `verdict="select"`, `decision="APPROVED"`.
  * Other surviving models $\rightarrow$ `verdict="rank_only"`, `decision="RANKED"`.

---

## 6. Prompt Refactoring (`judge_prompt.jinja2`)
The system prompt was revised to transition the model instructions:
* **Instructions**: The LLM is explicitly instructed that the survivors have already passed the floor gates. Its task is to rank the remaining models based on metrics, overfitting, complexity, and SHAP correlations.
* **Domain Reasoning Integration**: Instructs the model to check if high-importance SHAP features are flagged as `leakage_risk: high` in the domain reasoning context, and if so, demote their rank.
* **Output Format**: Shifted from generating commentaries and flags to returning a structured `ranking` array of JSON objects alongside `overall_commentary`.
