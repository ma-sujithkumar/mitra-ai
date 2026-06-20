---
name: epic_4/judge_agent
path: epic_4/judge_agent
purpose: Hybrid rule-based and LLM-based agent that evaluates trained models, scores them based on performance/overfitting/complexity, and emits training verdicts and HPT guidance.
interfaces:
  inputs:
    - name: metrics.json / train_metrics.json
      format: JSON
      upstream: epic_3/training / ray_executor
      description: Train and validation performance metrics for each trained model candidate.
    - name: shap_values.npy
      format: NumPy Array Binary (Optional)
      upstream: model_library / metrics / evaluators
      description: Precomputed SHAP feature explanations.
    - name: judge_config.yaml
      format: YAML
      upstream: config
      description: Defines threshold values (accuracy floor, R2 floor, weights, gap cap, and complexity normalization).
  outputs:
    - name: verdict.json
      format: JSON
      downstream: epic_4/overfitting_analysis_tool / HPT Agent
      description: Aggregated decision containing acceptance status, scoring details, ranked list, and next hyperparameter hints.
entry_points:
  - name: epic_4.judge_agent.judge_agent:JudgeAgent
    type: Python API
    description: Core orchestrator class executing the rule engine, rendering prompt templates, calling the LLM agent, and merging decisions.
  - name: epic_4.judge_agent.rule_engine:RuleEngine
    type: Python API
    description: Implements deterministic safety floors, multi-criteria scoring, and tie-breaking ranks.
  - name: epic_4.judge_agent.run_judge:main
    type: CLI
    description: Execution script for running the Judge agent on an input JSON payload.
dependencies:
  - google.adk
  - jinja2
  - numpy
  - pydantic
---

# Technical Architecture: Judge Agent

## Overview
The `judge_agent` implements a hybrid evaluation system. First, a deterministic `RuleEngine` enforces performance floors, checks train-validation metric gaps, and ranks models. Second, if `use_llm` is enabled, an ADK `LlmAgent` using Claude completes a prompt template to append qualitative commentary and fine-tuned hyperparameter tuning hints.

## Core Component Walkthrough
1. **`rule_engine.py`**:
   - `apply_hard_gates`: Discards models failing to meet performance baselines (e.g. accuracy < 0.55).
   - `score_candidates`: Computes scores based on weights:
     $$Score = W_{perf} \times Perf_{norm} + W_{overfit} \times (1 - Overfit_{norm}) + W_{complex} \times (1 - Complex_{norm})$$
   - `rank_candidates`: Sorts survivors by score, applying tie-breakers based on training time or size.
2. **`judge_agent.py`**:
   - Renders a markdown prompt with candidate details using Jinja2 templates (`prompts/judge_prompt.md.jinja2`).
   - Invokes Claude via `ClaudeAdkLlm` model wrapper.
   - Merges LLM commentary into the deterministic `JudgeDecision` object without altering rule-based rankings.
3. **`claude_adk_llm.py`**: Adapts LLM client wrappers to conform with `google.adk` runner expectations.

## Interfacing Guide
- **Upstream Integration:** Feeds on execution statistics (metrics, weights, dataset name) collected in preceding training stages.
- **Downstream Integration:** Outputs `verdict.json` specifying the accepted champion model or flagging models for tuning with `next_hp_hint` (e.g. increase/decrease parameters) for the HPT Agent.

## Suggested Cleanup/Refactoring
- **Consolidate LLM Factory:** Merge `custom_anthropic_client.py` and `claude_adk_llm.py` into a unified project-wide `LiteLLM` client factory to avoid duplicate anthropic SDK calls.
- **Standardize Schemas:** Re-use Pydantic data schemas defined under `epic_3/training_orchestrator/contracts.py` to prevent structural mismatch of metrics payloads.
