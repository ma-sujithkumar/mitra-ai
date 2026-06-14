# Feature Engineering Agent

Agentic feature engineering pipeline. Reads a raw tabular dataset, produces ML-ready features, and writes a replay artifact. Uses **Google ADK** as the agent harness and is provider-agnostic — bring your own LLM.

## Install

```
pip install -r requirements.txt
```

## Quick start

```
# Gemini
export GOOGLE_API_KEY=...
python main.py run data.csv --task classification --target churn --model gemini/gemini-2.0-flash

# OpenAI
export OPENAI_API_KEY=...
python main.py run data.csv --task regression --target price --model openai/gpt-4o

# Anthropic
python main.py run "test data/train.csv" --target SalePrice --model anthropic/claude-sonnet-4-6
```

## What it does

Eleven ADK tools, called by an ADK Agent (the orchestrator), in this default sequence:

```
profile_data → infer_types → handle_missing → handle_outliers
  → create_features_pre → encode_features → create_features_post
  → scale_features → select_features → validate_features → write_report
```

The orchestrator agent retries any tool that returns an error (up to 3 times), then skips and continues. All model calls happen through ADK using the user-supplied model string.

## Output

Each run writes to `pipeline_output/<run_id>/`:

- `engineered_dataset.csv` — transformed data, target last.
- `feature_artifact.json` — replay record (transforms, lineage, selection).
- `report.md` — Markdown summary.
- `execution_log.txt` — per-tool log lines.

## Contracts

See [schema.md](schema.md) for inputs, outputs, env vars, and the PipelineState contract.

## Spec & plan

- [Fe_Spec.md](Fe_Spec.md) — feature spec.
- [plan.md](plan.md) — implementation plan.
