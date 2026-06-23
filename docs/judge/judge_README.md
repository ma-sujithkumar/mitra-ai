# Judge Agent

Ranks 5-10 candidate ML models and nominates the top model for the leaderboard.

## Architecture

```
JudgeInput (adapter schema)
    |
    v
RuleEngine       <-- authoritative: hard gates, weighted score, tie-break
    |
    v
JudgeAgent       <-- optionally enriches with LLM rationale (additive only)
  model = ClaudeAdkLlm (ADK BaseLlm wrapping custom_anthropic_client)
  prompts = prompts/judge_prompt.jinja2
    |
    v
JudgeDecision -> judge_decision.json
```

## Setup

### 1. Python environment (Python 3.13 required)

google-adk 2.2.0 supports Python 3.10-3.13. Recreate the venv on 3.13 if needed.

Install dependencies:
```
pip install google-adk jinja2 python-dotenv pydantic PyYAML
```

Update `config/config.ini [python] PYTHON` to point at the 3.13 venv binary.

### 2. LLM credentials (for LLM enrichment)

The agent uses `custom_anthropic_client.py` which routes through the local `claude` CLI.

Set environment variables:
```
CLAUDE_CLI_PATH=<path to claude binary>
ANTHROPIC_MODEL_NAME=<haiku|sonnet|opus>
```

Verify: `claude -p "ping"`

### 3. Vendor custom_anthropic_client.py

Replace the stub `custom_anthropic_client.py` with the real file from
https://github.com/ma-sujithkumar/custom_anthropic_client when available.
The public interface (class name, method signatures, response schema) is identical.

## Usage

```
python run_judge.py -i <input_json> -o <output_dir> [-v] [--no-llm]
```

- `-i` : path to the adapter-schema input JSON (see input_format_requirement.md)
- `-o` : output directory (created if absent); writes `judge_decision.json`
- `-v` : verbose/debug logging
- `--no-llm` : rule-only mode (no LLM call; deterministic and offline)

Example:
```
python run_judge.py \
    -i tests/mock_data/judge_input_classification.json \
    -o claude_outputs/judge \
    --no-llm \
    -v
```

## Running Tests

Rule-only tests require no network or LLM:
```
python -m pytest tests/ -v
```

Live LLM test (requires CLAUDE_CLI_PATH and ANTHROPIC_MODEL_NAME):
```
python -m pytest tests/ -v -k "live_llm"
```

## Key files

| File | Purpose |
|------|---------|
| `schemas.py` | Pydantic models for all inputs and outputs |
| `adapter.py` | Maps upstream tool outputs into CandidateModel |
| `rule_engine.py` | Hard gates, scoring, tie-break (authoritative) |
| `claude_adk_llm.py` | ADK BaseLlm wrapping custom_anthropic_client |
| `judge_agent.py` | Orchestrates rule engine + LLM enrichment |
| `run_judge.py` | CLI entry point |
| `prompts/judge_prompt.jinja2` | Jinja2 LLM prompt template |
| `config/config.yaml` | All controllables (weights, floors, tie_break_pct, etc.) |
| `input_format_requirement.md` | Adapter input schema documentation |
