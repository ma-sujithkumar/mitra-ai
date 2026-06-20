# SPEC: Judge Agent (LLM-as-a-Judge for Model Selection)

## 1. GOAL
Rank 5-10 candidate ML models trained on the same dataset and nominate the
top-performing model for the leaderboard, returning to an orchestrator (a) a
per-model verdict (select / reject) and (b) an overall ranking. The agent is a
HYBRID judge: deterministic rules enforce hard gates and tie-breaks; an LLM
(Gemini, via Google ADK) reasons about the candidates and produces the human-
readable rationale WITHIN those guardrails. The LLM cannot override a hard gate,
which keeps the gating decision auditable and testable.

## 2. APPLICATION CONTEXT
1. Runs as a decision step inside an AutoML pipeline, downstream of the
   Overfitting Analysis Tool (`../SPEC.md`) and other per-model scripts.
2. Drives / advises an orchestrator agent: the orchestrator passes candidate
   model data in and receives the ranking + verdicts back to decide whether to
   promote a model to the leaderboard.

## 3. CONSTRAINTS
1. Use Google ADK only to build the agent (https://github.com/google/adk-python).
   Do not write agent plumbing from scratch; use ADK's provided constructs.
   The LLM backend is Claude (via the local `claude` CLI) plugged into ADK via a
   custom `BaseLlm` subclass (`claude_adk_llm.py::ClaudeAdkLlm`) that wraps
   `custom_anthropic_client.py`. No Gemini API key is required.
2. No prompts inline in the code. All prompts live in jinja2 templates rendered
   at runtime (`prompts/judge_prompt.jinja2`).
3. The LLM is asked for structured (JSON-schema) output so verdicts are
   reproducible. The `claude` CLI is invoked via `custom_anthropic_client.py`
   (compatible with `anthropic.Anthropic()` interface). Credentials: set
   `CLAUDE_CLI_PATH` and `ANTHROPIC_MODEL_NAME` env vars.
4. Deterministic-rule fallback: if the LLM / CLI is unavailable, the agent still
   produces gates, scores, and ranking from the rule engine alone (`--no-llm` /
   `use_llm: false` in config). This path is testable without network access.

## 4. INPUTS
The judge owns its own input contract (an ADAPTER schema) so it stays decoupled
from upstream output formats. The concrete schema is documented in
`input_format_requirement.md`; an adapter maps each upstream source into it.

Per-candidate-model fields:
1. `metrics` - performance metrics (classification or regression) for the model.
2. `overfitting` - adapted from the Overfitting Analysis Tool's
   `overfitting_analysis.json` (`is_overfitted`, primary `gap`,
   `train_vs_cv_gap`). The adapter translates that upstream schema into this
   block; the judge never parses the upstream file directly.
3. `complexity` - EXPLICIT complexity descriptor supplied per model, e.g.
   `{ "n_params": int, "depth": int, "family_rank": int }`. Required; the judge
   does not infer complexity.
4. `shap_summary` - SHAP explainability numbers/text (no images). CONTEXT ONLY in
   v1 (see Section 11): informs the LLM rationale/flags, not the score.
5. `hyperparam_sensitivity` - hyperparameter tuning sensitivity metrics. CONTEXT
   ONLY in v1.

Dataset-level fields (shared across candidates):
6. `minidata` - `pd.describe()` output for the dataset. CONTEXT ONLY in v1.
7. `metadata` - user metadata. Format not fixed yet (Section 11). CONTEXT ONLY.

## 5. DECISION LOGIC
Ordered factors by weightage: (i) performance, (ii) low overfitting,
(iii) low complexity. Weights are configurable (Section 7).

1. Hard gate (rules): reject any model whose primary performance metric is below
   the configured floor for its task type:
   - classification: `accuracy_floor`
   - regression: `r2_floor`
2. Rank survivors by a weighted score combining normalized performance,
   `(1 - overfitting_signal)`, and `(1 - normalized_complexity)`.
3. Tie-break (rules): when two models' performance differs by <= `tie_break_pct`
   (default 1%), prefer the simpler model (lower `complexity`).
4. LLM (Gemini via ADK): given the gated/ranked candidates plus the CONTEXT-ONLY
   inputs (SHAP, hyperparam sensitivity, minidata, metadata), produces the
   ranking rationale and may RAISE concern flags. It cannot promote a gated-out
   model or reorder past a hard rule outcome.

## 6. OUTPUT FORMAT
A JSON returned to the orchestrator (and written to
`<output_dir>/judge_decision.json`):
```json
{
  "dataset_id": "string-or-null",
  "selected_model": "XGBClassifier",
  "ranked_models": [
    {
      "model_name": "XGBClassifier",
      "rank": 1,
      "score": 0.91,
      "verdict": "select",
      "reasons": ["above accuracy_floor", "low overfitting gap"],
      "llm_flags": []
    }
  ],
  "decision_trace": {
    "rule_outcomes": {},
    "llm_commentary": "string-or-null"
  }
}
```
- `selected_model`: top nominee, or `null` if every candidate is rejected.
- `verdict`: `select` | `reject` per model.
- `decision_trace`: separates deterministic rule outcomes from LLM commentary so
  the decision is auditable.

## 7. CONTROLLABLES (config/config.yaml)
- `weights`: map of `performance` / `overfitting` / `complexity` weights.
- `accuracy_floor`: classification hard-gate threshold.
- `r2_floor`: regression hard-gate threshold.
- `tie_break_pct`: performance delta within which the simpler model wins
  (default 0.01).
- `complexity_normalization`: how `complexity` fields map to a [0,1] score.
- `llm_model`: Gemini model name.
- `llm_temperature`: default 0.
- `prompt_template_paths`: jinja2 template locations.

Python binary and paths live in `config/config.ini` (`[python] PYTHON=...`,
`[paths] ...`), matching the model_library convention. One config.ini per project.

## 8. CLI ARGS
- `-i, --input_json <path>`: REQUIRED. Path to the adapter-schema input JSON.
  Error if missing (no default).
- `-o, --output_dir <path>`: REQUIRED. Created with `mkdir -p`. Error if missing.
- `-v, --verbose`: enable debug logging.

## 9. DEVELOPMENT OUTPUTS
1. `config/config.ini` - python binary + paths.
2. `config/config.yaml` - all controllables (Section 7).
3. `input_format_requirement.md` - the adapter input schema (Section 4).
4. Agent module(s) built on ADK; imports at top of file; jinja2-rendered prompts.
5. `prompts/` - jinja2 prompt templates (no inline prompts in code).
6. `adapter` - maps the upstream `overfitting_analysis.json` (and other sources)
   into the judge input schema.
7. `tests/` - synthetic multi-model input for both task types; asserts gates,
   tie-break, output schema, and the rule-only (LLM-disabled) path.

## 10. ACCEPTANCE CRITERIA
1. Runs end-to-end on a synthetic 5-10 model input for both classification and
   regression and produces a schema-valid `judge_decision.json`.
2. Any model below its task-type floor (`accuracy_floor` / `r2_floor`) is always
   rejected, regardless of LLM output.
3. When two candidates' performance differs by <= `tie_break_pct`, the simpler
   model is ranked higher.
4. The rule-only path (LLM disabled / unavailable) runs without network access
   and still emits ranking + verdicts; LLM rationale fields are marked
   unavailable.
5. No agent plumbing is written from scratch; ADK constructs are reused. Prompts
   live only in jinja2 templates.

## 11. OPEN ITEMS / ASSUMPTIONS
- `metadata.csv` format is not fixed yet; treated as opaque CONTEXT for the LLM.
- SHAP summary, hyperparameter sensitivity, minidata, and metadata are CONTEXT
  ONLY in v1 (inform the LLM rationale/flags, not the numeric score). Promoting
  any of them to scored decision factors is deferred to v2.
- Complexity is supplied explicitly per model; deriving it upstream is out of
  scope for the judge.
