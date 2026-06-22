# Domain Reasoning Agent Specification

The **Domain Reasoning Agent** is a specialized LLM agent in the MITRA pipeline. Its primary role is to infer and explain the semantic meaning of dataset features, construct a real-world description of the prediction task, and identify potential data leakage risks. 

By analyzing feature metadata and value distributions, the agent provides a "real-world" contextual understanding of the dataset. This context enables downstream agents (like the LLM Judge) to evaluate feature importance and model behavior based on plausibility rather than raw performance metrics alone.

---

## 1. Overview and Goal
When raw data is uploaded, a machine learning pipeline can easily exploit "leaky" features (e.g., target-dependent variables or future data recorded post-decision). The Domain Reasoning Agent analyzes column semantics to flag these features.

- **Primary Goal**: Generate semantic documentation and evaluate timing/leakage risks for each feature in the dataset.
- **Output Artifact**: `.mitra/<session_id>/reports/domain_reasoning.json`

---

## 2. Core Inputs
The agent operates under strict sandbox conditions and is not allowed to read the full dataset or environment configurations. It accesses only the following parsed summaries:

1. **Metadata JSON (`reports/metadata.json`)**: 
   Contains the identified target column, problem type (classification/regression/etc.), and list of input columns.
2. **Mini Data Summary (`data/mini_data.csv`)**: 
   A transposed summary of dataset statistics (comparable to the output of `pandas.describe(include="all").transpose()`), giving value counts, unique counts, top frequent values, frequency, mean, standard deviation, min, and max.

---

## 3. Agent Architecture and Tools
The agent is built as a `google.adk.agents.LlmAgent` using a LiteLLM model client wrapper. It is equipped with specific tools to interact with the workspace:

### Available Tools
* **`read_metadata(session_id: str) -> dict[str, Any]`**: Reads the current session's `metadata.json`.
* **`read_mini_data(session_id: str) -> str`**: Reads the summary statistics from `mini_data.csv`.
* **`write_domain_reasoning(session_id: str, domain_reasoning: dict[str, Any]) -> dict[str, str]`**: Validates the schema of the generated domain reasoning dictionary and writes it to `.mitra/<session_id>/reports/domain_reasoning.json`.

---

## 4. Reasoning Logic and Output Format

The agent generates a structured JSON object containing:

### Target and Problem Context
* **`problem_summary`**: A one-to-two sentence description of the prediction problem in plain English (e.g., "Predicting passenger survival on the Titanic using biographical and travel details").
* **`target_explanation`**: A brief explanation of what the target variable represents in the real world.

### Column Explanations
For each feature column, the agent outputs:
1. **`meaning`**: A plain-English explanation of what the feature represents.
2. **`timing`**: Classified into one of:
   - `pre_decision`: Values are naturally known before or at the moment the prediction is made.
   - `post_decision`: Values are recorded only after the outcome/target event has occurred.
   - `unknown`: Ambiguous timing.
3. **`leakage_risk`**: Evaluated as:
   - `high`: Feature is `post_decision` and directly/trivially reveals the target.
   - `low`: Feature is `post_decision` but weakly correlated or ambiguous.
   - `none`: Feature is `pre_decision` or unrelated.
4. **`rationale`**: A sentence explaining the timing and leakage risk designation.

---

## 5. JSON Schema Validation
All output written by the agent via `write_domain_reasoning` is validated against a Draft-07 JSON Schema.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": [
    "session_id",
    "problem_summary",
    "target_explanation",
    "column_explanations",
    "overall_leakage_flags"
  ],
  "properties": {
    "session_id": { "type": "string" },
    "problem_summary": { "type": "string" },
    "target_explanation": { "type": "string" },
    "column_explanations": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["meaning", "timing", "leakage_risk", "rationale"],
        "properties": {
          "meaning": { "type": "string" },
          "timing": {
            "type": "string",
            "enum": ["pre_decision", "post_decision", "unknown"]
          },
          "leakage_risk": {
            "type": "string",
            "enum": ["none", "low", "high"]
          },
          "rationale": { "type": "string" }
        }
      }
    },
    "overall_leakage_flags": {
      "type": "array",
      "items": { "type": "string" }
    }
  }
}
```

---

## 6. Pipeline Integration

The agent is integrated into Stage 1.5 of the pipeline (`PipelineRunner` in [run_pipeline.py](file:///home/sujithma/mitra/backend/orchestration/run_pipeline.py)):

* **Execution Order**: Runs immediately after Stage 1 (Metadata Generation) and before Stage 2 (Feature Engineering).
* **Fault Tolerance**: Domain reasoning is treated as a **non-fatal** step. If the agent fails (e.g., due to model rate-limits or tool errors), the pipeline logs a warning and proceeds. The downstream LLM Judge is designed to check for `domain_reasoning.json` and gracefully fall back to rules-only decisions if the file is missing.
