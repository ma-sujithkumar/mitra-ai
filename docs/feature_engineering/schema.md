# Feature Engineering Agent — I/O Contract

## Inputs

| Arg | Type | Required | Description |
|---|---|---|---|
| `data` | path | yes | Path to a CSV with one header row. |
| `--task` | `classification \| regression` | no | Optional. Inferred from the target column if omitted (numeric target with `nunique > task_infer_nunique_threshold` → regression; otherwise classification). Unrecognised values raise `ValueError` at startup. |
| `--target` | str | yes | Target column name. Must exist in the dataset. |
| `--model` | str | yes | ADK/LiteLLM model identifier (e.g. `gemini/gemini-2.0-flash`, `openai/gpt-4o`). |
| `--config` | path | no | Path to `config.yaml`. Defaults to `config/config.yaml`. |

## API Key

Set `llm.api_key` in `config/config.yaml`. The orchestrator copies it into the provider-specific environment variable (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `ANTHROPIC_API_KEY`) at startup based on the `--model` prefix, before any ADK/LiteLLM import.

`config/config.yaml` is gitignored — never commit a real key.

## Outputs

Every run produces `pipeline_output/<run_id>/` with four files:

### `engineered_dataset.csv`
Transformed dataset. All feature columns are numeric (float). Target column is last.

### `feature_artifact.json`
Replay record for the pipeline.

```json
{
  "run_id": "20260613T143022_a3f1b2c4",
  "task": "classification",
  "target_column": "churn",
  "dropped_columns": ["id", "col_high_nulls"],
  "created_columns": [
    {"name": "col1_div_col2", "operation": "ratio", "sources": ["col1", "col2"]}
  ],
  "transformers": [
    {"step": "imputation", "column": "col1", "strategy": "median", "fill_value": 3.5},
    {"step": "encoding",   "column": "col2", "strategy": "label",  "classes": ["a","b","c"]},
    {"step": "scaling",    "column": "col3", "strategy": "standard","mean": 0.5, "std": 1.2}
  ],
  "selected_columns": ["col1", "col3", "col1_div_col2"],
  "selection_method": "mrmr",
  "warnings": ["col_x had 52% nulls and was dropped"]
}
```

### `report.md`
Markdown report covering data quality, encoding, features created, selection, and warnings. Written by the model from the structured summary; falls back to a string-constant template in `reporter.py` if the model call fails.

### `execution_log.txt`
Append-only per-tool log. One line per tool invocation:

```
[2026-06-13T14:30:22] profile_data ok (1.42s) profiled 12 columns
[2026-06-13T14:30:24] infer_types ok (2.10s) typed 12 columns
```

## PipelineState

Single dataclass passed through every tool. Tools read from and mutate it in place. No `model_fn` field — the ADK orchestrator agent owns all model calls.

Fields:
- `df: pd.DataFrame` — current feature dataframe.
- `target: pd.Series` — target column held separately from features.
- `task: str`, `target_column: str`, `run_id: str` — immutable inputs.
- `config: ConfigSchema` — validated config object.
- `profile: dict | None` — populated by DataProfiler.
- `column_types: dict | None` — populated by SemanticTypeInfer.
- `transformers: list`, `dropped_columns: list`, `created_columns: list`, `warnings: list` — append-only history.
- `selected_columns: list | None`, `selection_method: str | None` — populated by FeatureSelector.
- `output_dir: Path | None` — set by orchestrator.
- `pre_encoding_done: bool` — set by FeatureCreator.run_pre.
- `row_count_after_outlier: int | None` — set by OutlierHandler.

## ADK Tool Functions

The orchestrator agent has access to these tools (one per pipeline stage):

| Tool function | Wraps | Calls model? |
|---|---|---|
| `profile_data` | DataProfiler | no |
| `infer_types` | SemanticTypeInfer | yes (one call) |
| `handle_missing` | MissingValueHandler | yes (one batched call) |
| `handle_outliers` | OutlierHandler | yes (one batched call) |
| `create_features_pre` | FeatureCreator.run_pre | yes (first invocation only) |
| `encode_features` | Encoder | no |
| `create_features_post` | FeatureCreator.run_post | no (specs cached) |
| `scale_features` | Scaler | yes (one batched call) |
| `select_features` | FeatureSelector | yes (one call) |
| `validate_features` | FeatureValidator | no |
| `write_report` | FeatureReporter | yes (one call) |

Each tool returns `{"status": "ok", "detail": "..."}` or `{"status": "error", "detail": "..."}`. The agent retries on errors up to 3 times per tool, then skips and continues.
