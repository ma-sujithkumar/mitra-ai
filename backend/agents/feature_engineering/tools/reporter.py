import json
from typing import Callable

from backend.agents.feature_engineering.base import BaseTool, PostconditionError, PreconditionError
from backend.agents.feature_engineering.state import PipelineState

REPORT_PROMPT = """You are a feature engineering analyst. Write a concise Markdown report from this pipeline summary.

Use these section headings exactly: Data Quality, Encoding, Features Created, Feature Selection, Warnings.

Pipeline summary (JSON):
{summary}
"""

TEMPLATE = """# Feature Engineering Report — run {run_id}

## Data Quality
- Dropped columns: {dropped_columns}
- Imputation steps: {imputation_count}

## Encoding
- Encoded columns: {encoded_columns}

## Features Created
- Total: {created_count}
- Names: {created_names}

## Feature Selection
- Method: {selection_method}
- Selected: {selected_columns}

## Warnings
{warnings_block}
"""


class FeatureReporter(BaseTool):
    def __init__(self, model_call: Callable[[str], str] | None):
        # model_call is None => write report.md from the deterministic template
        # (no LLM call). This is the default per config.report.use_llm = false.
        self.model_call = model_call

    def precondition(self, state: PipelineState) -> None:
        if state.output_dir is None:
            raise PreconditionError("FeatureReporter: output_dir is None")
        if state.selected_columns is None:
            raise PreconditionError("FeatureReporter: validation must complete first")

    def run(self, state: PipelineState) -> None:
        summary = {
            "run_id": state.run_id,
            "task": state.task,
            "target_column": state.target_column,
            "dropped_columns": state.dropped_columns,
            "created_columns": state.created_columns,
            "transformers": state.transformers,
            "selected_columns": state.selected_columns,
            "selection_method": state.selection_method,
            "warnings": state.warnings,
        }
        if self.model_call is None:
            report_md = self._template(state, summary)
        else:
            prompt = REPORT_PROMPT.format(summary=json.dumps(summary, indent=2, default=str))
            try:
                report_md = self.model_call(prompt)
                if not isinstance(report_md, str) or not report_md.strip():
                    raise ValueError("empty model response")
            except Exception as e:
                state.warnings.append(f"Reporter model call failed: {e}; using template")
                report_md = self._template(state, summary)

        (state.output_dir / "report.md").write_text(report_md, encoding="utf-8")

    @staticmethod
    def _template(state: PipelineState, summary: dict) -> str:
        encoded = [t["column"] for t in state.transformers if t.get("step") == "encoding"]
        imputation_count = sum(1 for t in state.transformers if t.get("step") == "imputation")
        warnings_block = "\n".join(f"- {w}" for w in state.warnings) if state.warnings else "_None_"
        return TEMPLATE.format(
            run_id=state.run_id,
            dropped_columns=", ".join(state.dropped_columns) or "_None_",
            imputation_count=imputation_count,
            encoded_columns=", ".join(encoded) or "_None_",
            created_count=len(state.created_columns),
            created_names=", ".join(c["name"] for c in state.created_columns) or "_None_",
            selection_method=state.selection_method,
            selected_columns=", ".join(state.selected_columns or []) or "_None_",
            warnings_block=warnings_block,
        )

    def postcondition(self, state: PipelineState) -> None:
        report_path = state.output_dir / "report.md"
        if not report_path.exists():
            raise PostconditionError(f"FeatureReporter: {report_path} not written")
