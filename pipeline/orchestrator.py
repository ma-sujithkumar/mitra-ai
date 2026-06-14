from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import ray
from pandas.api.types import is_numeric_dtype

from pipeline.config import ConfigSchema, load_config
from pipeline.state import PipelineState
from pipeline.tools import adk_tools

ORCHESTRATOR_INSTRUCTION = """You orchestrate a feature engineering pipeline. Your job: call the tools below in the right order until the pipeline is complete, then stop.

Default tool sequence:
1. profile_data
2. infer_types
3. handle_missing
4. handle_outliers
5. create_features_pre
6. encode_features
7. create_features_post
8. scale_features
9. select_features
10. validate_features
11. write_report

Rules:
- Call each tool exactly once if it returns {"status": "ok"}.
- If a tool returns {"status": "error"}, retry it up to 3 times. On a 4th failure, skip to the next tool and continue.
- After write_report returns ok, respond with exactly the text: DONE. Nothing else.
- Do not invent tool names. Only call tools from the registered list.
"""

START_MESSAGE = "Begin the feature engineering pipeline. Start with profile_data."


def _make_model_call(model_string: str) -> Callable[[str], str]:
    """Returns a callable(prompt) -> response_text using ADK's LiteLlm wrapper.

    Resolves provider from the model string. Caller must set the relevant API key env var.
    """
    try:
        from google.adk.models.lite_llm import LiteLlm
        llm = LiteLlm(model=model_string)
    except Exception as e:
        raise RuntimeError(
            f"Failed to construct LiteLlm({model_string}). Install google-adk and set the relevant API key env var. Detail: {e}"
        )

    import asyncio
    from google.genai import types as genai_types

    def call(prompt: str) -> str:
        async def _go():
            from google.adk.models.llm_request import LlmRequest
            contents = [genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])]
            req = LlmRequest(model=model_string, contents=contents, config=genai_types.GenerateContentConfig())
            chunks: list[str] = []
            async for resp in llm.generate_content_async(req, stream=False):
                if resp.content and resp.content.parts:
                    for part in resp.content.parts:
                        if getattr(part, "text", None):
                            chunks.append(part.text)
            return "".join(chunks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(_go())

    return call


class FeatureEngineerOrchestrator:
    def __init__(
        self,
        data_path: str | Path,
        target_column: str,
        model_string: str,
        task: str | None = None,
        config_path: str | Path = "config/config.yaml",
    ):
        if task is not None and task not in {"classification", "regression"}:
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
        if not model_string:
            raise ValueError("model_string is required (e.g., 'gemini/gemini-2.0-flash', 'openai/gpt-4o')")
        self.data_path = Path(data_path)
        self.task = task  # may be None; resolved at run() once dataset is loaded
        self.target_column = target_column
        self.model_string = model_string
        self.config: ConfigSchema = load_config(config_path)

    def _infer_task(self, target: pd.Series) -> str:
        """Infer task from target column. Non-numeric -> classification.
        Numeric: strictly greater than threshold -> regression; at-or-below -> classification.
        """
        if not is_numeric_dtype(target):
            return "classification"
        threshold = self.config.pipeline.task_infer_nunique_threshold
        return "regression" if target.nunique(dropna=True) > threshold else "classification"

    def run(self) -> tuple[Path, str]:
        # Startup sequence (config already validated in __init__)
        df = pd.read_csv(self.data_path)
        if self.target_column not in df.columns:
            raise ValueError(f"target column {self.target_column!r} not in dataset columns {list(df.columns)}")
        target = df[self.target_column].copy()
        features = df.drop(columns=[self.target_column])

        # Resolve task: inference if not supplied
        if self.task is None:
            resolved_task = self._infer_task(target)
            task_source = "inferred"
        else:
            resolved_task = self.task
            task_source = "supplied"

        if not ray.is_initialized():
            ray.init(num_cpus=self.config.pipeline.max_workers, ignore_reinit_error=True, log_to_driver=False)

        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        output_dir = Path("pipeline_output") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Log task resolution at startup
        (output_dir / "execution_log.txt").write_text(
            f"[{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')}] startup task={resolved_task} ({task_source}) "
            f"target={self.target_column} nunique={target.nunique(dropna=True)}\n",
            encoding="utf-8",
        )

        state = PipelineState(
            df=features,
            target=target,
            task=resolved_task,
            target_column=self.target_column,
            run_id=run_id,
            config=self.config,
            output_dir=output_dir,
        )

        model_call = _make_model_call(self.model_string)
        # Smoke test
        try:
            smoke = model_call("Respond with the single word: ok")
            if not smoke or not smoke.strip():
                raise RuntimeError("smoke test returned empty response")
        except Exception as e:
            raise RuntimeError(f"Model smoke test failed: {e}")

        adk_tools.set_pipeline_state(state, model_call)

        # Run pipeline via ADK Agent
        self._run_adk_agent(state)

        # Write feature_artifact.json (orchestrator owns this per Plan ambiguity #7)
        artifact = {
            "run_id": run_id,
            "task": self.task,
            "target_column": self.target_column,
            "dropped_columns": state.dropped_columns,
            "created_columns": state.created_columns,
            "transformers": state.transformers,
            "selected_columns": state.selected_columns or [],
            "selection_method": state.selection_method,
            "warnings": state.warnings,
        }
        (output_dir / "feature_artifact.json").write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")

        # Write engineered_dataset.csv
        state.df.to_csv(output_dir / "engineered_dataset.csv", index=False)

        return output_dir, run_id

    def _run_adk_agent(self, state: PipelineState) -> None:
        try:
            from google.adk.agents import Agent
            from google.adk.runners import InMemoryRunner
            from google.genai import types as genai_types
        except ImportError as e:
            raise RuntimeError(f"google-adk not installed: {e}")

        agent = Agent(
            name="feature_engineer_orchestrator",
            model=self.model_string,
            description="Orchestrates the feature engineering pipeline.",
            instruction=ORCHESTRATOR_INSTRUCTION,
            tools=adk_tools.ALL_TOOLS,
        )

        runner = InMemoryRunner(agent=agent, app_name="fe_pipeline")
        import asyncio

        async def _run():
            session = await runner.session_service.create_session(
                app_name="fe_pipeline", user_id="caller", session_id=state.run_id
            )
            content = genai_types.Content(role="user", parts=[genai_types.Part(text=START_MESSAGE)])
            async for _ in runner.run_async(
                user_id="caller", session_id=session.id, new_message=content
            ):
                pass

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())
