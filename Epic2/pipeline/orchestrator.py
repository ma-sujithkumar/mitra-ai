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

ORCHESTRATOR_INSTRUCTION_TEMPLATE = """You orchestrate a feature engineering pipeline. Your job: call the tools below in the right order until the pipeline is complete, then stop.

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
- Call each tool exactly once if it returns {{"status": "ok"}}.
- If a tool returns {{"status": "error"}}, retry it up to {max_tool_retries} times. On the next failure, skip to the next tool and continue.
- After write_report returns ok, respond with exactly the text: DONE. Nothing else.
- CRITICAL: Tool names must be used EXACTLY as listed above. Never append, prepend, or attach any suffix, prefix, annotation, commentary, or extra characters to a tool name. For example, call "create_features_pre" not "create_features_pre..commentary" or "create_features_pre_note". Only the exact name as listed is valid.
- Do not add any notes, labels, or commentary by modifying the tool name. If you want to reason, do it in plain text before calling the tool.
"""

START_MESSAGE = "Begin the feature engineering pipeline. Start with profile_data."


def _make_model_call(
    model_string: str,
    max_tokens: int,
    api_key: str,
    base_url: str | None,
) -> Callable[[str], str]:
    """Returns a callable(prompt) -> response_text routed through our ADK-native
    OpenAICompatibleLlm. No litellm in the path."""
    from pipeline.openai_llm import OpenAICompatibleLlm

    llm = OpenAICompatibleLlm(
        model=model_string,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
    )

    import asyncio
    from google.genai import types as genai_types
    from google.adk.models.llm_request import LlmRequest

    def call(prompt: str) -> str:
        async def _go():
            contents = [genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])]
            req = LlmRequest(
                model=model_string,
                contents=contents,
                config=genai_types.GenerateContentConfig(max_output_tokens=max_tokens),
            )
            chunks: list[str] = []
            async for resp in llm.generate_content_async(req, stream=False):
                if resp.content and resp.content.parts:
                    for part in resp.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            chunks.append(text)
            return "".join(chunks)

        try:
            loop = asyncio.get_running_loop()
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

        # Read API key & endpoint from config. Inject into env BEFORE any ADK / LiteLlm import.
        # ADK imports are deliberately kept inside method bodies (_make_model_call, _run_adk_agent)
        # so env vars are set before the provider client reads them.
        api_key = (self.config.llm.api_key or "").strip()
        if not api_key or api_key == "your_actual_key_here":
            raise RuntimeError(
                "config.llm.api_key is empty or still set to the placeholder. "
                "Set a real API key in config/config.yaml before running."
            )
        # Env var name comes from config.llm.api_key_env_var. If it is left as
        # the default OPENAI_API_KEY but the model_string targets a different
        # provider, derive from the model prefix as a safety net so the right
        # provider client picks the key up. Plan ambiguity #22 mandates that
        # the override is logged so the user can see what happened.
        configured_env_var = (self.config.llm.api_key_env_var or "").strip() or "OPENAI_API_KEY"
        env_var = configured_env_var
        env_var_override_msg: str | None = None
        if env_var == "OPENAI_API_KEY":
            prefix = self.model_string.split("/", 1)[0].lower()
            derived = {
                "openai": "OPENAI_API_KEY",
                "gemini": "GOOGLE_API_KEY",
                "google": "GOOGLE_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }.get(prefix, "OPENAI_API_KEY")
            if derived != configured_env_var:
                env_var = derived
                env_var_override_msg = (
                    f"env_var_override: configured={configured_env_var} "
                    f"effective={env_var} (from model prefix '{prefix}')"
                )
        os.environ[env_var] = api_key
        if self.config.llm.base_url:
            os.environ["OPENAI_API_BASE"] = self.config.llm.base_url

        # Spec §4 "Null detection in categoricals": only empty strings become
        # NaN. Tokens like "NA", "N/A", "None" reach SemanticTypeInfer as
        # plain strings so they can be assigned as legitimate category labels
        # on categorical/binary columns.
        df = pd.read_csv(self.data_path, keep_default_na=False, na_values=[""])
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

        # Route every LLM call's raw response + outcome to raw_responses.txt
        # so fallback paths are debuggable without rerunning. Per-attempt cap
        # comes from config (spec §6 "Observability detail").
        from pipeline import responses as _responses
        _responses.set_raw_log(
            str(output_dir / "raw_responses.txt"),
            max_chars=self.config.validation.raw_log_max_chars,
        )

        # Log task resolution at startup plus any env-var override.
        log_lines = [
            f"[{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')}] startup task={resolved_task} ({task_source}) "
            f"target={self.target_column} nunique={target.nunique(dropna=True)}\n",
        ]
        if env_var_override_msg:
            log_lines.append(
                f"[{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')}] {env_var_override_msg}\n"
            )
        (output_dir / "execution_log.txt").write_text("".join(log_lines), encoding="utf-8")

        state = PipelineState(
            df=features,
            target=target,
            task=resolved_task,
            target_column=self.target_column,
            run_id=run_id,
            config=self.config,
            output_dir=output_dir,
        )

        model_call = _make_model_call(
            self.model_string,
            self.config.llm.max_tokens,
            api_key,
            self.config.llm.base_url,
        )
        # Smoke test — structured-output dry run (spec §5 "Startup smoke test").
        # Sends a one-column SemanticTypeInfer-shaped prompt through the same
        # model_call path the tools use, parses via validate_response with
        # relaxed thresholds, aborts on `failures=['parse']`.
        import traceback as _tb
        try:
            self._run_structured_smoke_test(model_call, output_dir)
        except Exception as e:
            raise RuntimeError(
                f"Model smoke test failed.\n"
                f"  model={self.model_string!r}\n"
                f"  base_url={self.config.llm.base_url!r}\n"
                f"  error_type={type(e).__name__}\n"
                f"  error={e}\n"
                f"--- traceback ---\n{_tb.format_exc()}"
            ) from e

        # Construct the Judge Agent: isolated ADK sub-agent that ranks
        # FeatureCreator proposals and plans FeatureSelector actions. Reuses
        # the same model/endpoint.
        from pipeline.judge_agent import JudgeAgent
        judge_agent = JudgeAgent(
            model_string=self.model_string,
            api_key=api_key,
            base_url=self.config.llm.base_url,
            max_tokens=self.config.llm.max_tokens,
        )

        adk_tools.set_pipeline_state(state, model_call, judge_agent=judge_agent)

        # Run pipeline via ADK Agent
        self._run_adk_agent(state)

        # Write feature_artifact.json (orchestrator owns it; reporter only writes report.md)
        artifact = {
            "run_id": run_id,
            "task": resolved_task,
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

        from pipeline.openai_llm import OpenAICompatibleLlm

        agent = Agent(
            name="feature_engineer_orchestrator",
            model=OpenAICompatibleLlm(
                model=self.model_string,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                max_tokens=self.config.llm.max_tokens,
            ),
            description="Orchestrates the feature engineering pipeline.",
            instruction=ORCHESTRATOR_INSTRUCTION_TEMPLATE.format(
                max_tool_retries=self.config.pipeline.max_tool_retries,
            ),
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

    def _run_structured_smoke_test(self, model_call: Callable[[str], str], output_dir: Path) -> None:
        """Send one structured-output prompt and parse it through validate_response.

        Per spec §5 / §7-Y: aborts startup on `failures=['parse']`. Other
        content failures are logged to execution_log.txt only — the smoke
        prompt is a transport/parse check, not a quality benchmark.
        """
        # Pre-filter: empty-response check (preserves old behavior).
        bare = model_call("Respond with the single word: ok")
        if not bare or not bare.strip():
            raise RuntimeError("smoke test pre-check returned empty response")

        # Structured probe.
        from pipeline.evidence import ColumnTypeEvidence, SemanticTypeInferEvidence, render
        from pipeline.responses import SemanticTypeInferResponse, validate_response
        from pipeline.tools.infer import INFER_PROMPT, _strategy_definitions_block

        packet = SemanticTypeInferEvidence(columns=[
            ColumnTypeEvidence(
                name="example_col",
                dtype="int64",
                null_rate=0.0,
                nunique=2,
                top_values=["0", "1"],
                random_samples=["0", "1"],
                regex_signature={"uuid": 0, "email": 0, "iso_date": 0, "phone": 0, "numeric_string": 2},
            )
        ])
        evidence_block, sent_fields = render(packet)
        prompt = INFER_PROMPT.format(
            strategy_definitions=_strategy_definitions_block(),
            min_rationale_chars=1,
            evidence_block=evidence_block,
        )
        raw = model_call(prompt)

        # Build a relaxed-validation config view from the real config.
        cfg = self.config
        relaxed = _RelaxedValidationView(cfg)
        parsed, failures = validate_response(SemanticTypeInferResponse, raw, sent_fields, relaxed)

        log_line = (
            f"[{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')}] smoke_test "
            f"failures={failures} parsed={'yes' if parsed is not None else 'no'}\n"
        )
        with open(output_dir / "execution_log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)

        if failures and "parse" in failures:
            raise RuntimeError(
                f"structured smoke test failed at parse stage: {failures}. "
                f"Raw response (first 1000 chars): {raw[:1000]!r}"
            )


class _RelaxedValidationView:
    """validate_response calls cfg.validation.*; this view relaxes the
    thresholds without mutating the real config."""

    def __init__(self, cfg):
        self._cfg = cfg
        self.validation = _RelaxedValidationSettings(cfg.validation)


class _RelaxedValidationSettings:
    def __init__(self, real):
        self.min_rationale_chars = 1
        self.min_alternatives = 0
        self.lazy_response_threshold = 1.01  # never trip degeneracy
        self.lazy_min_batch_size = getattr(real, "lazy_min_batch_size", 3)
        self.boilerplate_denylist: list[str] = []
