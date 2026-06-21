"""
JudgeAgent: orchestrates the full judge pipeline.

Flow:
  1. Rule engine gates and ranks candidates (deterministic, authoritative).
  2. If use_llm is enabled, renders a jinja2 prompt and invokes the LLM via
     ADK LlmAgent + LiteLlm (shared build_llm_model factory) to get rationale.
  3. Merges LLM output (additive only) into the JudgeDecision.
  4. Returns the final JudgeDecision.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from llm.adk_client import LlmSettings, build_llm_model
from backend.agents.metadata_gen_agent import LlmSettingsResolver
from backend.config_loader import ConfigLoader
from .config_loader import load_judge_config
from .rule_engine import RuleEngine
from .schemas import CandidateModel, JudgeDecision, JudgeInput, RankedModel

logger = logging.getLogger(__name__)


def _build_candidates_detail(candidates: List[CandidateModel]) -> Dict[str, Any]:
    """Build a model_name => candidate dict for the jinja2 template."""
    return {candidate.model_name: candidate for candidate in candidates}


def _render_prompt(
    judge_input: JudgeInput,
    decision: JudgeDecision,
    prompts_dir: str,
    template_name: str,
) -> str:
    """Render the jinja2 judge prompt with the rule-based decision and context inputs."""
    env = Environment(
        loader=FileSystemLoader(prompts_dir),
        autoescape=select_autoescape(disabled_extensions=("jinja2",)),
    )
    env.policies["json.dumps_kwargs"] = {"indent": 2}
    template = env.get_template(template_name)

    candidates_detail = _build_candidates_detail(judge_input.candidates)
    return template.render(
        dataset_id=judge_input.dataset_id,
        minidata=judge_input.minidata,
        metadata=judge_input.metadata,
        ranked_models=decision.ranked_models,
        candidates_detail=candidates_detail,
    )


async def _invoke_llm_agent(
    prompt_text: str,
    llm_settings: LlmSettings,
) -> Optional[str]:
    """Run the ADK LlmAgent via the shared LiteLlm factory and return the raw text response."""
    llm_model = build_llm_model(llm_settings)
    agent = LlmAgent(
        name="judge_llm",
        model=llm_model,
        instruction="You are a precise ML model ranking commentator. Respond only with the JSON schema requested.",
    )
    runner = InMemoryRunner(agent=agent, app_name="judge_agent")
    session_service = runner.session_service

    session = await session_service.create_session(app_name="judge_agent", user_id="judge")
    response_text_parts: List[str] = []

    # ADK's run_async expects a types.Content (with .role), not a plain dict.
    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt_text)],
    )
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=new_message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts or []:
                if hasattr(part, "text") and part.text:
                    response_text_parts.append(part.text)

    return "".join(response_text_parts) if response_text_parts else None


def _parse_llm_response(
    raw_response: str,
    decision: JudgeDecision,
) -> JudgeDecision:
    """Parse the LLM's JSON response and merge flags/commentary into the decision.

    The rule-based verdicts, scores, and ranking are never changed here.
    """
    raw_stripped = raw_response.strip()
    # Strip markdown code fences if the model wrapped the JSON.
    if raw_stripped.startswith("```"):
        lines = raw_stripped.splitlines()
        raw_stripped = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    parsed: Dict[str, Any] = json.loads(raw_stripped)
    commentary = parsed.get("llm_commentary", "")
    model_flags: Dict[str, List[str]] = parsed.get("model_flags", {})

    updated_ranked: List[RankedModel] = []
    for ranked_model in decision.ranked_models:
        flags = model_flags.get(ranked_model.model_name, [])
        updated_ranked.append(
            ranked_model.model_copy(update={"llm_flags": flags})
        )

    updated_trace = decision.decision_trace.model_copy(
        update={"llm_commentary": commentary}
    )
    return decision.model_copy(
        update={"ranked_models": updated_ranked, "decision_trace": updated_trace}
    )


class JudgeAgent:
    """End-to-end judge: rule engine => optional LLM rationale => JudgeDecision."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or load_judge_config()
        self._rule_engine = RuleEngine(self._config)

        # Resolve prompts directory from config.ini [paths].
        agent_root = os.path.dirname(__file__)
        prompts_subdir = self._config.get("prompts_dir", "prompts")
        self._prompts_dir = os.path.join(agent_root, prompts_subdir)
        self._prompt_template = self._config.get("prompt_template", "judge_prompt.jinja2")

    def judge(
        self,
        judge_input: JudgeInput,
        use_llm: Optional[bool] = None,
    ) -> JudgeDecision:
        """Run the full judge pipeline and return a JudgeDecision.

        Args:
            judge_input: Structured input with all candidate models.
            use_llm: Override config use_llm setting. Pass False for rule-only mode.

        Returns:
            JudgeDecision with rule-authoritative verdicts and optional LLM enrichment.
        """
        should_use_llm = use_llm if use_llm is not None else bool(self._config.get("use_llm", True))

        logger.debug(
            "=> JudgeAgent.judge: %d candidates, use_llm=%s",
            len(judge_input.candidates),
            should_use_llm,
        )

        # Step 1: deterministic gating and ranking.
        survivors, gate_outcomes = self._rule_engine.apply_hard_gates(judge_input.candidates)
        decision = self._rule_engine.rank(
            survivors=survivors,
            gate_outcomes=gate_outcomes,
            all_candidates=judge_input.candidates,
        )
        decision = decision.model_copy(update={"dataset_id": judge_input.dataset_id})

        # Step 2: LLM rationale (additive only; any failure falls back to rule-only output).
        if should_use_llm:
            decision = self._enrich_with_llm(judge_input, decision)

        logger.debug(
            "=> JudgeAgent decision: selected=%s total_ranked=%d",
            decision.selected_model,
            len(decision.ranked_models),
        )
        return decision

    def _enrich_with_llm(
        self,
        judge_input: JudgeInput,
        decision: JudgeDecision,
    ) -> JudgeDecision:
        """Invoke the LLM and merge its output into the decision. Falls back on any error."""
        # Opt-in litellm wire-level debug (judge prompts/responses in backend logs),
        # matching the other agents. Driven by judge config so it is not hardcoded.
        if self._config.get("litellm_debug", False):
            os.environ["LITELLM_LOG"] = "DEBUG"

        try:
            # Resolve credentials the SAME way every other agent does: via
            # LlmSettingsResolver, which reads .env (LLM_TYPE/LLM_MODEL/LLM_API_KEY).
            # Reading os.environ directly was broken -- .env is never exported to the
            # process environment (no load_dotenv anywhere), so the key was always
            # empty and the judge silently fell back to rule-only output.
            judge_llm_settings = LlmSettingsResolver(ConfigLoader()).resolve()
            # INFO so it is visible without DEBUG: confirms the judge actually
            # reaches the LLM call and which model/provider it uses.
            logger.info(
                "=> [JUDGE LLM] invoking provider=%s model=%s for %d candidate(s)",
                judge_llm_settings.provider,
                judge_llm_settings.model,
                len(judge_input.candidates),
            )

            prompt_text = _render_prompt(
                judge_input=judge_input,
                decision=decision,
                prompts_dir=self._prompts_dir,
                template_name=self._prompt_template,
            )
            # Log the full rendered prompt at INFO so it is debuggable from the
            # backend logs (the prompt is also persisted as the audit transcript).
            logger.info(
                "=> [JUDGE LLM] rendered prompt (%d chars):\n%s",
                len(prompt_text),
                prompt_text,
            )

            # Store the prompt text as the audit transcript inside decision_trace
            updated_trace = decision.decision_trace.model_copy(
                update={"transcript": prompt_text}
            )
            decision = decision.model_copy(update={"decision_trace": updated_trace})

            # Call the LLM with a 30-second timeout to prevent pipeline hanging
            llm_call_started = time.monotonic()
            raw_response = asyncio.run(
                asyncio.wait_for(
                    _invoke_llm_agent(prompt_text, judge_llm_settings),
                    timeout=30.0,
                )
            )
            llm_elapsed_sec = time.monotonic() - llm_call_started

            if not raw_response:
                logger.warning(
                    "=> [JUDGE LLM] empty response after %.1fs; using rule-only decision.",
                    llm_elapsed_sec,
                )
                return decision

            logger.info(
                "=> [JUDGE LLM] response received in %.1fs (%d chars):\n%s",
                llm_elapsed_sec,
                len(raw_response),
                raw_response,
            )

            enriched = _parse_llm_response(raw_response, decision)
            logger.info("=> [JUDGE LLM] enrichment applied successfully.")
            return enriched
        except asyncio.TimeoutError:
            # Surface the timeout explicitly -- this is the most common cause of
            # "judge takes a huge time": a slow/unreachable LLM endpoint.
            logger.warning(
                "=> [JUDGE LLM] timed out after 30s; falling back to rule-only decision. "
                "Check the LLM endpoint/credentials (.env LLM_TYPE/LLM_MODEL/LLM_API_KEY/LLM_GATEWAY_URL)."
            )
            return decision
        except Exception as exc:
            logger.warning(
                "=> [JUDGE LLM] enrichment failed (%s: %s); falling back to rule-only decision.",
                type(exc).__name__,
                exc,
            )
            return decision
