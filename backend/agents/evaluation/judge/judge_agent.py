"""
JudgeAgent: orchestrates the full judge pipeline.

Flow:
  1. Rule engine applies deterministic hard gates (floor / bias / leakage) and
     produces a provisional score-ordered ranking among survivors.
  2. If use_llm is enabled, renders a jinja2 prompt and invokes the LLM via
     ADK LlmAgent + LiteLlm to actually RANK the survivors (not just comment
     on a fixed order), grounded in metrics/overfitting/complexity/SHAP/domain
     reasoning fetched via tool calls. The LLM reorders ranked_models; the
     rule-engine score on each RankedModel is left untouched for auditability.
  3. Deterministic selection (RuleEngine.apply_selection) picks the top-N% of
     eligible models from whichever order is authoritative at this point
     (LLM-reordered if it succeeded, rule-only otherwise). The LLM never
     decides selection.
  4. Returns the final JudgeDecision.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

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
    max_accuracy_drop_pct: float,
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
        max_accuracy_drop_pct=max_accuracy_drop_pct,
    )


class JudgeTools:
    """Mandatory tools connected to the Judge Agent to fetch metadata, statistics and model evaluation details dynamically."""

    def __init__(self, judge_input: JudgeInput) -> None:
        self._judge_input = judge_input

    def get_dataset_metadata(self) -> dict[str, Any]:
        """Retrieve user-provided metadata for the dataset, including description and target column."""
        return self._judge_input.metadata or {}

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Retrieve descriptive statistics (minidata / pd.describe metrics) of the validation dataset."""
        return self._judge_input.minidata or {}

    def get_domain_reasoning(self) -> dict[str, Any]:
        """Retrieve the domain-reasoning agent's problem/column explanations and leakage-risk flags.

        Returns an empty dict (never raises, never hallucinated) when the
        domain-reasoning agent did not run or failed for this session.
        """
        return self._judge_input.domain_reasoning or {}

    def get_model_evaluation_details(self, model_name: str) -> dict[str, Any]:
        """Retrieve detailed validation metrics, overfitting checks, CV results, diagnostics, complexity, and SHAP features for a specific model candidate."""
        for candidate in self._judge_input.candidates:
            if candidate.model_name == model_name:
                res = {
                    "model_name": candidate.model_name,
                    "task_type": candidate.task_type,
                    "metrics": candidate.metrics,
                    "complexity": {
                        "n_params": candidate.complexity.n_params,
                        "depth": candidate.complexity.depth,
                        "family_rank": candidate.complexity.family_rank,
                    }
                }
                if candidate.overfitting:
                    res["overfitting"] = {
                        "is_overfitted": candidate.overfitting.is_overfitted,
                        "gap": candidate.overfitting.gap,
                        "train_vs_cv_gap": candidate.overfitting.train_vs_cv_gap,
                        "train_metrics": candidate.overfitting.train_metrics,
                        "test_metrics": candidate.overfitting.test_metrics,
                        "cv_results": candidate.overfitting.cv_results,
                        "diagnostics": candidate.overfitting.diagnostics,
                    }
                if candidate.shap_summary:
                    res["shap_summary"] = {
                        "top_features": candidate.shap_summary.get("top_features", []),
                        "mean_abs_shap": candidate.shap_summary.get("mean_abs_shap", {}),
                        "feature_concentration": candidate.shap_summary.get("feature_concentration"),
                    }
                if getattr(candidate, "hyperparam_sensitivity", None):
                    res["hyperparam_sensitivity"] = candidate.hyperparam_sensitivity
                return res
        return {"error": f"Model '{model_name}' not found."}


async def _invoke_llm_agent(
    prompt_text: str,
    llm_settings: LlmSettings,
    tools_instance: JudgeTools,
    tool_call_callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> Optional[str]:
    """Run the ADK LlmAgent via the shared LiteLlm factory, using connected tools to fetch data on demand."""
    llm_model = build_llm_model(llm_settings)
    agent = LlmAgent(
        name="judge_llm",
        model=llm_model,
        instruction=(
            "You are an expert ML model selection judge. The candidates have already "
            "passed a deterministic hard floor and bias/leakage gate -- that filtering "
            "is fixed. Your job is to RANK the survivors by judgment, grounded in their "
            "metrics, overfitting behavior, complexity, and SHAP feature importance "
            "correlated against domain reasoning. You do NOT decide which models are "
            "selected -- a separate deterministic step picks the top percentage of your "
            "ranking. To construct your response, you MUST call the tools to retrieve "
            "dataset metadata, statistics, domain reasoning, and model evaluation "
            "details for every candidate. Do not assume or guess. Respond only with the "
            "requested JSON ranking schema."
        ),
        tools=[
            tools_instance.get_dataset_metadata,
            tools_instance.get_dataset_statistics,
            tools_instance.get_domain_reasoning,
            tools_instance.get_model_evaluation_details,
        ],
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
        function_calls = event.get_function_calls() if hasattr(event, "get_function_calls") else []
        for call in function_calls:
            if tool_call_callback:
                tool_call_callback(call.name, call.args)

        if event.is_final_response() and event.content:
            for part in event.content.parts or []:
                if hasattr(part, "text") and part.text:
                    response_text_parts.append(part.text)

    return "".join(response_text_parts) if response_text_parts else None


# Exceptions that warrant retrying the LLM ranking call: transient timeouts,
# and response-shape failures (bad JSON, or a ranking that doesn't cover
# exactly the survivor set). Anything else (e.g. credential/auth errors) is
# not retried -- retrying won't fix it.
_RETRYABLE_LLM_RANKING_EXCEPTIONS = (asyncio.TimeoutError, json.JSONDecodeError, ValueError)


def _parse_llm_ranking_response(
    raw_response: str,
    decision: JudgeDecision,
) -> JudgeDecision:
    """Parse the LLM's JSON ranking response and reorder survivors by it.

    Validates that the LLM ranked exactly the survivor set (verdict=="select"
    entries -- selection has not been applied yet at this point in judge()).
    On any mismatch this raises ValueError, which the caller treats as a
    retryable parse failure rather than silently dropping models.

    The rule-engine `score` on each RankedModel is left untouched for
    auditability; only `rank` (renumbered to match the LLM's order),
    `llm_flags`, and `llm_ranking_reasoning` change. Gate-rejected models are
    never touched by the LLM and keep their original verdict/rank ordering
    relative to each other, renumbered to continue after the survivors.
    """
    raw_stripped = raw_response.strip()
    # Strip markdown code fences if the model wrapped the JSON.
    if raw_stripped.startswith("```"):
        lines = raw_stripped.splitlines()
        raw_stripped = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    parsed: Dict[str, Any] = json.loads(raw_stripped)
    ranking_entries: List[Dict[str, Any]] = parsed.get("ranking", [])
    overall_commentary = parsed.get("overall_commentary", "")

    survivors = [rm for rm in decision.ranked_models if rm.verdict == "select"]
    rejected = [rm for rm in decision.ranked_models if rm.verdict != "select"]
    survivor_names = {rm.model_name for rm in survivors}
    llm_ranked_names = [entry.get("model_name") for entry in ranking_entries]

    if len(llm_ranked_names) != len(survivors) or set(llm_ranked_names) != survivor_names:
        raise ValueError(
            f"LLM ranking covers {sorted(set(llm_ranked_names))} but survivors "
            f"are {sorted(survivor_names)} -- treating as a parse failure."
        )

    survivors_by_name: Dict[str, RankedModel] = {rm.model_name: rm for rm in survivors}
    reordered_survivors: List[RankedModel] = []
    for new_rank, entry in enumerate(ranking_entries, start=1):
        original = survivors_by_name[entry["model_name"]]
        reasoning_parts = [
            part for part in (entry.get("reasoning"), entry.get("shap_domain_correlation"))
            if part
        ]
        reordered_survivors.append(
            original.model_copy(
                update={
                    "rank": new_rank,
                    "llm_flags": entry.get("flags", []),
                    "llm_ranking_reasoning": "\n".join(reasoning_parts) or None,
                }
            )
        )

    reject_rank_start = len(reordered_survivors) + 1
    renumbered_rejected = [
        ranked_model.model_copy(update={"rank": reject_rank_start + offset})
        for offset, ranked_model in enumerate(rejected)
    ]

    updated_trace = decision.decision_trace.model_copy(
        update={
            "llm_commentary": overall_commentary,
            "llm_ranking_status": "applied",
            "llm_ranking_error": None,
        }
    )
    return decision.model_copy(
        update={
            "ranked_models": reordered_survivors + renumbered_rejected,
            "decision_trace": updated_trace,
            "selected_model": (
                reordered_survivors[0].model_name if reordered_survivors else decision.selected_model
            ),
        }
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
        status_callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
    ) -> JudgeDecision:
        """Run the full judge pipeline and return a JudgeDecision.

        Args:
            judge_input: Structured input with all candidate models.
            use_llm: Override config use_llm setting. Pass False for rule-only mode.
            status_callback: Optional callback to track live status and tool calls.

        Returns:
            JudgeDecision with rule-authoritative verdicts and optional LLM enrichment.
        """
        should_use_llm = use_llm if use_llm is not None else bool(self._config.get("use_llm", True))

        logger.debug(
            "=> JudgeAgent.judge: %d candidates, use_llm=%s",
            len(judge_input.candidates),
            should_use_llm,
        )

        # Step 1: deterministic gating and provisional score-ordered ranking.
        survivors, gate_outcomes = self._rule_engine.apply_hard_gates(judge_input.candidates)
        decision = self._rule_engine.rank(
            survivors=survivors,
            gate_outcomes=gate_outcomes,
            all_candidates=judge_input.candidates,
        )
        decision = decision.model_copy(update={"dataset_id": judge_input.dataset_id})

        # Step 2: LLM ranking (reorders survivors; never decides selection).
        # llm_ranking_status is always set so a missing value never means
        # "silently swallowed" -- it's explicitly "skipped" when use_llm=False.
        if should_use_llm:
            decision = self._enrich_with_llm(judge_input, decision, status_callback)
        else:
            skipped_trace = decision.decision_trace.model_copy(
                update={"llm_ranking_status": "skipped"}
            )
            decision = decision.model_copy(update={"decision_trace": skipped_trace})

        # Step 3: deterministic top-N% selection, always last, always plain
        # Python -- operates on whichever order is authoritative at this point.
        decision = self._rule_engine.apply_selection(decision)

        logger.debug(
            "=> JudgeAgent decision: selected_models=%s total_ranked=%d",
            decision.selected_models,
            len(decision.ranked_models),
        )
        return decision

    def _enrich_with_llm(
        self,
        judge_input: JudgeInput,
        decision: JudgeDecision,
        status_callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
    ) -> JudgeDecision:
        """Invoke the LLM to RANK survivors, retrying on transient failures.

        On success, reorders ranked_models (survivors only) by the LLM's
        ranking and sets decision_trace.llm_ranking_status="applied". On
        exhausting retries, decision_trace.llm_ranking_status is set to
        "failed" with the error recorded -- never silently falls back to an
        unmarked rule-only decision (the bug this rewrite closes).
        """
        # Opt-in litellm wire-level debug (judge prompts/responses in backend logs),
        # matching the other agents. Driven by judge config so it is not hardcoded.
        if self._config.get("litellm_debug", False):
            os.environ["LITELLM_LOG"] = "DEBUG"

        try:
            # Resolve credentials the SAME way every other agent does: via
            # LlmSettingsResolver, which reads .env (LLM_TYPE/LLM_MODEL/LLM_API_KEY).
            judge_llm_settings = LlmSettingsResolver(ConfigLoader()).resolve()
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
                max_accuracy_drop_pct=float(self._config["llm_ranking_max_accuracy_drop"]) * 100,
            )
            logger.info(
                "=> [JUDGE LLM] rendered prompt (%d chars):\n%s",
                len(prompt_text),
                prompt_text,
            )

            # Store the prompt text as the audit transcript inside decision_trace.
            updated_trace = decision.decision_trace.model_copy(
                update={"transcript": prompt_text}
            )
            decision = decision.model_copy(update={"decision_trace": updated_trace})

            tools_instance = JudgeTools(judge_input)
            active_calls: List[Dict[str, Any]] = []

            def tool_call_callback(name: str, args: dict[str, Any]) -> None:
                call_info = {
                    "name": name,
                    "args": args,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                active_calls.append(call_info)
                logger.info("=> [JUDGE LLM] tool call: %s with args %s", name, args)
                if status_callback:
                    status_callback(
                        f"Agent calling tool '{name}'...",
                        {"tool_calls": active_calls}
                    )

            max_attempts = int(self._config.get("llm_judge_max_retries", 2)) + 1
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    llm_call_started = time.monotonic()
                    raw_response = asyncio.run(
                        asyncio.wait_for(
                            _invoke_llm_agent(
                                prompt_text=prompt_text,
                                llm_settings=judge_llm_settings,
                                tools_instance=tools_instance,
                                tool_call_callback=tool_call_callback,
                            ),
                            timeout=30.0,
                        )
                    )
                    llm_elapsed_sec = time.monotonic() - llm_call_started

                    if not raw_response:
                        raise ValueError(
                            f"empty LLM response after {llm_elapsed_sec:.1f}s"
                        )

                    logger.info(
                        "=> [JUDGE LLM] response received in %.1fs (%d chars):\n%s",
                        llm_elapsed_sec,
                        len(raw_response),
                        raw_response,
                    )

                    enriched = _parse_llm_ranking_response(raw_response, decision)
                    # Deterministic guardrail: the LLM cannot rank a model
                    # above another with a much better primary metric just
                    # because SHAP/domain signal looks cleaner.
                    enriched = self._rule_engine.enforce_accuracy_reorder_guardrail(
                        enriched, judge_input.candidates
                    )
                    logger.info(
                        "=> [JUDGE LLM] ranking applied successfully (attempt %d/%d).",
                        attempt,
                        max_attempts,
                    )
                    return enriched
                except _RETRYABLE_LLM_RANKING_EXCEPTIONS as exc:
                    last_exc = exc
                    will_retry = attempt < max_attempts
                    logger.warning(
                        "=> [JUDGE LLM] attempt %d/%d failed (%s: %s)%s",
                        attempt,
                        max_attempts,
                        type(exc).__name__,
                        exc,
                        "; retrying..." if will_retry else "; exhausted retries.",
                    )

            # Exhausted retries on a retryable failure: surface it visibly
            # instead of silently returning an unmarked rule-only decision.
            failed_trace = decision.decision_trace.model_copy(
                update={
                    "llm_ranking_status": "failed",
                    "llm_ranking_error": (
                        f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown error"
                    ),
                }
            )
            return decision.model_copy(update={"decision_trace": failed_trace})
        except Exception as exc:
            # Non-retryable failure (e.g. missing/invalid LLM credentials):
            # also surfaced visibly, never a silent fallback.
            logger.warning(
                "=> [JUDGE LLM] ranking failed non-retryably (%s: %s); falling back to rule-only order.",
                type(exc).__name__,
                exc,
            )
            failed_trace = decision.decision_trace.model_copy(
                update={
                    "llm_ranking_status": "failed",
                    "llm_ranking_error": f"{type(exc).__name__}: {exc}",
                }
            )
            return decision.model_copy(update={"decision_trace": failed_trace})
