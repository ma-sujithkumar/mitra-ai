"""Judge Agent — an isolated ADK sub-agent that ranks FeatureCreator proposals.

Spec ref: Fe_Spec.md §7 Solution F. The Judge Agent inspects the candidate
feature operations proposed by FeatureCreator and returns a capped, ranked
subset. It runs in its own ADK Agent + Runner so its reasoning never enters
the orchestrator's context window.

Falls back to proxy-MI ranking if the LLM call or response parsing fails.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import numpy as np

JUDGE_INSTRUCTION = """You are the Judge Agent for a feature engineering pipeline.

You receive a list of candidate feature operations proposed by FeatureCreator,
each with a proxy mutual-information score against the target. Your job: choose
the best subset (at most `cap` items) that maximises relevance to the target
while avoiding redundancy across proposals.

Selection rules:
- Prefer operations whose source columns have higher mi_with_target.
- Penalise proposals whose sources overlap heavily with proposals you have
  already selected (redundancy).
- Prefer a diverse mix of operation types over many copies of the same op.
- Discard proposals whose name or sources look duplicative.

Respond with ONLY a JSON array of the `name` field of the kept proposals,
in priority order. No prose, no markdown, no commentary.
Example: ["price_per_sqft", "rooms_x_quality", "age_squared"]
"""


def _extract_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array in judge response: {text[:200]!r}")
    return json.loads(m.group(0))


def _proxy_mi(spec: dict, profile: dict) -> float:
    scores = [profile.get(s, {}).get("mi_with_target") or 0.0 for s in spec.get("sources", [])]
    return float(np.mean(scores)) if scores else 0.0


class JudgeAgent:
    """ADK sub-agent for ranking FeatureCreator proposals.

    Constructed once at orchestrator startup; reused for each FeatureCreator
    invocation. Talks to the same OpenAI-compatible endpoint as the
    orchestrator but through its own Agent + InMemoryRunner so context is
    isolated.
    """

    def __init__(
        self,
        model_string: str,
        api_key: str,
        base_url: str | None,
        max_tokens: int,
    ):
        self.model_string = model_string
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens

    def _build_prompt(
        self,
        specs: list[dict],
        profile: dict,
        target_column: str,
        task: str,
        cap: int,
    ) -> str:
        lines = []
        for i, s in enumerate(specs):
            mi = _proxy_mi(s, profile)
            lines.append(
                f"{i}. name={s.get('name')!r} op={s.get('operation')} "
                f"sources={s.get('sources')} temporal_class={s.get('temporal_class')} "
                f"proxy_mi={mi:.4f}"
            )
        return (
            f"target_column={target_column}\n"
            f"task={task}\n"
            f"cap={cap}\n"
            f"candidates:\n" + "\n".join(lines) + "\n\n"
            f"Return at most {cap} names as a JSON array."
        )

    def rank(
        self,
        specs: list[dict],
        profile: dict,
        target_column: str,
        task: str,
        cap: int,
    ) -> tuple[list[dict], str]:
        """Rank and cap proposals. Returns (kept_specs, source) where source is
        either 'judge' (LLM verdict applied) or 'fallback:proxy_mi' (LLM
        unavailable or response malformed)."""
        if not specs:
            return [], "judge"
        if len(specs) <= cap:
            return specs, "judge:no_cap_needed"

        try:
            chosen_names = self._call_llm(specs, profile, target_column, task, cap)
        except Exception:
            chosen_names = None

        if chosen_names:
            name_to_spec = {s["name"]: s for s in specs}
            kept: list[dict] = []
            seen: set[str] = set()
            for nm in chosen_names:
                if nm in name_to_spec and nm not in seen:
                    kept.append(name_to_spec[nm])
                    seen.add(nm)
                if len(kept) >= cap:
                    break
            if kept:
                return kept, "judge"

        # Fallback: proxy MI ranking
        ranked = sorted(specs, key=lambda s: _proxy_mi(s, profile), reverse=True)
        return ranked[:cap], "fallback:proxy_mi"

    def _call_llm(
        self,
        specs: list[dict],
        profile: dict,
        target_column: str,
        task: str,
        cap: int,
    ) -> list[str] | None:
        from google.adk.agents import Agent
        from google.adk.runners import InMemoryRunner
        from google.genai import types as genai_types

        from pipeline.openai_llm import OpenAICompatibleLlm

        llm = OpenAICompatibleLlm(
            model=self.model_string,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
        )
        agent = Agent(
            name="feature_judge",
            model=llm,
            description="Ranks candidate feature operations.",
            instruction=JUDGE_INSTRUCTION,
            tools=[],
        )
        runner = InMemoryRunner(agent=agent, app_name="fe_judge")
        prompt = self._build_prompt(specs, profile, target_column, task, cap)
        session_id = f"judge_{abs(hash(prompt)) % (10**12)}"

        async def _go() -> str:
            session = await runner.session_service.create_session(
                app_name="fe_judge", user_id="orchestrator", session_id=session_id
            )
            content = genai_types.Content(
                role="user", parts=[genai_types.Part(text=prompt)]
            )
            chunks: list[str] = []
            async for event in runner.run_async(
                user_id="orchestrator", session_id=session.id, new_message=content
            ):
                ec = getattr(event, "content", None)
                if ec and ec.parts:
                    for p in ec.parts:
                        t = getattr(p, "text", None)
                        if t:
                            chunks.append(t)
            return "".join(chunks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        text = loop.run_until_complete(_go())

        if not text or not text.strip():
            return None
        try:
            arr = _extract_json_array(text)
        except Exception:
            return None
        return [str(x) for x in arr if isinstance(x, (str, int))]
