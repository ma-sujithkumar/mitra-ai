"""Judge Agent — isolated ADK sub-agent for two decisions.

- FeatureCreator proposal ranking → `rank(...)`.
- FeatureSelector method choice → `plan(...)`.

Runs in its own ADK Agent + InMemoryRunner so its reasoning never enters the
orchestrator's context window. Both entry points use the same validate_response
+ one-revision + fall-through contract used by tool LLM calls.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import numpy as np

from pipeline.evidence import FeatureSelectorEvidence, render
from pipeline.responses import (
    FeatureCreatorResponse,
    SelectionPlanResponse,
    call_with_revision,
    log_raw,
    validate_response,
)

JUDGE_RANK_INSTRUCTION = """You rank candidate feature operations.

Inputs:
- A list of candidate operation specs with proxy mutual-information scores against the target.
- A cap on the number of operations to keep.

Selection mechanics:
- Each spec carries `operation`, `sources`, `name`, `temporal_class`, and a proxy_mi.
- Operations with overlapping `sources` are considered redundant.
- A diverse mix of operation types is preferred over many copies of the same op.

Respond with ONLY a JSON array of the kept `name` values, in priority order, at most `cap` items.
No prose, no markdown.
"""

JUDGE_PLAN_INSTRUCTION = """You produce a feature selection plan, one action per correlation cluster.

Action vocabulary (mechanical descriptions only):
- mrmr: keep features with high MI against the target and low redundancy with each other.
- pca: project the cluster onto a small number of principal components.
- mrmr_then_pca: first mRMR on the significant subset, then PCA over the residual redundant block.
- drop: discard the entire cluster.
- lasso: fit a sparse L1 regression and keep features with nonzero coefficients.
- rf_importance: fit a random forest and keep features by importance.

## RESPONSE SHAPE
Return ONLY a JSON object of this shape:
{{
  "plan": [
    {{
      "cluster_id": <int>,
      "action": "<one of mrmr|pca|mrmr_then_pca|drop|lasso|rf_importance>",
      "rationale": "<at least {min_rationale_chars} characters citing EVIDENCE fields>",
      "evidence_cited": ["<field paths from EVIDENCE you used>"],
      "alternatives_considered": ["<other actions you weighed>"]
    }}
  ]
}}

{evidence_block}
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

    # ---------- ranking ----------

    def _build_rank_prompt(
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
        if not specs:
            return [], "judge"
        if len(specs) <= cap:
            return specs, "judge:no_cap_needed"

        try:
            chosen_names, raw_text = self._call_llm(
                instruction=JUDGE_RANK_INSTRUCTION,
                prompt=self._build_rank_prompt(specs, profile, target_column, task, cap),
                parse_as_array=True,
                return_raw=True,
            )
        except Exception as e:
            log_raw("JudgeAgent.rank", "first", f"<exception: {e}>", "fallback", ["parse"])
            chosen_names = None
            raw_text = ""

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
                log_raw("JudgeAgent.rank", "first", raw_text, "ok", [])
                return kept, "judge"
            log_raw("JudgeAgent.rank", "first", raw_text, "rejected", ["no_known_names"])
        else:
            log_raw("JudgeAgent.rank", "first", raw_text, "rejected", ["parse"])

        ranked = sorted(specs, key=lambda s: _proxy_mi(s, profile), reverse=True)
        return ranked[:cap], "fallback:proxy_mi"

    # ---------- planning ----------

    def plan(self, evidence: FeatureSelectorEvidence, cfg) -> SelectionPlanResponse | None:
        """Return a per-cluster SelectionPlanResponse or None on hard failure."""
        evidence_block, sent_fields = render(evidence, truncate_after_chars=int(self.max_tokens * 0.7 * 4))
        prompt = JUDGE_PLAN_INSTRUCTION.format(
            min_rationale_chars=cfg.validation.min_rationale_chars,
            evidence_block=evidence_block,
        )

        try:
            raw = self._call_llm(
                instruction="You produce a feature selection plan.",
                prompt=prompt,
                parse_as_array=False,
            )
        except Exception as e:
            log_raw("JudgeAgent.plan", "first", f"<exception: {e}>", "fallback", ["parse"])
            return None

        if not raw:
            log_raw("JudgeAgent.plan", "first", "", "fallback", ["empty_response"])
            return None
        parsed, failures = validate_response(SelectionPlanResponse, raw, sent_fields, cfg)
        if parsed is not None and not failures:
            log_raw("JudgeAgent.plan", "first", raw, "ok", [])
            return parsed  # type: ignore[return-value]
        log_raw("JudgeAgent.plan", "first", raw, "rejected", failures)

        revision = (
            prompt
            + "\n\n## REVISION\nprior_response_was_uninformative=true\n"
            + "Failures: " + ", ".join(failures or ["parse"])
            + ".\nProduce a fresh plan with rationale ≥ "
            + f"{cfg.validation.min_rationale_chars} chars per cluster and explicit "
            + "evidence_cited / alternatives_considered fields."
        )
        try:
            raw2 = self._call_llm(
                instruction="You produce a feature selection plan.",
                prompt=revision,
                parse_as_array=False,
            )
        except Exception as e:
            log_raw("JudgeAgent.plan", "revision", f"<exception: {e}>", "fallback", failures or ["parse"])
            return None
        if not raw2:
            log_raw("JudgeAgent.plan", "revision", "", "fallback", ["empty_response"])
            return None
        parsed2, failures2 = validate_response(SelectionPlanResponse, raw2, sent_fields, cfg)
        if parsed2 is not None and not failures2:
            log_raw("JudgeAgent.plan", "revision", raw2, "ok:revised", [])
            return parsed2  # type: ignore[return-value]
        log_raw("JudgeAgent.plan", "revision", raw2, "fallback", failures2 or ["parse"])
        return None

    # ---------- transport ----------

    def _call_llm(
        self,
        instruction: str,
        prompt: str,
        parse_as_array: bool,
        return_raw: bool = False,
    ) -> Any:
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
            description="Judge sub-agent.",
            instruction=instruction,
            tools=[],
        )
        runner = InMemoryRunner(agent=agent, app_name="fe_judge")
        session_id = f"judge_{abs(hash(prompt)) % (10**12)}"

        async def _go() -> str:
            session = await runner.session_service.create_session(
                app_name="fe_judge", user_id="orchestrator", session_id=session_id
            )
            content = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])
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

        raw_text = text or ""
        if not raw_text.strip():
            return (None, raw_text) if return_raw else None
        if parse_as_array:
            try:
                arr = _extract_json_array(raw_text)
            except Exception:
                return (None, raw_text) if return_raw else None
            parsed = [str(x) for x in arr if isinstance(x, (str, int))]
            return (parsed, raw_text) if return_raw else parsed
        return (raw_text, raw_text) if return_raw else raw_text
