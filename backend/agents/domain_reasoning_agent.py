from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from backend.agents.metadata_gen_agent import LlmSettings
from backend.agents.tools import DomainReasoningTools
from backend.agents.tools import MetadataTools

logger = logging.getLogger("mitra.domain_reasoning_agent")


@dataclass(frozen=True)
class DomainReasoningInput:
    session_id: str
    workspace_root: Path
    llm_settings: LlmSettings


@dataclass(frozen=True)
class DomainReasoningResult:
    domain_reasoning: dict[str, Any]
    domain_reasoning_path: Path


class DomainReasoningError(RuntimeError):
    pass


class DomainReasoningToolAdapter:
    def __init__(
        self,
        domain_reasoning_tools: DomainReasoningTools,
        metadata_tools: MetadataTools,
    ) -> None:
        self.domain_reasoning_tools = domain_reasoning_tools
        self.metadata_tools = metadata_tools

    def read_mini_data(self, session_id: str) -> str:
        # Reuse the existing mini_data.csv reader instead of duplicating the
        # path-resolution logic already in MetadataTools.
        logger.info("tool read_mini_data called: session_id=%s", session_id)
        mini_data_text = self.metadata_tools.read_mini_data(session_id=session_id)
        logger.info(
            "tool read_mini_data returned: session_id=%s chars=%d",
            session_id,
            len(mini_data_text),
        )
        return mini_data_text

    def read_metadata(self, session_id: str) -> dict[str, Any]:
        logger.info("tool read_metadata called: session_id=%s", session_id)
        metadata = self.domain_reasoning_tools.read_metadata(session_id=session_id)
        logger.info(
            "tool read_metadata returned: session_id=%s keys=%s",
            session_id,
            sorted(metadata.keys()),
        )
        return metadata

    def write_domain_reasoning(
        self,
        session_id: str,
        domain_reasoning: dict[str, Any],
    ) -> dict[str, str]:
        logger.info(
            "tool write_domain_reasoning called: session_id=%s keys=%s",
            session_id,
            sorted(domain_reasoning.keys())
            if isinstance(domain_reasoning, dict)
            else "(string payload)",
        )
        result = self.domain_reasoning_tools.write_domain_reasoning(
            session_id=session_id,
            domain_reasoning=domain_reasoning,
        )
        logger.info(
            "tool write_domain_reasoning wrote: session_id=%s path=%s",
            session_id,
            result.domain_reasoning_path,
        )
        return {
            "session_id": result.session_id,
            "domain_reasoning_path": str(result.domain_reasoning_path),
        }


class DomainReasoningAgent:
    def __init__(
        self,
        llm_settings: LlmSettings,
        domain_reasoning_tools: DomainReasoningTools,
        metadata_tools: MetadataTools,
        prompt_path: Path | None = None,
    ) -> None:
        self.llm_settings = llm_settings
        self.domain_reasoning_tools = domain_reasoning_tools
        self.domain_reasoning_agent_tools = DomainReasoningToolAdapter(
            domain_reasoning_tools=domain_reasoning_tools,
            metadata_tools=metadata_tools,
        )
        self.prompt_path = (
            prompt_path
            or Path(__file__).resolve().parent / "prompts" / "domain_reasoning.md"
        )
        self.instruction = self.prompt_path.read_text(encoding="utf-8")
        self.agent = self._build_agent()

    def _build_agent(self) -> LlmAgent:
        lite_llm_kwargs: dict[str, str] = {}
        if self.llm_settings.api_key:
            lite_llm_kwargs["api_key"] = self.llm_settings.api_key
        effective_gateway_url = self.llm_settings.effective_gateway_url()
        if effective_gateway_url:
            lite_llm_kwargs["api_base"] = effective_gateway_url

        model = LiteLlm(
            model=self.llm_settings.model,
            **lite_llm_kwargs,
        )
        return LlmAgent(
            name="domain_reasoning_agent",
            description=(
                "Explains column semantics and the prediction problem from "
                "mini_data.csv and metadata.json, flagging post-decision "
                "(leakage-risk) columns for the Judge agent."
            ),
            model=model,
            instruction=self.instruction,
            tools=[
                self.domain_reasoning_agent_tools.read_mini_data,
                self.domain_reasoning_agent_tools.read_metadata,
                self.domain_reasoning_agent_tools.write_domain_reasoning,
            ],
        )


class DomainReasoningAgentRunner:
    app_name = "mitra_domain_reasoning"
    user_id = "mitra_epic7"

    def generate_domain_reasoning(
        self,
        generation_input: DomainReasoningInput,
    ) -> DomainReasoningResult:
        logger.info(
            "generate_domain_reasoning start: session_id=%s provider=%s model=%s gateway=%s",
            generation_input.session_id,
            generation_input.llm_settings.provider,
            generation_input.llm_settings.model,
            generation_input.llm_settings.gateway_url or "(none)",
        )
        domain_reasoning_tools = DomainReasoningTools(
            workspace_root=generation_input.workspace_root,
        )
        metadata_tools = MetadataTools(workspace_root=generation_input.workspace_root)
        domain_reasoning_agent = DomainReasoningAgent(
            llm_settings=generation_input.llm_settings,
            domain_reasoning_tools=domain_reasoning_tools,
            metadata_tools=metadata_tools,
        )
        session_service = InMemorySessionService()
        session_service.create_session_sync(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=generation_input.session_id,
        )
        runner = Runner(
            app_name=self.app_name,
            agent=domain_reasoning_agent.agent,
            session_service=session_service,
        )
        message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=self._build_user_message(generation_input=generation_input)
                )
            ],
        )

        logger.info(
            "entering agent run loop: session_id=%s", generation_input.session_id
        )
        asyncio.run(
            self._drain_agent_events(
                runner=runner,
                session_id=generation_input.session_id,
                message=message,
            )
        )
        logger.info(
            "agent run loop finished: session_id=%s", generation_input.session_id
        )

        domain_reasoning_path = (
            generation_input.workspace_root
            / generation_input.session_id
            / "reports"
            / "domain_reasoning.json"
        )
        if not domain_reasoning_path.is_file():
            logger.error(
                "domain_reasoning.json missing after run: session_id=%s expected_path=%s",
                generation_input.session_id,
                domain_reasoning_path,
            )
            raise DomainReasoningError(
                "Domain reasoning agent did not write domain_reasoning.json"
            )

        logger.info(
            "domain_reasoning.json found: session_id=%s path=%s",
            generation_input.session_id,
            domain_reasoning_path,
        )
        domain_reasoning = json.loads(domain_reasoning_path.read_text(encoding="utf-8"))
        return DomainReasoningResult(
            domain_reasoning=domain_reasoning,
            domain_reasoning_path=domain_reasoning_path,
        )

    async def _drain_agent_events(
        self,
        runner: Runner,
        session_id: str,
        message: types.Content,
    ) -> None:
        event_count = 0
        async for event in runner.run_async(
            user_id=self.user_id,
            session_id=session_id,
            new_message=message,
        ):
            event_count += 1
            self._log_agent_event(
                session_id=session_id,
                event_index=event_count,
                event=event,
            )
        logger.info(
            "agent emitted %d events: session_id=%s", event_count, session_id
        )

    @staticmethod
    def _log_agent_event(session_id: str, event_index: int, event: Any) -> None:
        author = getattr(event, "author", "unknown")
        function_calls = event.get_function_calls() if hasattr(event, "get_function_calls") else []
        function_responses = (
            event.get_function_responses() if hasattr(event, "get_function_responses") else []
        )
        text_parts = []
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)
        logger.info(
            "agent event %d: session_id=%s author=%s tool_calls=%s tool_responses=%s "
            "is_final=%s text=%r",
            event_index,
            session_id,
            author,
            [call.name for call in function_calls],
            [response.name for response in function_responses],
            event.is_final_response() if hasattr(event, "is_final_response") else "n/a",
            (" ".join(text_parts))[:300],
        )

    @staticmethod
    def _build_user_message(generation_input: DomainReasoningInput) -> str:
        return "\n".join(
            [
                f"session_id: {generation_input.session_id}",
                "",
                "Read metadata.json and mini_data for this session, infer the "
                "domain meaning of each input column and the problem being "
                "solved, flag post-decision/leakage-risk columns, and call "
                "write_domain_reasoning with the final JSON object.",
            ]
        )
