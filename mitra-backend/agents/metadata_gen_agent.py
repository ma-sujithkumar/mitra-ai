import logging
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Optional

import litellm
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.tools import read_mini_data, write_metadata
from config_loader import ConfigLoader

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "metadata_gen.md"
ENV_PATH = Path(__file__).parent.parent.parent / ".env"


class MetadataGenAgent:

    def __init__(self) -> None:
        load_dotenv(ENV_PATH)
        self.llm_type = os.environ.get("LLM_TYPE", "anthropic")
        self.llm_api_key = os.environ.get("LLM_API_KEY", "")
        self.llm_gateway_url = os.environ.get("LLM_GATEWAY_URL", "")
        self.max_retries = ConfigLoader.get_int("metadata_agent", "LLM_MAX_RETRIES")

        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

        litellm_model = self._resolve_litellm_model()

        self._agent = Agent(
            name="metadata_gen_agent",
            model=litellm_model,
            description="Generates metadata.json from mini_data.csv statistics.",
            instruction=system_prompt,
            tools=[read_mini_data, write_metadata],
        )

    def _resolve_litellm_model(self) -> str:
        model_map = {
            "anthropic": "anthropic/claude-sonnet-4-6",
            "openai":    "openai/gpt-4o",
            "gemini":    "gemini/gemini-2.0-flash",
        }
        return model_map.get(self.llm_type, "anthropic/claude-sonnet-4-6")

    async def run(
        self,
        session_id: str,
        description: str,
        target_col: Optional[str],
        problem_type: str,
    ) -> AsyncGenerator[dict, None]:
        classification_threshold = ConfigLoader.get_float(
            "metadata_agent", "CLASSIFICATION_UNIQUE_THRESHOLD"
        )
        categorical_ratio = ConfigLoader.get_float(
            "metadata_agent", "CATEGORICAL_UNIQUE_RATIO"
        )

        user_message = (
            f"Session ID: {session_id}\n"
            f"User description: {description}\n"
            f"Target column: {target_col or '(none - unsupervised)'}\n"
            f"Problem type hint: {problem_type}\n"
            f"CLASSIFICATION_UNIQUE_THRESHOLD: {classification_threshold}\n"
            f"CATEGORICAL_UNIQUE_RATIO: {categorical_ratio}\n\n"
            f"Please generate metadata.json for this session."
        )

        logger.info(f"=> MetadataGenAgent starting for session {session_id}")
        yield {"type": "status", "message": "Metadata agent started"}

        session_service = InMemorySessionService()
        adk_session = await session_service.create_session(
            app_name="mitra_metadata",
            user_id="system",
            session_id=session_id,
        )

        runner = Runner(
            agent=self._agent,
            app_name="mitra_metadata",
            session_service=session_service,
        )

        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        )

        async for event in runner.run_async(
            user_id="system",
            session_id=session_id,
            new_message=user_content,
        ):
            if event.is_final_response():
                yield {
                    "type": "done",
                    "artifact": "metadata.json",
                    "message": event.content.parts[0].text if event.content else "Complete",
                }
            elif hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        yield {"type": "progress", "message": part.text[:200]}
