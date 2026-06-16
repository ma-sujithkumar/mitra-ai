"""
ClaudeAdkLlm: ADK BaseLlm subclass that delegates LLM calls to CustomAnthropicClient.

This bridges Google ADK's LlmAgent framework with the custom_anthropic_client
that routes through the local `claude` CLI. Translation happens in both directions:
  ADK LlmRequest  (google.genai.types.Content list)  => Anthropic messages format
  Anthropic response (_Message / content blocks)     => ADK LlmResponse

This lets the judge agent be built entirely on ADK (SPEC constraint #1) while
using Claude as the underlying LLM via the local CLI (no API key required).
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, List, Optional

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from custom_anthropic_client import CustomAnthropicClient

logger = logging.getLogger(__name__)

# Sentinel role name used by ADK for system-level instructions.
ADK_SYSTEM_ROLE = "system"


def _content_to_anthropic_message(content: genai_types.Content) -> Optional[dict]:
    """Convert a single genai Content object to an Anthropic messages dict.

    Returns None for system-role content (handled separately as `system=`).
    """
    role = getattr(content, "role", "user") or "user"
    if role == ADK_SYSTEM_ROLE:
        return None

    # Map ADK roles to Anthropic roles.
    anthropic_role = "assistant" if role == "model" else "user"

    parts = getattr(content, "parts", []) or []
    text_parts: List[str] = []
    for part in parts:
        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)
        elif hasattr(part, "function_call") and part.function_call:
            # Tool call from ADK: render as JSON so the model sees it.
            func_call = part.function_call
            text_parts.append(
                json.dumps(
                    {
                        "function_call": {
                            "name": func_call.name,
                            "args": dict(func_call.args or {}),
                        }
                    }
                )
            )
        elif hasattr(part, "function_response") and part.function_response:
            func_resp = part.function_response
            text_parts.append(
                json.dumps(
                    {
                        "function_response": {
                            "name": func_resp.name,
                            "response": dict(func_resp.response or {}),
                        }
                    }
                )
            )
    return {"role": anthropic_role, "content": "\n".join(text_parts)}


def _extract_system_instruction(llm_request: LlmRequest) -> Optional[str]:
    """Pull system-role content out of an LlmRequest as a plain string."""
    system_instruction = getattr(llm_request, "system_instruction", None)
    if system_instruction:
        parts = getattr(system_instruction, "parts", []) or []
        texts = [part.text for part in parts if hasattr(part, "text") and part.text]
        if texts:
            return "\n".join(texts)

    # Fallback: look for system role in contents list.
    for content in (llm_request.contents or []):
        if getattr(content, "role", "") == ADK_SYSTEM_ROLE:
            parts = getattr(content, "parts", []) or []
            texts = [part.text for part in parts if hasattr(part, "text") and part.text]
            if texts:
                return "\n".join(texts)
    return None


def _anthropic_response_to_llm_response(anthropic_message: object) -> LlmResponse:
    """Convert a CustomAnthropicClient _Message to an ADK LlmResponse."""
    content_blocks = getattr(anthropic_message, "content", [])
    stop_reason = getattr(anthropic_message, "stop_reason", "end_turn")

    parts: List[genai_types.Part] = []
    for block in content_blocks:
        block_type = getattr(block, "type", "text")
        if block_type == "text":
            parts.append(genai_types.Part(text=block.text))
        elif block_type == "tool_use":
            parts.append(
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        name=block.name,
                        args=dict(block.input or {}),
                    )
                )
            )

    response_content = genai_types.Content(role="model", parts=parts)
    # ADK LlmResponse: content holds the model turn; partial=False for a complete response.
    return LlmResponse(content=response_content, partial=False)


class ClaudeAdkLlm(BaseLlm):
    """ADK BaseLlm that delegates to CustomAnthropicClient (claude CLI backend).

    Register with an ADK LlmAgent as:
        agent = LlmAgent(model=ClaudeAdkLlm(model="claude-sonnet"), ...)
    """

    @classmethod
    def supported_models(cls) -> List[str]:
        """Regex patterns matching Claude model names this wrapper handles."""
        return [r"claude.*"]

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        """Translate the ADK request to Anthropic format, call the CLI, translate back.

        Yields a single complete LlmResponse (non-streaming; the CLI is sync).
        """
        system_text = _extract_system_instruction(llm_request)
        anthropic_messages = []
        for content in (llm_request.contents or []):
            converted = _content_to_anthropic_message(content)
            if converted is not None:
                anthropic_messages.append(converted)

        if not anthropic_messages:
            # Ensure at least one user message so the API contract is satisfied.
            anthropic_messages = [{"role": "user", "content": "Please respond."}]

        # Build Anthropic tool schemas from ADK tool declarations if present.
        anthropic_tools = None
        adk_tools = getattr(llm_request, "tools", None) or []
        if adk_tools:
            anthropic_tools = []
            for tool in adk_tools:
                for func_decl in (getattr(tool, "function_declarations", None) or []):
                    anthropic_tools.append(
                        {
                            "name": func_decl.name,
                            "description": func_decl.description or "",
                            "input_schema": {
                                "type": "object",
                                "properties": dict(
                                    getattr(func_decl.parameters, "properties", {}) or {}
                                ),
                            },
                        }
                    )

        logger.debug(
            "=> ClaudeAdkLlm: sending %d messages to CLI (system=%s tools=%s)",
            len(anthropic_messages),
            bool(system_text),
            bool(anthropic_tools),
        )

        # CustomAnthropicClient.messages.create() is synchronous; run in thread.
        client = CustomAnthropicClient()
        anthropic_response = await asyncio.to_thread(
            client.messages.create,
            max_tokens=2048,
            messages=anthropic_messages,
            system=system_text,
            tools=anthropic_tools,
        )

        adk_response = _anthropic_response_to_llm_response(anthropic_response)
        logger.debug(
            "=> ClaudeAdkLlm: received response stop_reason=%s parts=%d",
            getattr(anthropic_response, "stop_reason", "unknown"),
            len(adk_response.content.parts if adk_response.content else []),
        )
        yield adk_response
