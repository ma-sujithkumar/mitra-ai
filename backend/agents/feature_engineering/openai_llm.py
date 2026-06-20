"""Google ADK BaseLlm that talks to OpenAI-compatible endpoints (e.g. NVIDIA NIM)
using the OpenAI Python SDK directly. No litellm in the call path.

Handles gpt-oss Harmony tokens by stripping `<|...|>` markers and `to=functions.xxx`
fragments from BOTH function call names and message text content. If those tokens
are allowed into conversation state, they echo back in the next request and the
endpoint's parser rejects the payload with a 400 BadRequestError.
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from pydantic import Field, PrivateAttr

# Harmony format markers used by gpt-oss models. Stripping these from text we
# echo back avoids endpoint parser failures like:
#   'unexpected tokens remaining in message header: Some("to=functions.xxx")'
_HARMONY_TAG_RE = re.compile(r"<\|[^|>]*\|>")
_HARMONY_TO_FN_RE = re.compile(r"\bto=functions\.[A-Za-z_][A-Za-z0-9_]*")
_HARMONY_CHANNEL_RE = re.compile(r"\b(?:channel|recipient|content_type)=[A-Za-z0-9_\-/]+")


def _strip_harmony(text: str | None) -> str | None:
    """Remove Harmony-format control tokens from free-form text content.

    Applied to both incoming model text/reasoning_content (before we store it
    in conversation state) and to outgoing assistant text (defense in depth).
    """
    if not text:
        return text
    cleaned = _HARMONY_TAG_RE.sub("", text)
    cleaned = _HARMONY_TO_FN_RE.sub("", cleaned)
    cleaned = _HARMONY_CHANNEL_RE.sub("", cleaned)
    # Collapse whitespace introduced by removals
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() or None


def _sanitize_function_name(name: str) -> str:
    """Strip Harmony tokens and `functions.` namespace from a function call name.

    gpt-oss Harmony format occasionally leaks channel markers into the function
    name slot — e.g. `functions.create_features_pre..commentary` (the
    `commentary` channel). Our tool names never contain dots, so anything
    after the first `.` is post-namespace garbage and is dropped.
    """
    if not name:
        return name
    cut = name.find("<|")
    if cut != -1:
        name = name[:cut]
    if name.startswith("functions."):
        name = name[len("functions.") :]
    if "." in name:
        name = name.split(".", 1)[0]
    return name.strip(" _.")


def _contents_to_messages(contents, system_instruction: str | None) -> list[dict]:
    """ADK Content list → OpenAI chat messages."""
    messages: list[dict] = []
    if system_instruction:
        messages.append({"role": "system", "content": str(system_instruction)})

    for content in contents or []:
        role = content.role
        parts = content.parts or []
        text_chunks: list[str] = []
        tool_calls: list[dict] = []
        function_responses: list[dict] = []

        for part in parts:
            text = getattr(part, "text", None)
            if text:
                text_chunks.append(text)
            fc = getattr(part, "function_call", None)
            if fc:
                args = fc.args if fc.args else {}
                args_str = args if isinstance(args, str) else json.dumps(args, default=str)
                tool_calls.append(
                    {
                        "id": getattr(fc, "id", None) or f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {"name": fc.name, "arguments": args_str},
                    }
                )
            fr = getattr(part, "function_response", None)
            if fr:
                resp = fr.response if fr.response else {}
                resp_str = resp if isinstance(resp, str) else json.dumps(resp, default=str)
                function_responses.append(
                    {
                        "id": getattr(fr, "id", None),
                        "name": fr.name,
                        "response": resp_str,
                    }
                )

        if role == "user":
            for fr in function_responses:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": fr["id"] or f"resp_{len(messages)}",
                        "content": fr["response"],
                    }
                )
            if text_chunks:
                user_text = _strip_harmony("\n".join(text_chunks))
                if user_text:
                    messages.append({"role": "user", "content": user_text})
        elif role == "model" or role == "assistant":
            msg: dict = {"role": "assistant"}
            assistant_text = _strip_harmony("\n".join(text_chunks)) if text_chunks else None
            msg["content"] = assistant_text
            if tool_calls:
                msg["tool_calls"] = tool_calls
            # Some endpoints reject assistant messages with both content=None and
            # no tool_calls — keep the message only if at least one side is set.
            if msg["content"] is not None or msg.get("tool_calls"):
                messages.append(msg)
        else:
            if text_chunks:
                other_text = _strip_harmony("\n".join(text_chunks))
                if other_text:
                    messages.append({"role": role or "user", "content": other_text})

    return messages


def _tools_to_openai(config) -> list[dict] | None:
    """ADK config.tools → OpenAI tools list."""
    if not config:
        return None
    tools_field = getattr(config, "tools", None) or []
    openai_tools: list[dict] = []
    for tool in tools_field:
        function_decls = getattr(tool, "function_declarations", None) or []
        for decl in function_decls:
            params = getattr(decl, "parameters", None)
            if params is None:
                params_dict: Any = {"type": "object", "properties": {}}
            elif hasattr(params, "model_dump"):
                params_dict = params.model_dump(exclude_none=True)
            else:
                params_dict = params
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": decl.name,
                        "description": decl.description or "",
                        "parameters": params_dict,
                    },
                }
            )
    return openai_tools or None


class OpenAICompatibleLlm(BaseLlm):
    """ADK-native BaseLlm that posts to an OpenAI-compatible endpoint.

    Tools, contents, and responses are translated between the ADK genai schema
    and OpenAI chat-completions schema. Used for NVIDIA NIM and similar
    bring-your-own-endpoint providers.
    """

    api_key: str
    base_url: str | None = None
    max_tokens: int = Field(default=2048)

    _client: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        config = llm_request.config
        system_instruction = getattr(config, "system_instruction", None) if config else None
        # system_instruction may be a Content object — extract text if so
        sys_text: str | None = None
        if system_instruction is not None:
            if isinstance(system_instruction, str):
                sys_text = system_instruction
            else:
                parts = getattr(system_instruction, "parts", None)
                if parts:
                    sys_text = "\n".join(
                        getattr(p, "text", "") or "" for p in parts
                    ).strip() or None

        messages = _contents_to_messages(llm_request.contents, sys_text)
        openai_tools = _tools_to_openai(config)

        max_out = None
        if config is not None:
            max_out = getattr(config, "max_output_tokens", None)
        if not max_out:
            max_out = self.max_tokens

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_out,
            "stream": False,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        completion = self._client.chat.completions.create(**kwargs)
        choices = getattr(completion, "choices", None)
        if not choices:
            err_payload = getattr(completion, "error", None) or getattr(completion, "message", None)
            raise RuntimeError(
                f"OpenAI-compatible endpoint returned no choices. "
                f"model={self.model!r} payload={err_payload!r}"
            )
        choice = choices[0]
        msg = choice.message

        parts: list[genai_types.Part] = []
        text = getattr(msg, "content", None)
        if not text or not str(text).strip():
            text = getattr(msg, "reasoning_content", None)
        # Strip Harmony markup BEFORE it enters conversation state, otherwise it
        # echoes back in the next request and the endpoint rejects with 400.
        text = _strip_harmony(str(text)) if text else None
        if text:
            parts.append(genai_types.Part(text=text))

        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            fn = tc.function
            name = _sanitize_function_name(fn.name)
            try:
                args = json.loads(fn.arguments) if fn.arguments else {}
            except json.JSONDecodeError:
                args = {"raw": fn.arguments}
            parts.append(
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id=getattr(tc, "id", None),
                        name=name,
                        args=args,
                    )
                )
            )

        content = genai_types.Content(role="model", parts=parts)

        yield LlmResponse(
            content=content,
            partial=False,
            turn_complete=True,
            finish_reason=self._map_finish_reason(choice.finish_reason),
        )

    @staticmethod
    def _map_finish_reason(reason: str | None):
        if reason is None:
            return None
        try:
            mapping = {
                "stop": genai_types.FinishReason.STOP,
                "length": genai_types.FinishReason.MAX_TOKENS,
                "tool_calls": genai_types.FinishReason.STOP,
                "function_call": genai_types.FinishReason.STOP,
            }
            return mapping.get(reason, genai_types.FinishReason.OTHER)
        except Exception:
            return None
