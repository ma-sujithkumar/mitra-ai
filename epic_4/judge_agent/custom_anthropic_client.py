"""
Vendored stub of custom_anthropic_client.py.

This file is a compatible re-implementation of the interface described at
https://github.com/ma-sujithkumar/custom_anthropic_client

It provides a drop-in replacement for anthropic.Anthropic() that routes
calls through the local `claude` CLI instead of requiring an API key.

IMPORTANT: Replace this file with the real custom_anthropic_client.py from
the upstream repository when available. The public interface (class names,
method signatures, and response schema) must remain identical.

Environment variables required:
    CLAUDE_CLI_PATH      - Path to the claude binary (e.g. /usr/local/bin/claude)
    ANTHROPIC_MODEL_NAME - Model name alias (e.g. haiku, sonnet, opus)
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class _TextBlock:
    """Mimics anthropic.types.TextBlock."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    """Mimics anthropic.types.ToolUseBlock."""

    def __init__(self, name: str, input_data: Dict[str, Any], block_id: str) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = input_data
        self.id = block_id


class _Message:
    """Mimics anthropic.types.Message."""

    def __init__(self, content: List[Any], stop_reason: str) -> None:
        self.content = content
        self.stop_reason = stop_reason


class _MessagesResource:
    """Mimics the client.messages namespace."""

    def __init__(self, cli_path: str, model_name: str) -> None:
        self._cli_path = cli_path
        self._model_name = model_name

    def create(
        self,
        max_tokens: int,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> _Message:
        """Call the claude CLI and return an Anthropic-schema response object.

        Flattens the system prompt, tool schemas, and message history into a
        single prompt string, invokes the claude binary, and parses the output
        back into SDK-compatible response objects.
        """
        prompt_parts: List[str] = []
        if system:
            prompt_parts.append(f"SYSTEM:\n{system}")
        if tools:
            prompt_parts.append(f"AVAILABLE TOOLS:\n{json.dumps(tools, indent=2)}")
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                # Already a list of blocks; flatten to text.
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif isinstance(block, dict):
                        text_parts.append(json.dumps(block))
                    else:
                        text_parts.append(str(block))
                content = "\n".join(text_parts)
            prompt_parts.append(f"{role.upper()}:\n{content}")

        full_prompt = "\n\n".join(prompt_parts)
        logger.debug("=> Invoking claude CLI with model=%s", self._model_name)

        result = subprocess.run(
            [self._cli_path, "--model", self._model_name, "-p", full_prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited with code {result.returncode}: {result.stderr.strip()}"
            )

        raw_output = result.stdout.strip()
        logger.debug("=> claude CLI raw response length=%d", len(raw_output))

        # Attempt to parse tool_use blocks if the output starts with JSON.
        content_blocks: List[Any] = []
        stop_reason = "end_turn"
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict) and parsed.get("type") == "tool_use":
                content_blocks.append(
                    _ToolUseBlock(
                        name=parsed["name"],
                        input_data=parsed.get("input", {}),
                        block_id=parsed.get("id", "tool_0"),
                    )
                )
                stop_reason = "tool_use"
            else:
                content_blocks.append(_TextBlock(raw_output))
        except (json.JSONDecodeError, KeyError):
            content_blocks.append(_TextBlock(raw_output))

        return _Message(content=content_blocks, stop_reason=stop_reason)


class CustomAnthropicClient:
    """Drop-in replacement for anthropic.Anthropic() routed through the claude CLI.

    Usage:
        client = CustomAnthropicClient()
        response = client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response.content[0].text)
    """

    def __init__(self) -> None:
        cli_path = os.environ.get("CLAUDE_CLI_PATH")
        model_name = os.environ.get("ANTHROPIC_MODEL_NAME")

        if not cli_path:
            raise EnvironmentError(
                "CLAUDE_CLI_PATH environment variable is not set. "
                "Set it to the path of your claude CLI binary."
            )
        if not model_name:
            raise EnvironmentError(
                "ANTHROPIC_MODEL_NAME environment variable is not set. "
                "Set it to the Claude model alias (e.g. haiku, sonnet, opus)."
            )

        self.messages = _MessagesResource(cli_path=cli_path, model_name=model_name)
