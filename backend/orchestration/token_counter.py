"""Token usage accumulator for the MITRA pipeline.

Wraps the ADK LiteLlm call path to accumulate per-agent token counts into
<session_dir>/token_usage.json after each invocation.

Usage (in any agent that calls an LLM):
    from backend.orchestration.token_counter import TokenCounter
    counter = TokenCounter(session_dir)
    response = counter.record(agent_name="judge", input_tokens=..., output_tokens=...)

The JSON format is:
    {
      "total_input_tokens": <int>,
      "total_output_tokens": <int>,
      "total_tokens": <int>,
      "agents": {
        "<agent_name>": {
          "calls": <int>,
          "input_tokens": <int>,
          "output_tokens": <int>
        }, ...
      }
    }
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# File-level write lock so concurrent threads (SHAP / overfitting / HPT parallel
# workers) don't corrupt the JSON.
_WRITE_LOCK = threading.Lock()

TOKEN_USAGE_FILENAME = "token_usage.json"


def _empty_usage() -> dict:
    return {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "agents": {},
    }


class TokenCounter:
    """Accumulates token usage for one pipeline session.

    Thread-safe for concurrent agent calls within the same process.
    """

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.token_usage_path = self.session_dir / TOKEN_USAGE_FILENAME
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        if self.token_usage_path.exists():
            with self.token_usage_path.open(encoding="utf-8") as usage_file:
                return json.load(usage_file)
        return _empty_usage()

    def _write(self, usage_data: dict) -> None:
        self.token_usage_path.write_text(json.dumps(usage_data, indent=2), encoding="utf-8")

    def record(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Add one call's token counts to the accumulator and persist immediately."""
        with _WRITE_LOCK:
            usage_data = self._read()
            agent_entry = usage_data["agents"].setdefault(
                agent_name, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            agent_entry["calls"] += 1
            agent_entry["input_tokens"] += input_tokens
            agent_entry["output_tokens"] += output_tokens

            usage_data["total_input_tokens"] += input_tokens
            usage_data["total_output_tokens"] += output_tokens
            usage_data["total_tokens"] = (
                usage_data["total_input_tokens"] + usage_data["total_output_tokens"]
            )
            self._write(usage_data)
            logger.debug(
                "=> token usage: agent=%s +in=%d +out=%d total=%d",
                agent_name,
                input_tokens,
                output_tokens,
                usage_data["total_tokens"],
            )

    def record_from_adk_response(
        self,
        agent_name: str,
        adk_event: Any,
    ) -> None:
        """Extract token counts from an ADK event and record them.

        ADK InMemoryRunner events may carry usage metadata in event.usage_metadata.
        Gracefully does nothing if the attribute is absent (not all events carry it).
        """
        usage_metadata = getattr(adk_event, "usage_metadata", None)
        if usage_metadata is None:
            return
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        if input_tokens or output_tokens:
            self.record(agent_name=agent_name, input_tokens=input_tokens, output_tokens=output_tokens)

    def summary(self) -> dict:
        """Return current usage totals without modifying anything."""
        with _WRITE_LOCK:
            return self._read()

    def reset(self) -> None:
        """Reset all counters (useful between test runs)."""
        with _WRITE_LOCK:
            self._write(_empty_usage())
            logger.debug("=> token_counter: reset for session %s", self.session_dir)
