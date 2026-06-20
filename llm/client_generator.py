"""Session-workspace LLM client generator.

During the Epic-1 setup step the user chooses a provider and model.
This module renders a concrete client.py into the session workspace at
  .mitra/<user_id>/<session_id>/llm/client.py
and optionally smoke-tests it before returning.

The generated client.py is the single LLM entry-point for that run.
The codebase itself stays read-only at runtime; all mutable artifacts
live under the session dir.
"""
from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from typing import Optional

from backend.agents.llm_smoke_test import LlmSmokeTester, LlmSmokeTestResult
from backend.agents.metadata_gen_agent import LlmSettings, LlmSettingsResolver
from backend.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Template for the generated session-scoped client.py.
# Uses only the standard library + the shared adk_client so no extra deps.
_CLIENT_TEMPLATE = textwrap.dedent("""\
    # Auto-generated LLM client for this session — do not edit manually.
    # Provider: {provider}  Model: {model}
    from llm.adk_client import build_llm_model, LlmSettings

    def get_session_llm():
        settings = LlmSettings(
            provider={provider!r},
            model={model!r},
            api_key={api_key!r},
            gateway_url={gateway_url!r},
        )
        return build_llm_model(settings)
""")


class ClientGenerator:
    """Generates and optionally smoke-tests a session-scoped LLM client file."""

    def __init__(
        self,
        config_loader: ConfigLoader,
        smoke_tester: Optional[LlmSmokeTester] = None,
    ) -> None:
        self.config_loader = config_loader
        self.resolver = LlmSettingsResolver(config_loader)
        self.smoke_tester = smoke_tester or LlmSmokeTester()

    def generate(
        self,
        session_dir: Path,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        gateway_url: Optional[str] = None,
        run_smoke_test: bool = True,
    ) -> tuple[Path, Optional[LlmSmokeTestResult]]:
        """Write client.py into session_dir/llm/ and optionally smoke-test it.

        Returns:
            (client_path, smoke_result) — smoke_result is None when
            run_smoke_test=False.
        """
        settings = self.resolver.resolve(
            provider=provider,
            model=model,
            api_key=api_key,
            gateway_url=gateway_url,
        )

        llm_dir = session_dir / "llm"
        llm_dir.mkdir(parents=True, exist_ok=True)
        client_path = llm_dir / "client.py"

        client_source = _CLIENT_TEMPLATE.format(
            provider=settings.provider,
            model=settings.model,
            # Never write key/url into tracked source; only session workspace.
            api_key=settings.api_key,
            gateway_url=settings.effective_gateway_url(),
        )
        client_path.write_text(client_source, encoding="utf-8")
        logger.info(
            "=> client.py written: provider=%s model=%s path=%s",
            settings.provider,
            settings.model,
            client_path,
        )

        smoke_result: Optional[LlmSmokeTestResult] = None
        if run_smoke_test:
            logger.info("=> running LLM smoke test ...")
            smoke_result = self.smoke_tester.run(settings)
            logger.info(
                "=> smoke test passed: latency=%dms", smoke_result.latency_ms
            )

        return client_path, smoke_result
