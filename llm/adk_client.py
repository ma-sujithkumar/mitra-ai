"""Shared ADK LLM factory for the MITRA pipeline.

All agents must use this module instead of constructing LiteLlm or
OpenAICompatibleLlm directly, so provider/key resolution stays in one place.

Usage::

    from llm.adk_client import build_llm_model, LlmSettings, LlmSettingsResolver

    resolver = LlmSettingsResolver(config_loader)
    settings = resolver.resolve(provider="openai", model="gpt-4o")
    llm_model = build_llm_model(settings)
    agent = LlmAgent(model=llm_model, ...)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm

# Re-export so callers only need to import from llm.adk_client
from backend.agents.metadata_gen_agent import LlmSettings, LlmSettingsResolver
from backend.agents.feature_engineering.openai_llm import OpenAICompatibleLlm

if TYPE_CHECKING:
    pass

# Providers that need the direct OpenAI SDK path (Harmony token stripping,
# no litellm in call path). All other providers go through LiteLlm.
_OPENAI_DIRECT_PROVIDERS = frozenset({"nvidia", "nim", "openai_direct"})


def build_llm_model(settings: LlmSettings) -> BaseLlm:
    """Return the appropriate ADK BaseLlm for the given LlmSettings.

    - NVIDIA NIM / openai_direct providers => OpenAICompatibleLlm (direct OpenAI
      SDK, Harmony token stripping).
    - All standard providers (openai, anthropic, gemini, ...) => LiteLlm
      (routes through litellm with api_key + api_base).
    """
    provider_lower = settings.provider.lower()

    if provider_lower in _OPENAI_DIRECT_PROVIDERS:
        return OpenAICompatibleLlm(
            model=settings.model,
            api_key=settings.api_key or "",
            base_url=settings.effective_gateway_url(),
        )

    # Standard path: LiteLlm handles openai, anthropic, gemini, azure, etc.
    lite_llm_kwargs: dict[str, str] = {}
    if settings.api_key:
        lite_llm_kwargs["api_key"] = settings.api_key
    effective_gateway = settings.effective_gateway_url()
    if effective_gateway:
        lite_llm_kwargs["api_base"] = effective_gateway

    return LiteLlm(model=settings.model, **lite_llm_kwargs)


__all__ = [
    "LlmSettings",
    "LlmSettingsResolver",
    "build_llm_model",
]
