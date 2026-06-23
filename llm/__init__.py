"""Shared LLM client package.

Exposes build_llm_model() factory and re-exports LlmSettings / LlmSettingsResolver
so every agent can import from one place instead of reaching into metadata_gen_agent.
"""
from .adk_client import build_llm_model

__all__ = ["build_llm_model"]
