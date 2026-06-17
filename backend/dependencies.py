from __future__ import annotations

from fastapi import Request

from backend.agents.llm_smoke_test import LlmSmokeTester
from backend.agents.metadata_gen_agent import MetadataAgentRunner
from backend.config_loader import ConfigLoader
from backend.jobs import JobRegistry
from backend.session import SessionManager


def get_config_loader(request: Request) -> ConfigLoader:
    return request.app.state.config_loader


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_job_registry(request: Request) -> JobRegistry:
    return request.app.state.job_registry


def get_metadata_agent_runner(request: Request) -> MetadataAgentRunner:
    return request.app.state.metadata_agent_runner


def get_llm_smoke_tester(request: Request) -> LlmSmokeTester:
    return request.app.state.llm_smoke_tester
