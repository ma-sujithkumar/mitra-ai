from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import certifi
from dotenv import dotenv_values
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from backend.agents.tools import MetadataTools
from backend.config_loader import ConfigLoader


@dataclass(frozen=True)
class LlmSettings:
    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    gateway_url: str | None = None
    ca_bundle_path: Path | None = field(default=None, repr=False)
    source: str = "config"

    def public_dict(self) -> dict[str, str | None]:
        return {
            "provider": self.provider,
            "model": self.model,
            "gateway_url": self.gateway_url,
            "source": self.source,
        }


@dataclass(frozen=True)
class MetadataGenerationInput:
    session_id: str
    workspace_root: Path
    llm_settings: LlmSettings
    description: str | None = None
    target_col: str | None = None
    problem_type: str | None = None
    user_metadata_context: str | None = None


@dataclass(frozen=True)
class MetadataGenerationResult:
    metadata: dict[str, Any]
    metadata_path: Path


class MetadataGenerationError(RuntimeError):
    pass


class MetadataAgentToolAdapter:
    def __init__(self, metadata_tools: MetadataTools) -> None:
        self.metadata_tools = metadata_tools

    def read_mini_data(self, session_id: str) -> str:
        return self.metadata_tools.read_mini_data(session_id=session_id)

    def write_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, str]:
        result = self.metadata_tools.write_metadata(
            session_id=session_id,
            metadata=metadata,
        )
        return {
            "session_id": result.session_id,
            "metadata_path": str(result.metadata_path),
        }


def configure_default_ssl_certificates(
    ca_bundle_path: Path | str | None = None,
) -> Path:
    explicit_certificate_path = _first_non_blank_string(
        str(ca_bundle_path) if ca_bundle_path is not None else None,
        os.environ.get("LLM_CA_BUNDLE"),
    )
    if explicit_certificate_path is not None:
        certificate_path = _validated_certificate_path(explicit_certificate_path)
        certificate_path_string = str(certificate_path)
        os.environ["SSL_CERT_FILE"] = certificate_path_string
        os.environ["REQUESTS_CA_BUNDLE"] = certificate_path_string
        return certificate_path

    existing_certificate_path = _first_non_blank_string(
        os.environ.get("SSL_CERT_FILE"),
        os.environ.get("REQUESTS_CA_BUNDLE"),
    )
    if existing_certificate_path is not None:
        certificate_path = _validated_certificate_path(existing_certificate_path)
        certificate_path_string = str(certificate_path)
        os.environ.setdefault("SSL_CERT_FILE", certificate_path_string)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certificate_path_string)
        return certificate_path

    certificate_path = _validated_certificate_path(certifi.where())
    certificate_path_string = str(certificate_path)
    os.environ["SSL_CERT_FILE"] = certificate_path_string
    os.environ["REQUESTS_CA_BUNDLE"] = certificate_path_string
    return certificate_path


def _validated_certificate_path(raw_path: str) -> Path:
    certificate_path = Path(raw_path).expanduser()
    if not certificate_path.is_file():
        raise FileNotFoundError(f"CA bundle file not found: {certificate_path}")
    try:
        ca_certificates = ssl.create_default_context(
            cafile=str(certificate_path)
        ).get_ca_certs()
    except ssl.SSLError as exc:
        raise ValueError(
            "LLM_CA_BUNDLE must point to a valid PEM CA bundle."
        ) from exc

    if not ca_certificates:
        raise ValueError(
            "LLM_CA_BUNDLE must contain at least one CA certificate."
        )
    return certificate_path


def _first_non_blank_string(*values: str | None) -> str | None:
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return None


class LlmSettingsResolver:
    def __init__(
        self,
        config_loader: ConfigLoader,
        env_path: Path | None = None,
    ) -> None:
        self.config_loader = config_loader
        self.env_path = env_path or config_loader.repo_root / ".env"

    def resolve(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        gateway_url: str | None = None,
    ) -> LlmSettings:
        env_settings = dotenv_values(self.env_path)
        resolved_provider = self._first_non_blank(
            provider,
            env_settings.get("LLM_TYPE"),
        )
        if resolved_provider is None:
            raise ValueError("LLM_TYPE is required")

        normalized_provider = resolved_provider.lower()
        resolved_model = self._first_non_blank(
            model,
            env_settings.get("LLM_MODEL"),
        )
        if resolved_model is None:
            resolved_model = self.config_loader.base_model_for_provider(
                provider=normalized_provider
            )

        resolved_api_key = self._first_non_blank(
            api_key,
            env_settings.get("LLM_API_KEY"),
        )
        resolved_gateway_url = self._first_non_blank(
            gateway_url,
            env_settings.get("LLM_GATEWAY_URL"),
        )
        resolved_ca_bundle_path = self._resolve_ca_bundle_path(
            self._first_non_blank(
                env_settings.get("LLM_CA_BUNDLE"),
                os.environ.get("LLM_CA_BUNDLE"),
            )
        )
        source = self._resolve_source(
            provider=provider,
            model=model,
            api_key=api_key,
            gateway_url=gateway_url,
            env_settings=env_settings,
        )
        return LlmSettings(
            provider=normalized_provider,
            model=resolved_model,
            api_key=resolved_api_key,
            gateway_url=resolved_gateway_url,
            ca_bundle_path=resolved_ca_bundle_path,
            source=source,
        )

    @staticmethod
    def _first_non_blank(*values: str | None) -> str | None:
        for value in values:
            if value is not None and value.strip():
                return value.strip()
        return None

    def _resolve_ca_bundle_path(self, raw_path: str | None) -> Path | None:
        ca_bundle_path_string = self._first_non_blank(raw_path)
        if ca_bundle_path_string is None:
            return None

        ca_bundle_path = Path(ca_bundle_path_string).expanduser()
        if not ca_bundle_path.is_absolute():
            ca_bundle_path = self.config_loader.repo_root / ca_bundle_path

        try:
            return _validated_certificate_path(str(ca_bundle_path))
        except FileNotFoundError as exc:
            raise ValueError(f"LLM_CA_BUNDLE file not found: {ca_bundle_path}") from exc

    @classmethod
    def _resolve_source(
        cls,
        provider: str | None,
        model: str | None,
        api_key: str | None,
        gateway_url: str | None,
        env_settings: dict[str, str | None],
    ) -> str:
        per_run_values = [provider, model, api_key, gateway_url]
        env_values = [
            env_settings.get("LLM_TYPE"),
            env_settings.get("LLM_MODEL"),
            env_settings.get("LLM_API_KEY"),
            env_settings.get("LLM_GATEWAY_URL"),
        ]
        if any(cls._first_non_blank(value) is not None for value in per_run_values):
            return "per_run"
        if any(cls._first_non_blank(value) is not None for value in env_values):
            return "env"
        return "config"


class MetadataGenAgent:
    def __init__(
        self,
        llm_settings: LlmSettings,
        metadata_tools: MetadataTools,
        prompt_path: Path | None = None,
    ) -> None:
        self.llm_settings = llm_settings
        self.metadata_tools = metadata_tools
        self.metadata_agent_tools = MetadataAgentToolAdapter(
            metadata_tools=metadata_tools
        )
        self.prompt_path = (
            prompt_path
            or Path(__file__).resolve().parent
            / "prompts"
            / "metadata_gen.md"
        )
        self.instruction = self.prompt_path.read_text(encoding="utf-8")
        self.agent = self._build_agent()

    def _build_agent(self) -> LlmAgent:
        lite_llm_kwargs: dict[str, str] = {}
        if self.llm_settings.api_key:
            lite_llm_kwargs["api_key"] = self.llm_settings.api_key
        if self.llm_settings.gateway_url:
            lite_llm_kwargs["api_base"] = self.llm_settings.gateway_url

        model = LiteLlm(
            model=self.llm_settings.model,
            **lite_llm_kwargs,
        )
        return LlmAgent(
            name="metadata_gen_agent",
            description="Generates MITRA metadata.json from mini_data.csv only.",
            model=model,
            instruction=self.instruction,
            tools=[
                self.metadata_agent_tools.read_mini_data,
                self.metadata_agent_tools.write_metadata,
            ],
        )


class MetadataAgentRunner:
    app_name = "mitra_metadata"
    user_id = "mitra_epic1"

    def generate_metadata(
        self,
        generation_input: MetadataGenerationInput,
    ) -> MetadataGenerationResult:
        configure_default_ssl_certificates(
            ca_bundle_path=generation_input.llm_settings.ca_bundle_path
        )
        metadata_tools = MetadataTools(workspace_root=generation_input.workspace_root)
        metadata_agent = MetadataGenAgent(
            llm_settings=generation_input.llm_settings,
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
            agent=metadata_agent.agent,
            session_service=session_service,
        )
        message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=self._build_user_message(
                        generation_input=generation_input
                    )
                )
            ],
        )

        for _event in runner.run(
            user_id=self.user_id,
            session_id=generation_input.session_id,
            new_message=message,
        ):
            pass

        metadata_path = (
            generation_input.workspace_root
            / generation_input.session_id
            / "reports"
            / "metadata.json"
        )
        if not metadata_path.is_file():
            raise MetadataGenerationError("Metadata agent did not write metadata.json")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return MetadataGenerationResult(
            metadata=metadata,
            metadata_path=metadata_path,
        )

    @staticmethod
    def _build_user_message(generation_input: MetadataGenerationInput) -> str:
        return "\n".join(
            [
                f"session_id: {generation_input.session_id}",
                f"description: {generation_input.description or ''}",
                f"target_col: {generation_input.target_col or ''}",
                f"problem_type: {generation_input.problem_type or ''}",
                "user_metadata_context:",
                generation_input.user_metadata_context or "",
                "",
                "Read mini_data for this session, infer metadata, and call "
                "write_metadata with the final JSON object.",
            ]
        )
