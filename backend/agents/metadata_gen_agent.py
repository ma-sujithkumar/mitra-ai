from __future__ import annotations

import asyncio
import json
import logging
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
from backend.pii import match_pii_columns


logger = logging.getLogger("mitra.metadata_agent")


# Maps the dtype-style values models commonly emit onto the column-type enum the
# metadata schema accepts ("categorical" / "numeric").
COLUMN_TYPE_SYNONYMS = {
    "numeric": "numeric",
    "number": "numeric",
    "numerical": "numeric",
    "continuous": "numeric",
    "float": "numeric",
    "float32": "numeric",
    "float64": "numeric",
    "double": "numeric",
    "decimal": "numeric",
    "real": "numeric",
    "int": "numeric",
    "int32": "numeric",
    "int64": "numeric",
    "integer": "numeric",
    "long": "numeric",
    "categorical": "categorical",
    "category": "categorical",
    "categoric": "categorical",
    "nominal": "categorical",
    "ordinal": "categorical",
    "factor": "categorical",
    "string": "categorical",
    "str": "categorical",
    "object": "categorical",
    "text": "categorical",
    "char": "categorical",
    "bool": "categorical",
    "boolean": "categorical",
}

# Maps problem-type phrasings onto the top-level schema enum (supervised /
# unsupervised). Legacy classification/regression values are mapped to supervised
# and recovered as a problem_subtype by _normalize_metadata_enums.
PROBLEM_TYPE_SYNONYMS = {
    "supervised": "supervised",
    "labelled": "supervised",
    "labeled": "supervised",
    "classification": "supervised",
    "binary": "supervised",
    "binary classification": "supervised",
    "multiclass": "supervised",
    "multi-class": "supervised",
    "multiclass classification": "supervised",
    "regression": "supervised",
    "regressor": "supervised",
    "unsupervised": "unsupervised",
    "clustering": "unsupervised",
    "cluster": "unsupervised",
}

# Maps problem-subtype phrasings onto the schema enum (classification /
# regression).
PROBLEM_SUBTYPE_SYNONYMS = {
    "classification": "classification",
    "classifier": "classification",
    "binary": "classification",
    "binary classification": "classification",
    "multiclass": "classification",
    "multi-class": "classification",
    "multiclass classification": "classification",
    "regression": "regression",
    "regressor": "regression",
    "continuous": "regression",
}


@dataclass(frozen=True)
class LlmSettings:
    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    gateway_url: str | None = None
    default_base_url: str | None = None
    ca_bundle_path: Path | None = field(default=None, repr=False)
    source: str = "config"

    def effective_gateway_url(self) -> str | None:
        # Explicit per-run/env gateway wins; otherwise fall back to the
        # provider's configured default base URL as the call endpoint.
        return self.gateway_url or self.default_base_url

    def public_dict(self) -> dict[str, str | None]:
        return {
            "provider": self.provider,
            "model": self.model,
            "gateway_url": self.effective_gateway_url(),
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
    pii_patterns: list[str] = field(default_factory=list)
    user_metadata_descriptions: dict[str, str] = field(default_factory=dict)
    user_metadata_important_cols: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetadataGenerationResult:
    metadata: dict[str, Any]
    metadata_path: Path


class MetadataGenerationError(RuntimeError):
    pass


class MetadataAgentToolAdapter:
    def __init__(
        self,
        metadata_tools: MetadataTools,
        user_metadata_descriptions: dict[str, str] | None = None,
        user_metadata_important_cols: list[str] | None = None,
    ) -> None:
        self.metadata_tools = metadata_tools
        self.user_metadata_descriptions = user_metadata_descriptions or {}
        self.user_metadata_important_cols = user_metadata_important_cols or []

    def read_mini_data(self, session_id: str) -> str:
        logger.info("tool read_mini_data called: session_id=%s", session_id)
        mini_data = self.metadata_tools.read_mini_data(session_id=session_id)
        logger.info(
            "tool read_mini_data returned: session_id=%s chars=%d",
            session_id,
            len(mini_data),
        )
        return mini_data

    def write_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, str]:
        # Some models (e.g. llama-3.3 via NVIDIA) return the metadata argument
        # as a JSON-encoded string instead of a structured object, so coerce it
        # back to a dict before writing.
        normalized_metadata = self._coerce_metadata_dict(metadata=metadata)
        # Map dtype-style enum values (e.g. "float") onto the schema vocabulary.
        normalized_metadata = self._normalize_metadata_enums(
            metadata=normalized_metadata
        )
        # Deterministically resolve which columns to drop (user-excluded plus
        # name-based PII), then enforce that decision across cols_to_drop,
        # input_cols, statistics, and the persisted mini_data.csv.
        drop_columns = self._resolve_drop_columns(
            session_id=session_id,
            normalized_metadata=normalized_metadata,
        )
        normalized_metadata["cols_to_drop"] = sorted(drop_columns)
        normalized_metadata["input_cols"] = self._filter_input_cols(
            input_cols=normalized_metadata.get("input_cols"),
            drop_columns=drop_columns,
        )
        normalized_metadata["important_cols"] = self._resolve_important_cols(
            session_id=session_id,
            normalized_metadata=normalized_metadata,
            drop_columns=drop_columns,
        )
        # Statistics are objective facts from mini_data.csv, so compute them
        # deterministically instead of trusting the model's transcription. Dropped
        # columns are excluded and descriptions are injected only from the uploaded
        # metadata file.
        normalized_metadata["statistics"] = self.metadata_tools.build_statistics(
            session_id=session_id,
            exclude_columns=drop_columns,
            descriptions=self.user_metadata_descriptions,
        )
        logger.info(
            "tool write_metadata called: session_id=%s keys=%s dropped=%s",
            session_id,
            sorted(normalized_metadata.keys()),
            sorted(drop_columns),
        )
        result = self.metadata_tools.write_metadata(
            session_id=session_id,
            metadata=normalized_metadata,
        )
        # Remove the dropped columns from the persisted mini_data.csv so the saved
        # artifact no longer exposes PII / excluded columns.
        self.metadata_tools.prune_mini_data(
            session_id=session_id,
            drop_columns=drop_columns,
        )
        logger.info(
            "tool write_metadata wrote: session_id=%s path=%s",
            session_id,
            result.metadata_path,
        )
        return {
            "session_id": result.session_id,
            "metadata_path": str(result.metadata_path),
        }

    def _resolve_drop_columns(
        self,
        session_id: str,
        normalized_metadata: dict[str, Any],
    ) -> set[str]:
        # Union of the LLM/description excludes and deterministic name-based PII,
        # restricted to real dataset columns, and never dropping the target column.
        dataset_columns = self.metadata_tools.mini_data_columns(session_id=session_id)
        dataset_column_set = set(dataset_columns)
        llm_drop = {
            str(column_name)
            for column_name in normalized_metadata.get("cols_to_drop", []) or []
        }
        pii_drop = set(
            match_pii_columns(
                column_names=dataset_columns,
                pii_patterns=self.metadata_tools.pii_patterns,
            )
        )
        drop_columns = (llm_drop | pii_drop) & dataset_column_set
        target_col = normalized_metadata.get("target_col")
        if isinstance(target_col, str):
            drop_columns.discard(target_col)
        return drop_columns

    @staticmethod
    def _filter_input_cols(
        input_cols: Any,
        drop_columns: set[str],
    ) -> list[Any]:
        if not isinstance(input_cols, list):
            return []
        return [
            input_col
            for input_col in input_cols
            if not (
                isinstance(input_col, dict)
                and str(input_col.get("name")) in drop_columns
            )
        ]

    def _resolve_important_cols(
        self,
        session_id: str,
        normalized_metadata: dict[str, Any],
        drop_columns: set[str],
    ) -> list[str]:
        # Merge LLM-proposed important columns with those the user flagged in the
        # uploaded metadata file, keeping only real, non-dropped columns.
        dataset_column_set = set(
            self.metadata_tools.mini_data_columns(session_id=session_id)
        )
        candidate_important = [
            str(column_name)
            for column_name in (normalized_metadata.get("important_cols") or [])
        ]
        candidate_important.extend(self.user_metadata_important_cols)
        resolved: list[str] = []
        for column_name in candidate_important:
            if (
                column_name in dataset_column_set
                and column_name not in drop_columns
                and column_name not in resolved
            ):
                resolved.append(column_name)
        return resolved

    @staticmethod
    def _coerce_metadata_dict(metadata: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            parsed_metadata = json.loads(metadata)
            if not isinstance(parsed_metadata, dict):
                raise ValueError(
                    "metadata tool argument must decode to a JSON object, got "
                    f"{type(parsed_metadata).__name__}"
                )
            return parsed_metadata
        raise ValueError(
            f"metadata tool argument must be an object, got {type(metadata).__name__}"
        )

    @classmethod
    def _normalize_metadata_enums(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        normalized_metadata = dict(metadata)

        # A legacy problem_type of classification/regression is recovered as the
        # subtype before the top-level value is mapped to supervised.
        raw_problem_type = normalized_metadata.get("problem_type")
        recovered_subtype = cls._map_enum_value(
            value=raw_problem_type,
            synonyms=PROBLEM_SUBTYPE_SYNONYMS,
        )
        normalized_problem_type = cls._map_enum_value(
            value=raw_problem_type,
            synonyms=PROBLEM_TYPE_SYNONYMS,
        )
        if normalized_problem_type is not None:
            normalized_metadata["problem_type"] = normalized_problem_type

        normalized_subtype = cls._map_enum_value(
            value=normalized_metadata.get("problem_subtype"),
            synonyms=PROBLEM_SUBTYPE_SYNONYMS,
        )
        if normalized_subtype is None:
            normalized_subtype = recovered_subtype
        if normalized_metadata.get("problem_type") == "supervised":
            normalized_metadata["problem_subtype"] = normalized_subtype
        elif normalized_metadata.get("problem_type") == "unsupervised":
            # Unsupervised runs have no subtype.
            normalized_metadata["problem_subtype"] = None

        target_col_type = normalized_metadata.get("target_col_type")
        normalized_target_col_type = cls._map_enum_value(
            value=target_col_type,
            synonyms=COLUMN_TYPE_SYNONYMS,
        )
        if normalized_target_col_type is not None:
            normalized_metadata["target_col_type"] = normalized_target_col_type

        input_cols = normalized_metadata.get("input_cols")
        if isinstance(input_cols, list):
            normalized_metadata["input_cols"] = [
                cls._normalize_input_col(input_col=input_col)
                for input_col in input_cols
            ]
        return normalized_metadata

    @classmethod
    def _normalize_input_col(cls, input_col: Any) -> Any:
        if not isinstance(input_col, dict):
            return input_col
        normalized_input_col = dict(input_col)
        normalized_col_type = cls._map_enum_value(
            value=normalized_input_col.get("col_type"),
            synonyms=COLUMN_TYPE_SYNONYMS,
        )
        if normalized_col_type is not None:
            normalized_input_col["col_type"] = normalized_col_type
        return normalized_input_col

    @staticmethod
    def _map_enum_value(value: Any, synonyms: dict[str, str]) -> str | None:
        # Returns the mapped enum value, or None when there is nothing to map
        # (non-string, or an unknown value left untouched for the validator).
        if not isinstance(value, str):
            return None
        return synonyms.get(value.strip().lower())


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
        # The explicit gateway (per-run or env) decides credential presence, so
        # it must stay independent of the provider's default base URL fallback.
        resolved_gateway_url = self._first_non_blank(
            gateway_url,
            env_settings.get("LLM_GATEWAY_URL"),
        )
        # Default base URL paired with the provider's default model; applied as
        # the call endpoint only when no explicit gateway is supplied.
        resolved_default_base_url = self._default_base_url_for_provider(
            provider=normalized_provider
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
            default_base_url=resolved_default_base_url,
            ca_bundle_path=resolved_ca_bundle_path,
            source=source,
        )

    @staticmethod
    def _first_non_blank(*values: str | None) -> str | None:
        for value in values:
            if value is not None and value.strip():
                return value.strip()
        return None

    def _default_base_url_for_provider(self, provider: str) -> str | None:
        # Unknown providers (e.g. an OpenAI-compatible gateway named via
        # LLM_TYPE) have no configured default, so return None instead of
        # raising and let the caller/litellm decide.
        try:
            return self.config_loader.base_url_for_provider(provider=provider)
        except ValueError:
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
        user_metadata_descriptions: dict[str, str] | None = None,
        user_metadata_important_cols: list[str] | None = None,
    ) -> None:
        self.llm_settings = llm_settings
        self.metadata_tools = metadata_tools
        self.metadata_agent_tools = MetadataAgentToolAdapter(
            metadata_tools=metadata_tools,
            user_metadata_descriptions=user_metadata_descriptions,
            user_metadata_important_cols=user_metadata_important_cols,
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
        effective_gateway_url = self.llm_settings.effective_gateway_url()
        if effective_gateway_url:
            lite_llm_kwargs["api_base"] = effective_gateway_url

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
        logger.info(
            "generate_metadata start: session_id=%s provider=%s model=%s gateway=%s",
            generation_input.session_id,
            generation_input.llm_settings.provider,
            generation_input.llm_settings.model,
            generation_input.llm_settings.gateway_url or "(none)",
        )
        configure_default_ssl_certificates(
            ca_bundle_path=generation_input.llm_settings.ca_bundle_path
        )
        metadata_tools = MetadataTools(
            workspace_root=generation_input.workspace_root,
            pii_patterns=generation_input.pii_patterns,
        )
        metadata_agent = MetadataGenAgent(
            llm_settings=generation_input.llm_settings,
            metadata_tools=metadata_tools,
            user_metadata_descriptions=generation_input.user_metadata_descriptions,
            user_metadata_important_cols=generation_input.user_metadata_important_cols,
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

        # Drive the agent through the supported async runner. The synchronous
        # Runner.run is deprecated and stalls with async LiteLLM clients.
        logger.info(
            "entering agent run loop: session_id=%s", generation_input.session_id
        )
        asyncio.run(
            self._drain_agent_events(
                runner=runner,
                session_id=generation_input.session_id,
                message=message,
            )
        )
        logger.info(
            "agent run loop finished: session_id=%s", generation_input.session_id
        )

        metadata_path = (
            generation_input.workspace_root
            / generation_input.session_id
            / "reports"
            / "metadata.json"
        )
        if not metadata_path.is_file():
            logger.error(
                "metadata.json missing after run: session_id=%s expected_path=%s",
                generation_input.session_id,
                metadata_path,
            )
            raise MetadataGenerationError("Metadata agent did not write metadata.json")

        logger.info(
            "metadata.json found: session_id=%s path=%s",
            generation_input.session_id,
            metadata_path,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return MetadataGenerationResult(
            metadata=metadata,
            metadata_path=metadata_path,
        )

    async def _drain_agent_events(
        self,
        runner: Runner,
        session_id: str,
        message: types.Content,
    ) -> None:
        event_count = 0
        async for event in runner.run_async(
            user_id=self.user_id,
            session_id=session_id,
            new_message=message,
        ):
            event_count += 1
            self._log_agent_event(
                session_id=session_id,
                event_index=event_count,
                event=event,
            )
        logger.info(
            "agent emitted %d events: session_id=%s", event_count, session_id
        )

    @staticmethod
    def _log_agent_event(session_id: str, event_index: int, event: Any) -> None:
        author = getattr(event, "author", "unknown")
        function_calls = event.get_function_calls() if hasattr(event, "get_function_calls") else []
        function_responses = (
            event.get_function_responses() if hasattr(event, "get_function_responses") else []
        )
        text_parts = []
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)
        logger.info(
            "agent event %d: session_id=%s author=%s tool_calls=%s tool_responses=%s "
            "is_final=%s text=%r",
            event_index,
            session_id,
            author,
            [call.name for call in function_calls],
            [response.name for response in function_responses],
            event.is_final_response() if hasattr(event, "is_final_response") else "n/a",
            (" ".join(text_parts))[:300],
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

 