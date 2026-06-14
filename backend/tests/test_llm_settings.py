import json
import os
import ssl
from pathlib import Path

import certifi
import pytest

from backend.agents.metadata_gen_agent import LlmSettings
from backend.agents.metadata_gen_agent import LlmSettingsResolver
from backend.agents.metadata_gen_agent import MetadataAgentToolAdapter
from backend.agents.metadata_gen_agent import MetadataGenAgent
from backend.agents.metadata_gen_agent import configure_default_ssl_certificates
from backend.agents.tools import MetadataTools
from backend.config_loader import ConfigLoader
from backend.session import SessionManager


def write_test_ca_bundle(ca_bundle_path: Path) -> None:
    ca_bundle_path.write_text(
        Path(certifi.where()).read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def test_per_run_llm_settings_override_env(
    tmp_path: Path,
    test_config_loader: ConfigLoader,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_TYPE=openai\n"
        "LLM_API_KEY=env-key\n"
        "LLM_MODEL=openai/env-model\n"
        "LLM_GATEWAY_URL=https://env.example.test\n",
        encoding="utf-8",
    )
    resolver = LlmSettingsResolver(
        config_loader=test_config_loader,
        env_path=env_path,
    )

    settings = resolver.resolve(
        provider="anthropic",
        model="anthropic/run-model",
        api_key="run-key",
        gateway_url="https://run.example.test",
    )

    assert settings.provider == "anthropic"
    assert settings.model == "anthropic/run-model"
    assert settings.api_key == "run-key"
    assert settings.gateway_url == "https://run.example.test"
    assert settings.source == "per_run"


def test_blank_model_resolves_to_provider_base_model(
    tmp_path: Path,
    test_config_loader: ConfigLoader,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_TYPE=gemini\n"
        "LLM_API_KEY=env-key\n"
        "LLM_MODEL=\n",
        encoding="utf-8",
    )
    resolver = LlmSettingsResolver(
        config_loader=test_config_loader,
        env_path=env_path,
    )

    settings = resolver.resolve(model=" ")

    assert settings.provider == "gemini"
    assert settings.model == "gemini/gemini-3-pro"
    assert settings.source == "env"


def test_llm_settings_resolves_ca_bundle_from_env_file(
    tmp_path: Path,
    test_config_loader: ConfigLoader,
) -> None:
    ca_bundle_path = tmp_path / "ca-bundle.pem"
    write_test_ca_bundle(ca_bundle_path=ca_bundle_path)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_TYPE=openai\n"
        "LLM_API_KEY=env-key\n"
        "LLM_CA_BUNDLE=ca-bundle.pem\n",
        encoding="utf-8",
    )
    resolver = LlmSettingsResolver(
        config_loader=test_config_loader,
        env_path=env_path,
    )

    settings = resolver.resolve()

    assert settings.ca_bundle_path == ca_bundle_path
    assert "ca_bundle" not in settings.public_dict()


def test_llm_settings_rejects_missing_ca_bundle(
    tmp_path: Path,
    test_config_loader: ConfigLoader,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_TYPE=openai\n"
        "LLM_API_KEY=env-key\n"
        "LLM_CA_BUNDLE=missing.pem\n",
        encoding="utf-8",
    )
    resolver = LlmSettingsResolver(
        config_loader=test_config_loader,
        env_path=env_path,
    )

    with pytest.raises(ValueError, match="LLM_CA_BUNDLE"):
        resolver.resolve()


def test_llm_settings_rejects_invalid_ca_bundle(
    tmp_path: Path,
    test_config_loader: ConfigLoader,
) -> None:
    ca_bundle_path = tmp_path / "ca-bundle.pem"
    ca_bundle_path.write_text("not a certificate", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_TYPE=openai\n"
        "LLM_API_KEY=env-key\n"
        "LLM_CA_BUNDLE=ca-bundle.pem\n",
        encoding="utf-8",
    )
    resolver = LlmSettingsResolver(
        config_loader=test_config_loader,
        env_path=env_path,
    )

    with pytest.raises(ValueError, match="valid PEM CA bundle"):
        resolver.resolve()


def test_api_key_is_not_serialized_to_public_settings(
    tmp_path: Path,
    test_config_loader: ConfigLoader,
) -> None:
    resolver = LlmSettingsResolver(
        config_loader=test_config_loader,
        env_path=tmp_path / ".env",
    )

    settings = resolver.resolve(
        provider="openai",
        api_key="secret-key",
    )

    assert "api_key" not in settings.public_dict()
    assert "secret-key" not in repr(settings)


def test_api_key_is_not_written_to_session_files(
    tmp_path: Path,
    test_config_loader: ConfigLoader,
) -> None:
    session_manager = SessionManager(workspace_root=tmp_path / ".mitra")
    session_info = session_manager.create_session(original_filename="dataset.csv")
    resolver = LlmSettingsResolver(
        config_loader=test_config_loader,
        env_path=tmp_path / ".env",
    )

    resolver.resolve(
        provider="openai",
        api_key="secret-key",
    )

    session_payload = json.loads(
        (session_info.session_path / "session.json").read_text(encoding="utf-8")
    )
    assert "secret-key" not in json.dumps(session_payload)


def test_metadata_agent_builds_adk_litellm_wrapper(tmp_path: Path) -> None:
    metadata_tools = MetadataTools(workspace_root=tmp_path)
    settings = LlmSettings(
        provider="openai",
        model="openai/test-model",
        api_key="secret-key",
    )

    metadata_agent = MetadataGenAgent(
        llm_settings=settings,
        metadata_tools=metadata_tools,
    )

    assert metadata_agent.agent.name == "metadata_gen_agent"


def test_metadata_agent_tool_adapter_returns_json_schema_friendly_dict(
    tmp_path: Path,
) -> None:
    session_manager = SessionManager(workspace_root=tmp_path)
    session_info = session_manager.create_session(original_filename="dataset.csv")
    adapter = MetadataAgentToolAdapter(
        metadata_tools=MetadataTools(workspace_root=tmp_path)
    )
    metadata = {
        "session_id": session_info.session_id,
        "problem_type": "unsupervised",
        "target_col": None,
        "target_col_type": None,
        "input_cols": [],
        "cols_to_drop": [],
        "statistics": {},
    }

    result = adapter.write_metadata(
        session_id=session_info.session_id,
        metadata=metadata,
    )

    assert result == {
        "session_id": session_info.session_id,
        "metadata_path": str(session_info.reports_dir / "metadata.json"),
    }


def test_configure_default_ssl_certificates_sets_certifi_bundle(monkeypatch) -> None:
    monkeypatch.delenv("LLM_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    certificate_path = configure_default_ssl_certificates()

    assert certificate_path == Path(certifi.where())
    assert certificate_path.is_file()
    assert certificate_path == Path(ssl.get_default_verify_paths().cafile)


def test_configure_default_ssl_certificates_preserves_user_cert_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    custom_certificate_path = tmp_path / "custom.pem"
    write_test_ca_bundle(ca_bundle_path=custom_certificate_path)
    monkeypatch.delenv("LLM_CA_BUNDLE", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(custom_certificate_path))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    configure_default_ssl_certificates()

    assert Path(os.environ["SSL_CERT_FILE"]) == custom_certificate_path
    assert Path(os.environ["REQUESTS_CA_BUNDLE"]) == custom_certificate_path


def test_configure_default_ssl_certificates_uses_explicit_ca_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stale_certificate_path = tmp_path / "stale.pem"
    custom_certificate_path = tmp_path / "custom.pem"
    stale_certificate_path.write_text("", encoding="utf-8")
    write_test_ca_bundle(ca_bundle_path=custom_certificate_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(stale_certificate_path))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(stale_certificate_path))

    certificate_path = configure_default_ssl_certificates(
        ca_bundle_path=custom_certificate_path
    )

    assert certificate_path == custom_certificate_path
    assert Path(os.environ["SSL_CERT_FILE"]) == custom_certificate_path
    assert Path(os.environ["REQUESTS_CA_BUNDLE"]) == custom_certificate_path
