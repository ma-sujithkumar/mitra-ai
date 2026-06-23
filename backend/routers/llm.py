from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel

from backend.agents.llm_smoke_test import LlmSmokeTester
from backend.agents.llm_smoke_test import LlmSmokeTestError
from backend.agents.metadata_gen_agent import LlmSettings
from backend.agents.metadata_gen_agent import LlmSettingsResolver
from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader
from backend.dependencies import get_llm_smoke_tester


router = APIRouter(prefix="/api", tags=["llm"])
LLM_ENV_KEYS = ("LLM_TYPE", "LLM_MODEL", "LLM_API_KEY", "LLM_GATEWAY_URL")


class LlmSmokeTestRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    gateway_url: str | None = None


@router.post("/llm/smoke-test")
def run_llm_smoke_test(
    smoke_test_request: LlmSmokeTestRequest,
    config_loader: ConfigLoader = Depends(get_config_loader),
    llm_smoke_tester: LlmSmokeTester = Depends(get_llm_smoke_tester),
) -> dict[str, object]:
    llm_settings = _resolve_llm_settings(
        smoke_test_request=smoke_test_request,
        config_loader=config_loader,
    )
    _ensure_credentials(llm_settings=llm_settings)

    try:
        result = llm_smoke_tester.run(llm_settings=llm_settings)
    except LlmSmokeTestError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "LLM_SMOKE_TEST_FAILED",
                "message": str(exc),
            },
        ) from exc

    _persist_llm_settings(
        env_path=config_loader.repo_root / ".env",
        llm_settings=llm_settings,
    )

    return {
        "status": "ok",
        "provider": result.provider,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "source": llm_settings.source,
    }


def _resolve_llm_settings(
    smoke_test_request: LlmSmokeTestRequest,
    config_loader: ConfigLoader,
) -> LlmSettings:
    resolver = LlmSettingsResolver(config_loader=config_loader)
    try:
        return resolver.resolve(
            provider=smoke_test_request.provider,
            model=smoke_test_request.model,
            api_key=smoke_test_request.api_key,
            gateway_url=smoke_test_request.gateway_url,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "LLM_CONFIGURATION_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc


def _ensure_credentials(llm_settings: LlmSettings) -> None:
    if llm_settings.api_key or llm_settings.gateway_url:
        return

    raise HTTPException(
        status_code=503,
        detail={
            "error": "LLM_CREDENTIALS_REQUIRED",
            "message": "Provide an LLM API key or gateway URL.",
        },
    )


def _persist_llm_settings(env_path: Path, llm_settings: LlmSettings) -> None:
    env_updates = {
        "LLM_TYPE": llm_settings.provider,
        "LLM_MODEL": llm_settings.model,
        "LLM_API_KEY": llm_settings.api_key or "",
        "LLM_GATEWAY_URL": llm_settings.gateway_url or "",
    }
    existing_lines = (
        env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    )
    updated_keys: set[str] = set()
    next_lines: list[str] = []

    # Update only the LLM runtime keys so unrelated deployment/auth settings stay intact.
    for existing_line in existing_lines:
        env_key = _read_env_line_key(existing_line)
        if env_key in env_updates:
            next_lines.append(f"{env_key}={_format_env_value(env_updates[env_key])}")
            updated_keys.add(env_key)
            continue
        next_lines.append(existing_line)

    for env_key in LLM_ENV_KEYS:
        if env_key not in updated_keys:
            next_lines.append(f"{env_key}={_format_env_value(env_updates[env_key])}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    temp_env_path = env_path.with_name(f".{env_path.name}.tmp")
    temp_env_path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")
    temp_env_path.replace(env_path)


def _read_env_line_key(env_line: str) -> str | None:
    stripped_line = env_line.lstrip()
    if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
        return None
    env_key = stripped_line.split("=", 1)[0].strip()
    return env_key or None


def _format_env_value(env_value: str) -> str:
    escaped_value = env_value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped_value}"'
