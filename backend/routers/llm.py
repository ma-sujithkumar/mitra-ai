from __future__ import annotations

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
    print(llm_settings)
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
