from __future__ import annotations

from time import time

from dotenv import dotenv_values
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request

from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader


router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health_check(
    request: Request,
    config_loader: ConfigLoader = Depends(get_config_loader),
) -> dict[str, object]:
    env_settings = dotenv_values(config_loader.repo_root / ".env")
    provider = _first_non_blank(env_settings.get("LLM_TYPE"))
    has_credentials = (
        _first_non_blank(env_settings.get("LLM_API_KEY")) is not None
        or _first_non_blank(env_settings.get("LLM_GATEWAY_URL")) is not None
    )
    started_at_epoch = float(request.app.state.started_at_epoch)

    return {
        "status": "ok",
        "uptime_seconds": max(0.0, round(time() - started_at_epoch, 3)),
        "llm": {
            "provider": provider,
            "env_configured": provider is not None and has_credentials,
        },
    }


def _first_non_blank(value: str | None) -> str | None:
    if value is not None and value.strip():
        return value.strip()
    return None
