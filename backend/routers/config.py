from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends

from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader


router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/public")
def public_config(
    config_loader: ConfigLoader = Depends(get_config_loader),
) -> dict[str, object]:
    return {
        "upload": {
            "allowed_extensions": config_loader.upload.allowed_extensions,
            "max_file_size_mb": config_loader.upload.max_file_size_mb,
            "recent_upload_limit": config_loader.upload.recent_upload_limit,
        },
        "pipeline": {
            "train_test_split": config_loader.pipeline.train_test_split,
            "max_ml_models": config_loader.pipeline.max_ml_models,
            "max_hpt_trials": config_loader.pipeline.max_hpt_trials,
        },
        "llm": {
            "providers": ["openai", "anthropic", "gemini"],
            "base_models": config_loader.llm_models.as_provider_map(),
        },
        "metadata_agent": {
            "metadata_context_char_limit": (
                config_loader.metadata_agent.metadata_context_char_limit
            ),
        },
    }
