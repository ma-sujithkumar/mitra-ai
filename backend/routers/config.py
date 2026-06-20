from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel

from backend.config_loader import ConfigLoader
from backend.dependencies import get_config_loader
from backend.dependencies import get_session_manager
from backend.session import SessionManager


router = APIRouter(prefix="/api/config", tags=["config"])

# File written into a session dir holding the user's page-1 advanced overrides.
# The pipeline reads this copy at invoke time (config.yaml-equivalent per run).
ADVANCED_OVERRIDES_FILENAME = "config_overrides.json"


# Type coercion + validation handlers keyed by spec "type" so we never branch
# with an if-else ladder when reading/casting params (CLAUDE.md rule 4/23).
def _coerce_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_enum_factory(allowed_choices: list[str]) -> Callable[[Any], str]:
    def _coerce(raw_value: Any) -> str:
        candidate = str(raw_value).strip().lower()
        if candidate not in allowed_choices:
            raise ValueError(f"must be one of {allowed_choices}")
        return candidate

    return _coerce


TYPE_COERCERS: dict[str, Callable[[Any], Any]] = {
    "int": lambda value: int(value),
    "float": lambda value: float(value),
    "bool": _coerce_bool,
    "str": lambda value: str(value),
}


# Single source of truth for every UI-surfaced advanced param. Each entry maps a
# config.ini section/option to a typed, labelled control. Secrets and absolute
# paths are intentionally excluded from this surface.
ADVANCED_PARAM_SPECS: list[dict[str, Any]] = [
    {
        "group": "Pipeline",
        "key": "pipeline.train_test_split",
        "default": 0.8,
        "section": "pipeline",
        "option": "TRAIN_TEST_SPLIT",
        "label": "Train/test split ratio",
        "type": "float",
        "minimum": 0.5,
        "maximum": 0.95,
    },
    {
        "group": "Pipeline",
        "key": "pipeline.max_ml_models",
        "default": 10,
        "section": "pipeline",
        "option": "MAX_ML_MODELS",
        "label": "Max candidate models",
        "type": "int",
        "minimum": 1,
        "maximum": 50,
    },
    {
        "group": "Pipeline",
        "key": "pipeline.max_hpt_trials",
        "default": 5,
        "section": "pipeline",
        "option": "MAX_HPT_TRIALS",
        "label": "Max HPT trials",
        "type": "int",
        "minimum": 1,
        "maximum": 200,
    },
    {
        "group": "Pipeline",
        "key": "pipeline.run_post_training_eval",
        "default": True,
        "section": "pipeline",
        "option": "RUN_POST_TRAINING_EVAL",
        "label": "Run SHAP + overfitting + HPT + judge after training",
        "type": "bool",
    },
    {
        "group": "Pipeline",
        "key": "pipeline.max_judge_turns",
        "default": 3,
        "section": "pipeline",
        "option": "MAX_JUDGE_TURNS",
        "label": "Max judge feedback turns",
        "type": "int",
        "minimum": 1,
        "maximum": 10,
    },
    {
        "group": "Training",
        "key": "training_api.default_execution_mode",
        "default": "ray",
        "section": "training_api",
        "option": "DEFAULT_EXECUTION_MODE",
        "label": "Execution mode",
        "type": "enum",
        "choices": ["ray", "local"],
    },
    {
        "group": "Training",
        "key": "training_api.max_concurrent_runs",
        "default": 2,
        "section": "training_api",
        "option": "MAX_CONCURRENT_RUNS",
        "label": "Max concurrent training runs",
        "type": "int",
        "minimum": 1,
        "maximum": 16,
    },
    {
        "group": "Hyperparameter Tuning",
        "key": "hpt.overfitting_gap_threshold",
        "default": 0.10,
        "section": "hpt",
        "option": "OVERFITTING_GAP_THRESHOLD",
        "label": "Overfitting gap threshold",
        "type": "float",
        "minimum": 0.0,
        "maximum": 1.0,
    },
    {
        "group": "Hyperparameter Tuning",
        "key": "hpt.val_split_ratio",
        "default": 0.2,
        "section": "hpt",
        "option": "VAL_SPLIT_RATIO",
        "label": "HPT validation split ratio",
        "type": "float",
        "minimum": 0.05,
        "maximum": 0.5,
    },
    {
        "group": "Hyperparameter Tuning",
        "key": "hpt.optuna_seed",
        "default": 42,
        "section": "hpt",
        "option": "OPTUNA_SEED",
        "label": "Optuna random seed",
        "type": "int",
        "minimum": 0,
        "maximum": 2_147_483_647,
    },
]

# Fast lookup from "section.option" key -> spec, built once at import time.
SPEC_BY_KEY: dict[str, dict[str, Any]] = {
    spec["key"]: spec for spec in ADVANCED_PARAM_SPECS
}


class AdvancedConfigUpdate(BaseModel):
    overrides: dict[str, Any]


def _coercer_for_spec(spec: dict[str, Any]) -> Callable[[Any], Any]:
    if spec["type"] == "enum":
        return _coerce_enum_factory(spec["choices"])
    return TYPE_COERCERS[spec["type"]]


def _base_value(config_loader: ConfigLoader, spec: dict[str, Any]) -> Any:
    # Read the raw value straight from the parser so we never duplicate the
    # config.ini -> dataclass mapping; coerce to the declared type. When the
    # section/option is absent (older config files), fall back to the declared
    # default so the param is still surfaced.
    raw_value = config_loader.parser.get(spec["section"], spec["option"], fallback="")
    if str(raw_value).strip() == "":
        return spec["default"]
    coerce = _coercer_for_spec(spec)
    return coerce(raw_value)


def _read_session_overrides(session_dir: Path) -> dict[str, Any]:
    overrides_path = session_dir / ADVANCED_OVERRIDES_FILENAME
    if not overrides_path.is_file():
        return {}
    return json.loads(overrides_path.read_text(encoding="utf-8"))


def _validate_range(spec: dict[str, Any], value: Any) -> None:
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    if minimum is not None and value < minimum:
        raise ValueError(f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"must be <= {maximum}")


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
            "base_urls": config_loader.llm_base_urls.as_provider_map(),
        },
        "metadata_agent": {
            "metadata_context_char_limit": (
                config_loader.metadata_agent.metadata_context_char_limit
            ),
        },
    }


@router.get("/advanced")
def advanced_config(
    session_id: str | None = None,
    config_loader: ConfigLoader = Depends(get_config_loader),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, object]:
    """Return every UI-surfaced advanced param with its effective value.

    The effective value is the config.ini base overlaid with any per-session
    overrides previously saved via PUT (so the UI shows what the next run uses).
    """
    session_overrides: dict[str, Any] = {}
    if session_id:
        session_dir = session_manager.get_session_path(session_id=session_id)
        if session_dir.is_dir():
            session_overrides = _read_session_overrides(session_dir)

    params: list[dict[str, Any]] = []
    for spec in ADVANCED_PARAM_SPECS:
        effective_value = session_overrides.get(spec["key"], _base_value(config_loader, spec))
        param_view = {
            "key": spec["key"],
            "group": spec["group"],
            "label": spec["label"],
            "type": spec["type"],
            "value": effective_value,
        }
        if spec["type"] == "enum":
            param_view["choices"] = spec["choices"]
        if "minimum" in spec:
            param_view["minimum"] = spec["minimum"]
        if "maximum" in spec:
            param_view["maximum"] = spec["maximum"]
        params.append(param_view)

    return {"session_id": session_id, "params": params}


@router.put("/advanced")
def update_advanced_config(
    payload: AdvancedConfigUpdate,
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, object]:
    """Validate and persist advanced overrides into the session dir.

    Unknown keys, wrong types, or out-of-range values are rejected. The saved
    file is the per-run config the pipeline reads at invoke time.
    """
    session_dir = session_manager.get_session_path(session_id=session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail={"message": f"Unknown session: {session_id}"})

    validated_overrides: dict[str, Any] = {}
    rejected: dict[str, str] = {}
    for override_key, raw_value in payload.overrides.items():
        spec = SPEC_BY_KEY.get(override_key)
        if spec is None:
            rejected[override_key] = "unknown parameter"
            continue
        coerce = _coercer_for_spec(spec)
        # Coerce + range-check; collect a readable reason instead of 500ing.
        try:
            coerced_value = coerce(raw_value)
            _validate_range(spec, coerced_value)
        except (ValueError, TypeError) as coercion_error:  # noqa: BLE001 - reported to caller
            rejected[override_key] = str(coercion_error)
            continue
        validated_overrides[override_key] = coerced_value

    if rejected:
        raise HTTPException(
            status_code=422,
            detail={"message": "Invalid advanced config", "rejected": rejected},
        )

    overrides_path = session_dir / ADVANCED_OVERRIDES_FILENAME
    overrides_path.write_text(
        json.dumps(validated_overrides, indent=2),
        encoding="utf-8",
    )
    return {"session_id": session_id, "saved": validated_overrides}
