"""Validate the supplied model_name against the detected model type.

Implements spec.md Section 8 Rules 1-4. The supplied model_name is treated as
integration metadata only; the detected model type always takes precedence for
SHAP explainer selection (spec.md Sec 4.2, Sec 8).

Rules enforced here:
  Rule 1 - Supplied name matches detected family: silent pass, continue.
  Rule 2 - Supplied name differs from detected family: non-terminating WARNING,
           recorded on SessionContext and in execution log.
  Rule 3 - Model type cannot be determined: terminating ModelValidationError.
  Rule 4 - Detected model type is unsupported: terminating ModelValidationError.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.agents.evaluation.shap.errors import ModelValidationError
from backend.agents.evaluation.shap.loaders.model_loader import LoadedModel
from backend.agents.evaluation.shap.session_context import ModelNameValidationStatus, SessionContext
from backend.agents.evaluation.shap.utils.logger import ExecutionLogger

_MODEL_TYPE_CONFIG_FILENAME: str = "model_type_detection.json"
_CLASS_NAME_TO_FAMILY_KEY: str = "class_name_to_model_family"
_SUPPLIED_NAME_TO_FAMILY_KEY: str = "supplied_name_to_family"


@dataclass(frozen=True)
class ModelValidationResult:
    """Structured outcome of Spec Sec 8 model name validation (Rules 1-4).

    Attributes:
        status: The ModelNameValidationStatus enum value from the validation.
        message: Human-readable description of the validation outcome.
        model_family: The confirmed supported model family, or None when Rules 3/4
            terminate execution before a family can be confirmed.
    """

    status: ModelNameValidationStatus
    message: str
    model_family: Optional[str]


class ModelValidator:
    """Validates the supplied model_name against the detected model type.

    Responsibilities (spec.md Sec 8, architecture.md Section 3 and 4):
      - Apply Rules 1-4 from Sec 8 to the (supplied_model_name, detected_model_type) pair.
      - Raise ModelValidationError and mark SessionContext as failed for Rules 3 and 4.
      - Log a non-terminating WARNING and update SessionContext for Rule 2 without raising.
      - Use a config-driven lookup table for supplied-name-to-family matching (CLAUDE.md rule 4).
    """

    def __init__(
        self,
        execution_logger: ExecutionLogger,
        model_type_config_path: Optional[Path] = None,
    ) -> None:
        """Initializes the ModelValidator.

        Args:
            execution_logger: Session-scoped logger for recording Sec 19 events.
            model_type_config_path: Path to the model type detection JSON config.
                Defaults to epic_4_shap/config/model_type_detection.json resolved
                relative to this module's location.

        Raises:
            ModelValidationError: If the model type config file cannot be found or parsed.
        """
        self._execution_logger: ExecutionLogger = execution_logger
        resolved_config_path: Path = (
            model_type_config_path
            if model_type_config_path is not None
            else self._default_model_type_config_path()
        )
        raw_config: dict = self._load_type_detection_config(resolved_config_path)
        self._supported_families: frozenset[str] = frozenset(
            raw_config.get(_CLASS_NAME_TO_FAMILY_KEY, {}).values()
        )
        # Normalize all alias keys to lowercase at init time so matching is case-insensitive.
        self._supplied_name_to_family: dict[str, str] = {
            alias_name.lower(): family_name
            for alias_name, family_name in raw_config.get(
                _SUPPLIED_NAME_TO_FAMILY_KEY, {}
            ).items()
        }

    def validate(
        self,
        supplied_model_name: str,
        loaded_model: LoadedModel,
        session_context: SessionContext,
    ) -> ModelValidationResult:
        """Applies Spec Sec 8 Rules 1-4 against the loaded model artifact.

        Terminating failures (Rules 3 and 4) mark session_context as failed and
        raise ModelValidationError. The non-terminating mismatch (Rule 2) adds a
        warning to session_context and returns normally. Rule 1 returns silently.

        Args:
            supplied_model_name: The model_name field from the integration payload (Sec 4.2).
            loaded_model: The deserialized model artifact from ModelLoader.
            session_context: Mutable pipeline state; validation status and any warning
                are written here before returning.

        Returns:
            ModelValidationResult with the validation outcome for all non-terminating paths.

        Raises:
            ModelValidationError: For Rule 3 (undetectable type) or Rule 4 (unsupported type).
        """
        # Rule 3: Model type cannot be determined (detected_class_name absent or empty).
        if not loaded_model.detected_class_name:
            failure_message = (
                "Model type could not be determined from the loaded artifact. "
                "The model class name is unavailable."
            )
            self._execution_logger.log_model_name_validation(
                failure_message, logging.ERROR
            )
            session_context.model_name_validation_status = (
                ModelNameValidationStatus.UNDETECTABLE
            )
            session_context.model_name_validation_message = failure_message
            session_context.mark_failed(failure_message)
            raise ModelValidationError(failure_message)

        # Rule 4: Detected model type is not in the supported families map.
        if loaded_model.model_family is None:
            failure_message = (
                f"Detected model type '{loaded_model.detected_class_name}' is not supported. "
                f"Supported model families: {sorted(self._supported_families)}."
            )
            self._execution_logger.log_model_name_validation(
                failure_message, logging.ERROR
            )
            session_context.model_name_validation_status = (
                ModelNameValidationStatus.UNSUPPORTED
            )
            session_context.model_name_validation_message = failure_message
            session_context.mark_failed(failure_message)
            raise ModelValidationError(failure_message)

        # Rules 1 and 2: Compare supplied name against the detected model family.
        detected_family: str = loaded_model.model_family
        names_match: bool = self._supplied_name_matches_family(
            supplied_model_name, detected_family
        )

        if names_match:
            # Rule 1: Supplied name matches detected family — continue silently.
            match_message = (
                f"Supplied model name '{supplied_model_name}' matches detected "
                f"model family '{detected_family}'."
            )
            self._execution_logger.log_model_name_validation(match_message)
            session_context.model_name_validation_status = ModelNameValidationStatus.MATCH
            session_context.model_name_validation_message = match_message
            return ModelValidationResult(
                status=ModelNameValidationStatus.MATCH,
                message=match_message,
                model_family=detected_family,
            )

        # Rule 2: Supplied name differs from detected family — WARNING only, never terminates.
        mismatch_message = (
            f"Supplied model name '{supplied_model_name}' differs from detected model "
            f"type '{loaded_model.detected_class_name}' (family: '{detected_family}'). "
            "Detected model type used for SHAP explainer selection. "
            "Supplied name retained for metadata and traceability."
        )
        self._execution_logger.log_model_name_validation(mismatch_message, logging.WARNING)
        session_context.model_name_validation_status = ModelNameValidationStatus.MISMATCH
        session_context.model_name_validation_message = mismatch_message
        session_context.add_warning(mismatch_message)
        return ModelValidationResult(
            status=ModelNameValidationStatus.MISMATCH,
            message=mismatch_message,
            model_family=detected_family,
        )

    def _supplied_name_matches_family(
        self, supplied_model_name: str, detected_family: str
    ) -> bool:
        """Determines if the supplied name corresponds to the detected model family.

        Uses the config-driven supplied_name_to_family alias map first (CLAUDE.md
        rule 4). Falls back to a direct case-insensitive comparison against the
        family name itself to handle exact matches like "XGBoost" vs "XGBoost".

        Args:
            supplied_model_name: The raw value from the integration payload.
            detected_family: The model family resolved by ModelLoader.

        Returns:
            True if the supplied name maps to the detected family, False otherwise.
        """
        normalized_supplied_name: str = supplied_model_name.strip().lower()
        mapped_family: Optional[str] = self._supplied_name_to_family.get(
            normalized_supplied_name
        )
        if mapped_family is not None:
            return mapped_family == detected_family
        # Direct case-insensitive fallback for names not explicitly aliased.
        return normalized_supplied_name == detected_family.lower()

    @staticmethod
    def _load_type_detection_config(config_path: Path) -> dict:
        """Loads and parses the model type detection JSON config.

        Raises:
            ModelValidationError: If the config file is missing or not valid JSON.
        """
        if not config_path.is_file():
            raise ModelValidationError(
                f"Model type detection config not found: {config_path}"
            )
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)

    @staticmethod
    def _default_model_type_config_path() -> Path:
        """Returns the default config path resolved relative to this module's location.

        model_validator.py is at: epic_4_shap/src/backend.agents.evaluation.shap/validators/
        config/ is at:            epic_4_shap/config/
        """
        return (
            Path(__file__).resolve().parent.parent
            / "config"
            / _MODEL_TYPE_CONFIG_FILENAME
        )
