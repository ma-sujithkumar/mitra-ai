# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
import difflib
from typing import List

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, field_validator, model_validator

from core.config_loader import EXPECTED_MODELS
from core.data_bundle import DataBundle, CommonData


VALID_MODEL_NAMES: List[str] = EXPECTED_MODELS
VALID_TRAINING_MODES: List[str] = ["full_train", "fine_tune"]


class MLKitInputValidator(PydanticBaseModel):
    """Validates MLKit constructor arguments before any model is instantiated."""

    model_name: str
    training_mode: str

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("model_name")
    @classmethod
    def check_model_name(cls, model_name_value: str) -> str:
        if model_name_value not in VALID_MODEL_NAMES:
            close_matches = difflib.get_close_matches(
                model_name_value, VALID_MODEL_NAMES, n=3, cutoff=0.6
            )
            suggestion_text = (
                f" Did you mean: {close_matches}?" if close_matches else ""
            )
            raise ValueError(
                f"Unknown model name '{model_name_value}'.{suggestion_text} "
                f"Use one of the {len(VALID_MODEL_NAMES)} registered model names "
                f"(see README.md for the full list)."
            )
        return model_name_value

    @field_validator("training_mode")
    @classmethod
    def check_training_mode(cls, training_mode_value: str) -> str:
        if training_mode_value not in VALID_TRAINING_MODES:
            raise ValueError(
                f"Invalid training_mode '{training_mode_value}'. "
                f"Must be one of: {VALID_TRAINING_MODES}"
            )
        return training_mode_value


class CommonDataValidator(PydanticBaseModel):
    """Validates shape consistency of CommonData arrays."""

    X_train: object
    y_train: object
    X_test: object
    y_test: object

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_array_shapes(self) -> "CommonDataValidator":
        X_train_arr = np.asarray(self.X_train)
        y_train_arr = np.asarray(self.y_train)
        X_test_arr = np.asarray(self.X_test)
        y_test_arr = np.asarray(self.y_test)

        if X_train_arr.ndim != 2:
            raise ValueError(
                f"X_train must be 2D (n_samples, n_features), got shape {X_train_arr.shape}"
            )
        if X_test_arr.ndim != 2:
            raise ValueError(
                f"X_test must be 2D (n_samples, n_features), got shape {X_test_arr.shape}"
            )
        if X_train_arr.shape[0] != y_train_arr.shape[0]:
            raise ValueError(
                f"X_train and y_train sample count mismatch: "
                f"{X_train_arr.shape[0]} vs {y_train_arr.shape[0]}"
            )
        if X_test_arr.shape[0] != y_test_arr.shape[0]:
            raise ValueError(
                f"X_test and y_test sample count mismatch: "
                f"{X_test_arr.shape[0]} vs {y_test_arr.shape[0]}"
            )
        if X_train_arr.shape[1] != X_test_arr.shape[1]:
            raise ValueError(
                f"X_train and X_test feature count mismatch: "
                f"{X_train_arr.shape[1]} vs {X_test_arr.shape[1]}"
            )
        return self


def validate_model_name(model_name: str, training_mode: str = "fine_tune") -> None:
    """Raises ValueError with suggestions if model_name or training_mode is invalid."""
    MLKitInputValidator(model_name=model_name, training_mode=training_mode)


def validate_data_bundle(data: DataBundle) -> None:
    """Raises ValueError if DataBundle shapes are inconsistent."""
    CommonDataValidator(
        X_train=data.common.X_train,
        y_train=data.common.y_train,
        X_test=data.common.X_test,
        y_test=data.common.y_test,
    )
