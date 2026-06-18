"""Unit tests for shap_explainability.validators.model_validator."""

import json
import uuid
from pathlib import Path

import pytest

from shap_explainability.errors import ModelValidationError
from shap_explainability.session_context import ExecutionStatus, ModelNameValidationStatus
from shap_explainability.validators.model_validator import ModelValidationResult, ModelValidator
from tests.fixtures.fixture_factory import FixtureFactory


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _write_model_type_config(directory: Path) -> Path:
    """Writes a complete model type detection JSON config including alias entries."""
    config_directory = directory / "config"
    config_directory.mkdir(parents=True, exist_ok=True)
    config_path = config_directory / "model_type_detection.json"
    detection_config = {
        "class_name_to_model_family": {
            "XGBClassifier": "XGBoost",
            "XGBRegressor": "XGBoost",
            "RandomForestClassifier": "RandomForest",
            "RandomForestRegressor": "RandomForest",
            "LGBMClassifier": "LightGBM",
            "LGBMRegressor": "LightGBM",
            "CatBoostClassifier": "CatBoost",
            "CatBoostRegressor": "CatBoost",
            "LogisticRegression": "LogisticRegression",
        },
        "supplied_name_to_family": {
            "xgboost": "XGBoost",
            "xgb": "XGBoost",
            "randomforest": "RandomForest",
            "random_forest": "RandomForest",
            "rf": "RandomForest",
            "lightgbm": "LightGBM",
            "lgbm": "LightGBM",
            "catboost": "CatBoost",
            "logisticregression": "LogisticRegression",
            "logistic_regression": "LogisticRegression",
            "logreg": "LogisticRegression",
            "lr": "LogisticRegression",
        },
    }
    config_path.write_text(json.dumps(detection_config), encoding="utf-8")
    return config_path


def _make_validator(tmp_path: Path) -> ModelValidator:
    """Creates a ModelValidator with a local config and test logger."""
    config_path = _write_model_type_config(tmp_path)
    logger = FixtureFactory.make_execution_logger(tmp_path)
    return ModelValidator(execution_logger=logger, model_type_config_path=config_path)


# ---------------------------------------------------------------------------
# Rule 1: Supplied name matches detected family
# ---------------------------------------------------------------------------

def test_exact_supplied_name_match_returns_match_status(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="xgboost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    result = validator.validate("xgboost", loaded_model, session_context)

    assert result.status == ModelNameValidationStatus.MATCH
    assert result.model_family == "XGBoost"


def test_case_insensitive_supplied_name_matches(tmp_path: Path) -> None:
    """Supplied name 'XGBoost' (mixed case) should still match family 'XGBoost'."""
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="XGBoost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    result = validator.validate("XGBoost", loaded_model, session_context)

    assert result.status == ModelNameValidationStatus.MATCH


def test_alias_supplied_name_matches_family(tmp_path: Path) -> None:
    """Supplied name 'xgb' should map to XGBoost family via the alias table."""
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="xgb")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    result = validator.validate("xgb", loaded_model, session_context)

    assert result.status == ModelNameValidationStatus.MATCH


def test_match_sets_match_status_on_session_context(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="xgboost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    validator.validate("xgboost", loaded_model, session_context)

    assert session_context.model_name_validation_status == ModelNameValidationStatus.MATCH
    assert session_context.model_name_validation_message is not None


def test_match_does_not_add_warning_to_session_context(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="xgboost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    validator.validate("xgboost", loaded_model, session_context)

    assert len(session_context.warnings) == 0
    assert session_context.execution_status == ExecutionStatus.RUNNING


# ---------------------------------------------------------------------------
# Rule 2: Supplied name differs from detected family (non-terminating WARNING)
# ---------------------------------------------------------------------------

def test_mismatched_supplied_name_returns_mismatch_status(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="catboost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    result = validator.validate("catboost", loaded_model, session_context)

    assert result.status == ModelNameValidationStatus.MISMATCH
    assert result.model_family == "XGBoost"


def test_mismatch_does_not_raise(tmp_path: Path) -> None:
    """Rule 2 must never raise — it is a non-terminating WARNING (spec.md Sec 8 Rule 2)."""
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="catboost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    result = validator.validate("catboost", loaded_model, session_context)

    assert result is not None


def test_mismatch_adds_warning_to_session_context(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="catboost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    validator.validate("catboost", loaded_model, session_context)

    assert len(session_context.warnings) == 1
    assert "catboost" in session_context.warnings[0].lower()


def test_mismatch_sets_warning_execution_status_not_failed(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="catboost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    validator.validate("catboost", loaded_model, session_context)

    assert session_context.execution_status == ExecutionStatus.WARNING
    assert not session_context.has_failed()


def test_mismatch_sets_mismatch_status_on_session_context(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="catboost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    validator.validate("catboost", loaded_model, session_context)

    assert (
        session_context.model_name_validation_status == ModelNameValidationStatus.MISMATCH
    )


# ---------------------------------------------------------------------------
# Rule 3: Model type cannot be determined (terminating)
# ---------------------------------------------------------------------------

def test_empty_detected_class_name_raises_model_validation_error(tmp_path: Path) -> None:
    """Rule 3: empty detected_class_name means type is undetectable."""
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="", model_family=None
    )

    with pytest.raises(ModelValidationError):
        validator.validate("xgboost", loaded_model, session_context)


def test_undetectable_type_marks_session_context_failed(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="", model_family=None
    )

    with pytest.raises(ModelValidationError):
        validator.validate("xgboost", loaded_model, session_context)

    assert session_context.has_failed()
    assert (
        session_context.model_name_validation_status
        == ModelNameValidationStatus.UNDETECTABLE
    )


# ---------------------------------------------------------------------------
# Rule 4: Detected model type is unsupported (terminating)
# ---------------------------------------------------------------------------

def test_none_model_family_raises_model_validation_error(tmp_path: Path) -> None:
    """Rule 4: model_family=None means the class is not in the supported map."""
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="SomeUnsupportedClassifier", model_family=None
    )

    with pytest.raises(ModelValidationError, match="not supported"):
        validator.validate("xgboost", loaded_model, session_context)


def test_unsupported_family_marks_session_context_failed(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="SomeUnsupportedClassifier", model_family=None
    )

    with pytest.raises(ModelValidationError):
        validator.validate("xgboost", loaded_model, session_context)

    assert session_context.has_failed()


def test_unsupported_family_sets_unsupported_status_on_session_context(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="SomeUnsupportedClassifier", model_family=None
    )

    with pytest.raises(ModelValidationError):
        validator.validate("xgboost", loaded_model, session_context)

    assert (
        session_context.model_name_validation_status == ModelNameValidationStatus.UNSUPPORTED
    )


# ---------------------------------------------------------------------------
# All five supported model families — Rule 1 match
# ---------------------------------------------------------------------------

def test_random_forest_supplied_name_matches(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="randomforest")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="RandomForestClassifier", model_family="RandomForest"
    )

    result = validator.validate("randomforest", loaded_model, session_context)

    assert result.status == ModelNameValidationStatus.MATCH


def test_lightgbm_supplied_name_matches(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="lightgbm")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="LGBMClassifier", model_family="LightGBM"
    )

    result = validator.validate("lightgbm", loaded_model, session_context)

    assert result.status == ModelNameValidationStatus.MATCH


def test_logistic_regression_alias_logreg_matches(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="logreg")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="LogisticRegression", model_family="LogisticRegression"
    )

    result = validator.validate("logreg", loaded_model, session_context)

    assert result.status == ModelNameValidationStatus.MATCH


# ---------------------------------------------------------------------------
# Return value correctness
# ---------------------------------------------------------------------------

def test_validation_result_contains_detected_family(tmp_path: Path) -> None:
    """The returned ModelValidationResult.model_family reflects the detected family."""
    validator = _make_validator(tmp_path)
    session_context = FixtureFactory.make_session_context(supplied_model_name="xgboost")
    loaded_model = FixtureFactory.make_loaded_model(
        detected_class_name="XGBClassifier", model_family="XGBoost"
    )

    result = validator.validate("xgboost", loaded_model, session_context)

    assert isinstance(result, ModelValidationResult)
    assert result.model_family == "XGBoost"
    assert result.message != ""
