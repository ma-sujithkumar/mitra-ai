"""Unit tests for backend.agents.evaluation.shap.session_context."""

from backend.agents.evaluation.shap.session_context import (
    ExecutionStatus,
    ModelNameValidationStatus,
    SessionContext,
)


def _build_minimal_session_context() -> SessionContext:
    return SessionContext(
        session_id="session_001",
        supplied_model_name="xgboost",
        pickle_file_path="/models/model.pkl",
        engineered_dataset_path="/data/engineered_dataset.csv",
    )


def test_required_fields_are_stored() -> None:
    session_context = _build_minimal_session_context()

    assert session_context.session_id == "session_001"
    assert session_context.supplied_model_name == "xgboost"
    assert session_context.pickle_file_path == "/models/model.pkl"
    assert session_context.engineered_dataset_path == "/data/engineered_dataset.csv"


def test_optional_fields_default_to_none_or_empty() -> None:
    session_context = _build_minimal_session_context()

    assert session_context.detected_model_type is None
    assert session_context.model_name_validation_status is None
    assert session_context.model_name_validation_message is None
    assert session_context.target_column_name is None
    assert session_context.feature_names is None
    assert session_context.num_samples is None
    assert session_context.num_features is None
    assert session_context.explainer_name is None
    assert session_context.shap_values is None
    assert session_context.global_feature_importance is None
    assert session_context.feature_shap_mapping is None
    assert session_context.warnings == []
    assert session_context.error_message is None
    assert session_context.extra_metadata == {}
    assert session_context.execution_status == ExecutionStatus.RUNNING


def test_two_instances_do_not_share_mutable_defaults() -> None:
    first_session_context = _build_minimal_session_context()
    second_session_context = _build_minimal_session_context()

    first_session_context.add_warning("first warning")
    first_session_context.extra_metadata["key"] = "value"

    assert second_session_context.warnings == []
    assert second_session_context.extra_metadata == {}


def test_add_warning_appends_message_and_promotes_status() -> None:
    session_context = _build_minimal_session_context()

    session_context.add_warning("Provided model name differs from detected model type.")

    assert session_context.warnings == [
        "Provided model name differs from detected model type."
    ]
    assert session_context.execution_status == ExecutionStatus.WARNING


def test_add_warning_accumulates_multiple_messages() -> None:
    session_context = _build_minimal_session_context()

    session_context.add_warning("first warning")
    session_context.add_warning("second warning")

    assert session_context.warnings == ["first warning", "second warning"]


def test_add_warning_does_not_downgrade_failed_status() -> None:
    session_context = _build_minimal_session_context()
    session_context.mark_failed("Unsupported model type.")

    session_context.add_warning("This should not change the failed status.")

    assert session_context.execution_status == ExecutionStatus.FAILED


def test_mark_failed_sets_status_and_error_message() -> None:
    session_context = _build_minimal_session_context()

    session_context.mark_failed("Model artifact could not be loaded.")

    assert session_context.execution_status == ExecutionStatus.FAILED
    assert session_context.error_message == "Model artifact could not be loaded."
    assert session_context.has_failed() is True


def test_mark_success_from_running_sets_success() -> None:
    session_context = _build_minimal_session_context()

    session_context.mark_success()

    assert session_context.execution_status == ExecutionStatus.SUCCESS


def test_mark_success_after_warning_keeps_warning_status() -> None:
    session_context = _build_minimal_session_context()
    session_context.add_warning("Mismatch detected.")

    session_context.mark_success()

    assert session_context.execution_status == ExecutionStatus.WARNING


def test_mark_success_after_failed_keeps_failed_status() -> None:
    session_context = _build_minimal_session_context()
    session_context.mark_failed("Could not read dataset.")

    session_context.mark_success()

    assert session_context.execution_status == ExecutionStatus.FAILED


def test_has_failed_is_false_when_running_or_succeeded() -> None:
    session_context = _build_minimal_session_context()
    assert session_context.has_failed() is False

    session_context.mark_success()
    assert session_context.has_failed() is False


def test_model_name_validation_status_enum_values() -> None:
    assert ModelNameValidationStatus.MATCH.value == "match"
    assert ModelNameValidationStatus.MISMATCH.value == "mismatch"
    assert ModelNameValidationStatus.UNDETECTABLE.value == "undetectable"
    assert ModelNameValidationStatus.UNSUPPORTED.value == "unsupported"


def test_execution_status_enum_values() -> None:
    assert ExecutionStatus.RUNNING.value == "running"
    assert ExecutionStatus.SUCCESS.value == "success"
    assert ExecutionStatus.WARNING.value == "warning"
    assert ExecutionStatus.FAILED.value == "failed"
