"""Unit tests for backend.agents.evaluation.shap.utils.logger."""

import logging
import uuid
from pathlib import Path

import pytest

from backend.agents.evaluation.shap.utils.logger import ExecutionLogger


def _unique_session_id() -> str:
    """Generates a unique session_id so tests do not share logging.Logger state."""
    return f"test-session-{uuid.uuid4().hex}"


def test_log_file_is_created_at_given_path(tmp_path: Path) -> None:
    log_file_path = tmp_path / "logs" / "execution.log"

    ExecutionLogger(_unique_session_id(), log_file_path)

    assert log_file_path.exists()


def test_log_directory_is_created_if_missing(tmp_path: Path) -> None:
    log_file_path = tmp_path / "nested" / "logs" / "execution.log"

    ExecutionLogger(_unique_session_id(), log_file_path)

    assert log_file_path.parent.is_dir()


def test_log_execution_start_writes_event_and_message(tmp_path: Path) -> None:
    log_file_path = tmp_path / "execution.log"
    execution_logger = ExecutionLogger(_unique_session_id(), log_file_path)

    execution_logger.log_execution_start("Pipeline execution started.")

    log_content = log_file_path.read_text(encoding="utf-8")
    assert "execution_start" in log_content
    assert "Pipeline execution started." in log_content


@pytest.mark.parametrize(
    "method_name,event_name",
    [
        ("log_dataset_validation", "dataset_validation"),
        ("log_model_validation", "model_validation"),
        ("log_model_loading", "model_loading"),
        ("log_model_type_detection", "model_type_detection"),
        ("log_model_name_validation", "model_name_validation"),
        ("log_schema_validation", "schema_validation"),
        ("log_explainer_selection", "explainer_selection"),
        ("log_shap_generation", "shap_generation"),
        ("log_plot_generation", "plot_generation"),
        ("log_csv_export", "csv_export"),
        ("log_metadata_generation", "metadata_generation"),
        ("log_execution_completion", "execution_completion"),
    ],
)
def test_each_named_method_logs_its_event_category(
    tmp_path: Path, method_name: str, event_name: str
) -> None:
    log_file_path = tmp_path / "execution.log"
    execution_logger = ExecutionLogger(_unique_session_id(), log_file_path, log_level="DEBUG")
    logging_method = getattr(execution_logger, method_name)

    logging_method("sample message")

    log_content = log_file_path.read_text(encoding="utf-8")
    assert event_name in log_content
    assert "sample message" in log_content


def test_log_execution_failure_logs_at_error_level(tmp_path: Path) -> None:
    log_file_path = tmp_path / "execution.log"
    execution_logger = ExecutionLogger(_unique_session_id(), log_file_path, log_level="ERROR")

    execution_logger.log_execution_failure("Model artifact could not be loaded.")

    log_content = log_file_path.read_text(encoding="utf-8")
    assert "execution_failure" in log_content
    assert "ERROR" in log_content
    assert "Model artifact could not be loaded." in log_content


def test_log_level_filters_out_lower_severity_messages(tmp_path: Path) -> None:
    log_file_path = tmp_path / "execution.log"
    execution_logger = ExecutionLogger(_unique_session_id(), log_file_path, log_level="ERROR")

    execution_logger.log_dataset_validation("This should be filtered out.")

    log_content = log_file_path.read_text(encoding="utf-8")
    assert "This should be filtered out." not in log_content


def test_model_name_validation_can_be_logged_as_warning(tmp_path: Path) -> None:
    log_file_path = tmp_path / "execution.log"
    execution_logger = ExecutionLogger(_unique_session_id(), log_file_path, log_level="INFO")

    execution_logger.log_model_name_validation(
        "Provided model name differs from detected model type.", level=logging.WARNING
    )

    log_content = log_file_path.read_text(encoding="utf-8")
    assert "WARNING" in log_content
    assert "model_name_validation" in log_content


def test_invalid_log_level_raises_value_error(tmp_path: Path) -> None:
    log_file_path = tmp_path / "execution.log"

    with pytest.raises(ValueError, match="log_level"):
        ExecutionLogger(_unique_session_id(), log_file_path, log_level="TRACE")


def test_constructing_logger_twice_does_not_duplicate_log_lines(tmp_path: Path) -> None:
    log_file_path = tmp_path / "execution.log"
    session_id = _unique_session_id()

    ExecutionLogger(session_id, log_file_path)
    second_execution_logger = ExecutionLogger(session_id, log_file_path)
    second_execution_logger.log_execution_start("started once")

    log_content = log_file_path.read_text(encoding="utf-8")
    assert log_content.count("started once") == 1
