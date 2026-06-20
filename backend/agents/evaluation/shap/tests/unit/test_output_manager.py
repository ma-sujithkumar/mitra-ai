"""Unit tests for backend.agents.evaluation.shap.utils.output_manager."""

from pathlib import Path

import pytest

from backend.agents.evaluation.shap.utils.output_manager import OutputManager, OutputManagerError


def test_initialize_creates_session_and_artifact_subfolders(tmp_path: Path) -> None:
    output_manager = OutputManager(output_root=tmp_path, session_id="session_001")

    output_manager.initialize()

    session_directory = tmp_path / "session_001"
    assert session_directory.is_dir()
    assert (session_directory / "plots").is_dir()
    assert (session_directory / "csv").is_dir()
    assert (session_directory / "metadata").is_dir()
    assert (session_directory / "logs").is_dir()


def test_initialize_is_idempotent(tmp_path: Path) -> None:
    output_manager = OutputManager(output_root=tmp_path, session_id="session_001")

    output_manager.initialize()
    output_manager.initialize()

    assert (tmp_path / "session_001" / "plots").is_dir()


def test_initialize_creates_output_root_if_missing(tmp_path: Path) -> None:
    output_root = tmp_path / "does_not_exist_yet"
    output_manager = OutputManager(output_root=output_root, session_id="session_001")

    output_manager.initialize()

    assert (output_root / "session_001").is_dir()


def test_plot_path_returns_path_under_plots_subfolder(tmp_path: Path) -> None:
    output_manager = OutputManager(output_root=tmp_path, session_id="session_001")

    plot_path = output_manager.plot_path("summary_plot.png")

    assert plot_path == tmp_path / "session_001" / "plots" / "summary_plot.png"


def test_csv_path_returns_path_under_csv_subfolder(tmp_path: Path) -> None:
    output_manager = OutputManager(output_root=tmp_path, session_id="session_001")

    csv_path = output_manager.csv_path("global_feature_importance.csv")

    assert csv_path == tmp_path / "session_001" / "csv" / "global_feature_importance.csv"


def test_metadata_path_returns_metadata_json_under_metadata_subfolder(tmp_path: Path) -> None:
    output_manager = OutputManager(output_root=tmp_path, session_id="session_001")

    metadata_path = output_manager.metadata_path()

    assert metadata_path == tmp_path / "session_001" / "metadata" / "metadata.json"


def test_log_path_returns_execution_log_under_logs_subfolder(tmp_path: Path) -> None:
    output_manager = OutputManager(output_root=tmp_path, session_id="session_001")

    log_path = output_manager.log_path()

    assert log_path == tmp_path / "session_001" / "logs" / "execution.log"


def test_different_sessions_do_not_collide(tmp_path: Path) -> None:
    first_output_manager = OutputManager(output_root=tmp_path, session_id="session_001")
    second_output_manager = OutputManager(output_root=tmp_path, session_id="session_002")

    first_output_manager.initialize()
    second_output_manager.initialize()

    assert (tmp_path / "session_001").is_dir()
    assert (tmp_path / "session_002").is_dir()
    assert first_output_manager.session_directory != second_output_manager.session_directory


def test_initialize_raises_output_manager_error_when_path_blocked_by_file(tmp_path: Path) -> None:
    blocking_file_path = tmp_path / "session_001"
    blocking_file_path.write_text("this is a file, not a directory", encoding="utf-8")
    output_manager = OutputManager(output_root=tmp_path, session_id="session_001")

    with pytest.raises(OutputManagerError):
        output_manager.initialize()
