"""Unit tests for backend.agents.evaluation.shap.config_loader."""

from pathlib import Path

import pytest

from backend.agents.evaluation.shap.config_loader import AppConfig, ConfigLoader, ConfigValidationError


def test_load_returns_valid_app_config(config_ini_writer, existing_python_path: str, tmp_path: Path) -> None:
    config_file_path = config_ini_writer(python_path=existing_python_path)

    app_config = ConfigLoader(config_file_path).load()

    assert isinstance(app_config, AppConfig)
    assert app_config.python_path == Path(existing_python_path)
    assert app_config.output_root == (tmp_path / "outputs").resolve()
    assert app_config.log_level == "INFO"
    assert app_config.target_column_candidates == ("target", "label", "outcome")
    assert app_config.plot_format == "PNG"


def test_load_uppercases_log_level_and_plot_format(config_ini_writer, existing_python_path: str) -> None:
    config_file_path = config_ini_writer(
        python_path=existing_python_path, log_level="debug", plot_format="png"
    )

    app_config = ConfigLoader(config_file_path).load()

    assert app_config.log_level == "DEBUG"
    assert app_config.plot_format == "PNG"


def test_load_strips_whitespace_from_candidate_names(config_ini_writer, existing_python_path: str) -> None:
    config_file_path = config_ini_writer(
        python_path=existing_python_path, candidate_names=" target , label ,outcome "
    )

    app_config = ConfigLoader(config_file_path).load()

    assert app_config.target_column_candidates == ("target", "label", "outcome")


def test_load_resolves_absolute_output_root_unchanged(
    config_ini_writer, existing_python_path: str, tmp_path: Path
) -> None:
    absolute_output_root = tmp_path / "absolute_outputs"
    config_file_path = config_ini_writer(
        python_path=existing_python_path, output_root=str(absolute_output_root)
    )

    app_config = ConfigLoader(config_file_path).load()

    assert app_config.output_root == absolute_output_root


def test_missing_config_file_raises_file_not_found_error(tmp_path: Path) -> None:
    missing_config_file_path = tmp_path / "config" / "config.ini"

    with pytest.raises(FileNotFoundError):
        ConfigLoader(missing_config_file_path).load()


def test_missing_section_raises_config_validation_error(config_ini_writer) -> None:
    config_file_path = config_ini_writer(
        sections={
            "output": {"OUTPUT_ROOT": "../outputs"},
            "logging": {"LOG_LEVEL": "INFO"},
            "target_column": {"CANDIDATE_NAMES": "target"},
            "plot": {"PLOT_FORMAT": "PNG"},
        }
    )

    with pytest.raises(ConfigValidationError, match="python"):
        ConfigLoader(config_file_path).load()


def test_empty_required_value_raises_config_validation_error(
    config_ini_writer, existing_python_path: str
) -> None:
    config_file_path = config_ini_writer(
        sections={
            "python": {"PYTHON": existing_python_path},
            "output": {"OUTPUT_ROOT": "  "},
            "logging": {"LOG_LEVEL": "INFO"},
            "target_column": {"CANDIDATE_NAMES": "target"},
            "plot": {"PLOT_FORMAT": "PNG"},
        }
    )

    with pytest.raises(ConfigValidationError, match="OUTPUT_ROOT"):
        ConfigLoader(config_file_path).load()


def test_invalid_log_level_raises_config_validation_error(
    config_ini_writer, existing_python_path: str
) -> None:
    config_file_path = config_ini_writer(python_path=existing_python_path, log_level="TRACE")

    with pytest.raises(ConfigValidationError, match="LOG_LEVEL"):
        ConfigLoader(config_file_path).load()


def test_invalid_plot_format_raises_config_validation_error(
    config_ini_writer, existing_python_path: str
) -> None:
    config_file_path = config_ini_writer(python_path=existing_python_path, plot_format="JPEG")

    with pytest.raises(ConfigValidationError, match="PLOT_FORMAT"):
        ConfigLoader(config_file_path).load()


def test_nonexistent_python_path_raises_config_validation_error(
    config_ini_writer, tmp_path: Path
) -> None:
    nonexistent_python_path = str(tmp_path / "no_such_python.exe")
    config_file_path = config_ini_writer(python_path=nonexistent_python_path)

    with pytest.raises(ConfigValidationError, match="PYTHON"):
        ConfigLoader(config_file_path).load()


def test_empty_candidate_names_raises_config_validation_error(
    config_ini_writer, existing_python_path: str
) -> None:
    config_file_path = config_ini_writer(python_path=existing_python_path, candidate_names=" , , ")

    with pytest.raises(ConfigValidationError, match="CANDIDATE_NAMES"):
        ConfigLoader(config_file_path).load()


def test_default_config_file_path_points_to_project_config_directory() -> None:
    loader = ConfigLoader()

    assert loader.config_file_path.name == "config.ini"
    assert loader.config_file_path.parent.name == "config"
