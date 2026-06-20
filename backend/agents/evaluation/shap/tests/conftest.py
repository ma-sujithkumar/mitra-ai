"""Shared pytest fixtures for the Epic 4 SHAP explainability test suite."""

import sys
from pathlib import Path
from typing import Optional

import pytest


@pytest.fixture
def existing_python_path() -> str:
    """Returns a real, existing interpreter path for config.ini PYTHON validation.

    Uses the interpreter currently running the test suite instead of a hardcoded
    path, so tests do not depend on any specific machine's installed Python
    locations.
    """
    return sys.executable


@pytest.fixture
def config_ini_writer(tmp_path: Path):
    """Returns a factory that writes a config.ini under <tmp_path>/config/config.ini.

    Mirrors the real project layout (epic_4_shap/config/config.ini) so relative
    path resolution inside ConfigLoader behaves the same as in production.
    """

    def _write(
        *,
        python_path: Optional[str] = None,
        output_root: str = "../outputs",
        log_level: str = "INFO",
        candidate_names: str = "target,label,outcome",
        plot_format: str = "PNG",
        sections: Optional[dict[str, dict[str, str]]] = None,
    ) -> Path:
        config_directory = tmp_path / "config"
        config_directory.mkdir(parents=True, exist_ok=True)
        config_file_path = config_directory / "config.ini"

        if sections is not None:
            content_lines: list[str] = []
            for section_name, options in sections.items():
                content_lines.append(f"[{section_name}]")
                for key, value in options.items():
                    content_lines.append(f"{key} = {value}")
                content_lines.append("")
            config_text = "\n".join(content_lines)
        else:
            resolved_python_path = python_path if python_path is not None else sys.executable
            config_text = (
                "[python]\n"
                f"PYTHON = {resolved_python_path}\n"
                "\n"
                "[output]\n"
                f"OUTPUT_ROOT = {output_root}\n"
                "\n"
                "[logging]\n"
                f"LOG_LEVEL = {log_level}\n"
                "\n"
                "[target_column]\n"
                f"CANDIDATE_NAMES = {candidate_names}\n"
                "\n"
                "[plot]\n"
                f"PLOT_FORMAT = {plot_format}\n"
                "MAX_DISPLAY_FEATURES = 20\n"
            )

        config_file_path.write_text(config_text, encoding="utf-8")
        return config_file_path

    return _write
