"""Loads and validates the single config.ini file for the Epic 4 SHAP module.

Per project convention, this module is the only place config.ini is parsed; every
other module receives an already-resolved AppConfig instance rather than reading the
file itself (architecture.md Section 2). Configurable parameters CFG-01..04 from
spec.md Section 28 are: output root directory, logging level, target column
candidate names, and plot output format.
"""

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ConfigValidationError(Exception):
    """Raised when config.ini is missing required keys or contains invalid values."""


_SUPPORTED_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
_SUPPORTED_PLOT_FORMATS: tuple[str, ...] = ("PNG",)
_DEFAULT_CONFIG_FILENAME: str = "config.ini"


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration resolved from config.ini.

    Attributes:
        python_path: Absolute path to the Python interpreter designated for this
            project (project-wide convention: a single CREATE PYTHON entry).
        output_root: Absolute path to the root directory under which every session
            folder is created (CFG-01).
        log_level: One of DEBUG, INFO, WARNING, ERROR (CFG-02).
        target_column_candidates: Ordered, case-sensitive candidate column names used
            to identify the optional target column to exclude from SHAP processing
            (CFG-03).
        plot_format: Output image format for generated plots (CFG-04).
    """

    python_path: Path
    output_root: Path
    log_level: str
    target_column_candidates: tuple[str, ...]
    plot_format: str
    max_display_features: int = 20


class ConfigLoader:
    """Reads epic_4_shap/config/config.ini and exposes a validated AppConfig."""

    def __init__(self, config_file_path: Optional[Path] = None) -> None:
        """Initializes the loader with the config.ini path to read.

        Args:
            config_file_path: Absolute path to config.ini. When None, resolves to
                the single project config.ini at <epic_4_shap>/config/config.ini.
        """
        self.config_file_path: Path = config_file_path or self._default_config_file_path()

    def load(self) -> AppConfig:
        """Parses config.ini and returns an immutable, validated AppConfig.

        Returns:
            The validated AppConfig built from config.ini.

        Raises:
            FileNotFoundError: If config.ini does not exist at the resolved path.
            ConfigValidationError: If a required key is missing, empty, or invalid.
        """
        if not self.config_file_path.is_file():
            raise FileNotFoundError(
                f"config.ini not found at {self.config_file_path}. Create it from "
                "the documented [python]/[output]/[logging]/[target_column]/[plot] "
                "template before running the pipeline."
            )

        config_parser = configparser.ConfigParser()
        config_parser.read(self.config_file_path)

        python_path_value = self._read_required(config_parser, "python", "PYTHON")
        output_root_value = self._read_required(config_parser, "output", "OUTPUT_ROOT")
        log_level_value = self._read_required(config_parser, "logging", "LOG_LEVEL").upper()
        target_column_value = self._read_required(
            config_parser, "target_column", "CANDIDATE_NAMES"
        )
        plot_format_value = self._read_required(config_parser, "plot", "PLOT_FORMAT").upper()

        self._validate_allowed_value(
            field_name="LOG_LEVEL",
            value=log_level_value,
            allowed_values=_SUPPORTED_LOG_LEVELS,
        )
        self._validate_allowed_value(
            field_name="PLOT_FORMAT",
            value=plot_format_value,
            allowed_values=_SUPPORTED_PLOT_FORMATS,
        )

        python_path = Path(python_path_value)
        if not python_path.is_file():
            raise ConfigValidationError(
                f"PYTHON path '{python_path}' configured in {self.config_file_path} "
                "does not point to an existing file."
            )

        output_root = self._resolve_relative_to_config_directory(Path(output_root_value))

        target_column_candidates = tuple(
            candidate_name.strip()
            for candidate_name in target_column_value.split(",")
            if candidate_name.strip()
        )
        if not target_column_candidates:
            raise ConfigValidationError(
                f"CANDIDATE_NAMES in {self.config_file_path} must contain at least "
                "one non-empty, comma-separated column name."
            )

        max_display_features_raw = config_parser.get(
            "plot", "MAX_DISPLAY_FEATURES", fallback="20"
        ).strip()
        try:
            max_display_features_value = int(max_display_features_raw)
        except ValueError:
            raise ConfigValidationError(
                f"MAX_DISPLAY_FEATURES '{max_display_features_raw}' in "
                f"{self.config_file_path} must be a positive integer."
            )
        if max_display_features_value < 1:
            raise ConfigValidationError(
                f"MAX_DISPLAY_FEATURES must be a positive integer, got "
                f"{max_display_features_value} in {self.config_file_path}."
            )

        return AppConfig(
            python_path=python_path,
            output_root=output_root,
            log_level=log_level_value,
            target_column_candidates=target_column_candidates,
            plot_format=plot_format_value,
            max_display_features=max_display_features_value,
        )

    def _default_config_file_path(self) -> Path:
        """Resolves the single project config.ini relative to this module's location."""
        module_directory = Path(__file__).resolve().parent
        project_root = module_directory  # package dir now contains config/
        return project_root / "config" / _DEFAULT_CONFIG_FILENAME

    def _resolve_relative_to_config_directory(self, candidate_path: Path) -> Path:
        """Resolves a config.ini path value, treating relative paths as relative to
        the directory that contains config.ini itself (e.g. OUTPUT_ROOT = ../outputs
        resolves to a sibling of the config/ directory)."""
        if candidate_path.is_absolute():
            return candidate_path
        config_directory = self.config_file_path.resolve().parent
        return (config_directory / candidate_path).resolve()

    @staticmethod
    def _read_required(
        config_parser: configparser.ConfigParser, section_name: str, key_name: str
    ) -> str:
        """Reads one required, non-empty key from config.ini.

        Raises:
            ConfigValidationError: If the section/key is missing or the value is
                empty after stripping whitespace.
        """
        if not config_parser.has_section(section_name) or not config_parser.has_option(
            section_name, key_name
        ):
            raise ConfigValidationError(
                f"Missing required '[{section_name}] {key_name}' entry in config.ini."
            )
        raw_value = config_parser.get(section_name, key_name).strip()
        if not raw_value:
            raise ConfigValidationError(
                f"'[{section_name}] {key_name}' entry in config.ini must not be empty."
            )
        return raw_value

    def _validate_allowed_value(
        self, field_name: str, value: str, allowed_values: tuple[str, ...]
    ) -> None:
        """Raises ConfigValidationError if value is not one of allowed_values."""
        if value not in allowed_values:
            raise ConfigValidationError(
                f"{field_name} '{value}' in {self.config_file_path} is not one of "
                f"{allowed_values}."
            )
