from __future__ import annotations

import configparser
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PythonConfig:
    python_binary: str


@dataclass(frozen=True)
class PathConfig:
    workspace_root: Path
    session_log_dir: Path


@dataclass(frozen=True)
class UploadConfig:
    max_file_size_mb: int
    allowed_extensions: list[str]
    mini_data_sample_rows: int
    chunk_size_rows: int
    recent_upload_limit: int
    min_rows: int
    null_threshold: float
    pii_patterns: list[str]


@dataclass(frozen=True)
class PipelineConfig:
    train_test_split: float
    max_ml_models: int
    max_hpt_trials: int


@dataclass(frozen=True)
class LlmModelsConfig:
    openai_base_model: str
    anthropic_base_model: str
    gemini_base_model: str

    def as_provider_map(self) -> dict[str, str]:
        return {
            "openai": self.openai_base_model,
            "anthropic": self.anthropic_base_model,
            "gemini": self.gemini_base_model,
        }


@dataclass(frozen=True)
class LlmBaseUrlsConfig:
    openai_base_url: str | None
    anthropic_base_url: str | None
    gemini_base_url: str | None

    def as_provider_map(self) -> dict[str, str | None]:
        return {
            "openai": self.openai_base_url,
            "anthropic": self.anthropic_base_url,
            "gemini": self.gemini_base_url,
        }


@dataclass(frozen=True)
class MetadataAgentConfig:
    classification_unique_threshold: float
    categorical_unique_ratio: float
    llm_max_retries: int
    metadata_context_char_limit: int


@dataclass(frozen=True)
class TrainingApiConfig:
    model_library_root: Path
    default_execution_mode: str
    session_output_dir: str
    metadata_candidates: list[str]
    model_config_candidates: list[str]
    train_candidates: list[str]
    test_candidates: list[str]
    run_status_filename: str
    manifest_filename: str
    summary_filename: str
    max_concurrent_runs: int
    ray_timeout_sec: float


class ConfigLoader:
    required_sections = [
        "python",
        "paths",
        "upload",
        "pipeline",
        "llm_models",
        "metadata_agent",
    ]

    def __init__(
        self,
        config_path: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[1]
        self.config_path = config_path or self.repo_root / "config.ini"
        self.parser = configparser.ConfigParser()
        loaded_files = self.parser.read(self.config_path)
        if not loaded_files:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        self._validate_required_sections()

        self.python = PythonConfig(
            python_binary=self.parser.get("python", "PYTHON", fallback=""),
        )
        self.paths = PathConfig(
            workspace_root=self._resolve_repo_path(
                self.parser.get("paths", "WORKSPACE_ROOT")
            ),
            session_log_dir=self._resolve_repo_path(
                self.parser.get("paths", "SESSION_LOG_DIR")
            ),
        )
        self.upload = UploadConfig(
            max_file_size_mb=self.parser.getint("upload", "MAX_FILE_SIZE_MB"),
            allowed_extensions=self._parse_csv_list(
                self.parser.get("upload", "ALLOWED_EXTENSIONS")
            ),
            mini_data_sample_rows=self.parser.getint(
                "upload", "MINI_DATA_SAMPLE_ROWS"
            ),
            chunk_size_rows=self.parser.getint("upload", "CHUNK_SIZE_ROWS"),
            recent_upload_limit=self.parser.getint("upload", "RECENT_UPLOAD_LIMIT"),
            min_rows=self.parser.getint("upload", "MIN_ROWS"),
            null_threshold=self.parser.getfloat("upload", "NULL_THRESHOLD"),
            pii_patterns=self._parse_json_string_list(
                self.parser.get("upload", "PII_PATTERNS")
            ),
        )
        self.pipeline = PipelineConfig(
            train_test_split=self.parser.getfloat("pipeline", "TRAIN_TEST_SPLIT"),
            max_ml_models=self.parser.getint("pipeline", "MAX_ML_MODELS"),
            max_hpt_trials=self.parser.getint("pipeline", "MAX_HPT_TRIALS"),
        )
        self.llm_models = LlmModelsConfig(
            openai_base_model=self.parser.get("llm_models", "OPENAI_BASE_MODEL"),
            anthropic_base_model=self.parser.get(
                "llm_models", "ANTHROPIC_BASE_MODEL"
            ),
            gemini_base_model=self.parser.get("llm_models", "GEMINI_BASE_MODEL"),
        )
        # Optional section: blank/missing base URLs leave litellm to use its
        # provider defaults, so old config files keep working unchanged.
        self.llm_base_urls = LlmBaseUrlsConfig(
            openai_base_url=self._optional_base_url("OPENAI_BASE_URL"),
            anthropic_base_url=self._optional_base_url("ANTHROPIC_BASE_URL"),
            gemini_base_url=self._optional_base_url("GEMINI_BASE_URL"),
        )
        self.metadata_agent = MetadataAgentConfig(
            classification_unique_threshold=self.parser.getfloat(
                "metadata_agent", "CLASSIFICATION_UNIQUE_THRESHOLD"
            ),
            categorical_unique_ratio=self.parser.getfloat(
                "metadata_agent", "CATEGORICAL_UNIQUE_RATIO"
            ),
            llm_max_retries=self.parser.getint("metadata_agent", "LLM_MAX_RETRIES"),
            metadata_context_char_limit=self.parser.getint(
                "metadata_agent",
                "METADATA_CONTEXT_CHAR_LIMIT",
            ),
        )
        self.training_api = TrainingApiConfig(
            model_library_root=self._resolve_repo_path(
                self.parser.get(
                    "training_api",
                    "MODEL_LIBRARY_ROOT",
                    fallback="model_library",
                )
            ),
            default_execution_mode=self.parser.get(
                "training_api",
                "DEFAULT_EXECUTION_MODE",
                fallback="ray",
            ).strip().lower(),
            session_output_dir=self.parser.get(
                "training_api",
                "SESSION_OUTPUT_DIR",
                fallback="training",
            ).strip(),
            metadata_candidates=self._parse_csv_list(
                self.parser.get(
                    "training_api",
                    "METADATA_CANDIDATES",
                    fallback="reports/metadata.json,metadata.json",
                )
            ),
            model_config_candidates=self._parse_csv_list(
                self.parser.get(
                    "training_api",
                    "MODEL_CONFIG_CANDIDATES",
                    fallback="model_config.json,reports/model_config.json",
                )
            ),
            train_candidates=self._parse_csv_list(
                self.parser.get(
                    "training_api",
                    "TRAIN_CANDIDATES",
                    fallback="data/train.csv,train.csv",
                )
            ),
            test_candidates=self._parse_csv_list(
                self.parser.get(
                    "training_api",
                    "TEST_CANDIDATES",
                    fallback="data/test.csv,test.csv",
                )
            ),
            run_status_filename=self.parser.get(
                "training_api",
                "RUN_STATUS_FILENAME",
                fallback="training_run.json",
            ).strip(),
            manifest_filename=self.parser.get(
                "training_api",
                "MANIFEST_FILENAME",
                fallback="training_jobs.json",
            ).strip(),
            summary_filename=self.parser.get(
                "training_api",
                "SUMMARY_FILENAME",
                fallback="training_summary.json",
            ).strip(),
            max_concurrent_runs=self.parser.getint(
                "training_api",
                "MAX_CONCURRENT_RUNS",
                fallback=2,
            ),
            ray_timeout_sec=self.parser.getfloat(
                "training_api",
                "RAY_TIMEOUT_SEC",
                fallback=300.0,
            ),
        )
        if self.training_api.default_execution_mode not in {"ray", "local"}:
            raise ValueError(
                "training_api.DEFAULT_EXECUTION_MODE must be 'ray' or 'local'"
            )

    def base_model_for_provider(self, provider: str) -> str:
        provider_models = self.llm_models.as_provider_map()
        normalized_provider = provider.strip().lower()
        if normalized_provider not in provider_models:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        return provider_models[normalized_provider]

    def base_url_for_provider(self, provider: str) -> str | None:
        provider_base_urls = self.llm_base_urls.as_provider_map()
        normalized_provider = provider.strip().lower()
        if normalized_provider not in provider_base_urls:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        return provider_base_urls[normalized_provider]

    def _optional_base_url(self, option_name: str) -> str | None:
        raw_value = self.parser.get(
            "llm_base_urls",
            option_name,
            fallback="",
        )
        stripped_value = raw_value.strip()
        return stripped_value or None

    def _validate_required_sections(self) -> None:
        missing_sections = [
            section_name
            for section_name in self.required_sections
            if not self.parser.has_section(section_name)
        ]
        if missing_sections:
            raise ValueError(f"Missing config section(s): {missing_sections}")

    def _resolve_repo_path(self, raw_path: str) -> Path:
        candidate_path = Path(raw_path).expanduser()
        if candidate_path.is_absolute():
            return candidate_path
        return self.repo_root / candidate_path

    @staticmethod
    def _parse_csv_list(raw_value: str) -> list[str]:
        return [
            item.strip()
            for item in raw_value.split(",")
            if item.strip()
        ]

    @staticmethod
    def _parse_json_string_list(raw_value: str) -> list[str]:
        parsed_value: Any = json.loads(raw_value)
        if not isinstance(parsed_value, list):
            raise ValueError("Expected JSON array")

        string_values = [
            item
            for item in parsed_value
            if isinstance(item, str)
        ]
        if len(string_values) != len(parsed_value):
            raise ValueError("Expected JSON array of strings")
        return string_values
