from pathlib import Path

import pytest

from backend.config_loader import ConfigLoader


def write_valid_config(config_path: Path) -> None:
    config_path.write_text(
        "[python]\nPYTHON=\n"
        "[paths]\nWORKSPACE_ROOT=.mitra\nSESSION_LOG_DIR=.mitra/logs\n"
        "[upload]\nMAX_FILE_SIZE_MB=200\nALLOWED_EXTENSIONS=.csv,.xlsx\n"
        "MINI_DATA_SAMPLE_ROWS=1000\nCHUNK_SIZE_ROWS=50000\nRECENT_UPLOAD_LIMIT=5\n"
        "MIN_ROWS=10\nNULL_THRESHOLD=0.8\nPII_PATTERNS=[\"(?i)email\"]\n"
        "[pipeline]\nTRAIN_TEST_SPLIT=0.8\nMAX_ML_MODELS=10\nMAX_HPT_TRIALS=5\n"
        "[llm_models]\nOPENAI_BASE_MODEL=openai/gpt-5.1\n"
        "ANTHROPIC_BASE_MODEL=anthropic/claude-sonnet-4-5-20250929\n"
        "GEMINI_BASE_MODEL=gemini/gemini-3-pro\n"
        "[metadata_agent]\nCLASSIFICATION_UNIQUE_THRESHOLD=0.05\n"
        "CATEGORICAL_UNIQUE_RATIO=0.05\nLLM_MAX_RETRIES=3\n"
        "METADATA_CONTEXT_CHAR_LIMIT=20000\n",
        encoding="utf-8",
    )


def test_config_loader_reads_upload_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.ini"
    write_valid_config(config_path=config_path)

    loader = ConfigLoader(config_path=config_path)

    assert loader.upload.max_file_size_mb == 200
    assert loader.upload.allowed_extensions == [".csv", ".xlsx"]
    assert loader.upload.pii_patterns == ["(?i)email"]
    assert loader.metadata_agent.metadata_context_char_limit == 20000
    assert loader.base_model_for_provider("openai") == "openai/gpt-5.1"


def test_config_loader_resolves_workspace_relative_to_repo_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.ini"
    write_valid_config(config_path=config_path)

    loader = ConfigLoader(config_path=config_path, repo_root=tmp_path)

    assert loader.paths.workspace_root == tmp_path / ".mitra"
    assert loader.paths.session_log_dir == tmp_path / ".mitra" / "logs"


def test_config_loader_rejects_unknown_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.ini"
    write_valid_config(config_path=config_path)
    loader = ConfigLoader(config_path=config_path)

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        loader.base_model_for_provider("cohere")
