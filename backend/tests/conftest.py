from pathlib import Path

import pytest

from backend.config_loader import ConfigLoader


@pytest.fixture
def test_config_loader(tmp_path: Path) -> ConfigLoader:
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "[python]\nPYTHON=\n"
        f"[paths]\nWORKSPACE_ROOT={tmp_path / '.mitra'}\n"
        f"SESSION_LOG_DIR={tmp_path / '.mitra' / 'logs'}\n"
        "[upload]\nMAX_FILE_SIZE_MB=200\nALLOWED_EXTENSIONS=.csv,.xls,.xlsx\n"
        "MINI_DATA_SAMPLE_ROWS=1000\nCHUNK_SIZE_ROWS=50000\nRECENT_UPLOAD_LIMIT=5\n"
        "MIN_ROWS=10\nNULL_THRESHOLD=0.8\nPII_PATTERNS=[\"(?i)email\"]\n"
        "METADATA_MATCH_MIN_OVERLAP=0.5\n"
        "[pipeline]\nTRAIN_TEST_SPLIT=0.8\nMAX_ML_MODELS=10\nMAX_HPT_TRIALS=5\n"
        "[llm_models]\nOPENAI_BASE_MODEL=openai/gpt-5.1\n"
        "ANTHROPIC_BASE_MODEL=anthropic/claude-sonnet-4-5-20250929\n"
        "GEMINI_BASE_MODEL=gemini/gemini-3-pro\n"
        "[llm_base_urls]\nOPENAI_BASE_URL=https://api.openai.com/v1\n"
        "ANTHROPIC_BASE_URL=https://api.anthropic.com\n"
        "GEMINI_BASE_URL=https://generativelanguage.googleapis.com\n"
        "[metadata_agent]\nCLASSIFICATION_UNIQUE_THRESHOLD=0.05\n"
        "CATEGORICAL_UNIQUE_RATIO=0.05\nLLM_MAX_RETRIES=3\n"
        "METADATA_CONTEXT_CHAR_LIMIT=20000\n"
        "[authdb]\n"
        f"USER_WORKSPACE_ROOT={tmp_path / 'mitra'}\n"
        f"FALLBACK_DB_URL=sqlite:///{tmp_path / 'auth.db'}\n",
        encoding="utf-8",
    )
    return ConfigLoader(config_path=config_path, repo_root=tmp_path)
