from __future__ import annotations

from pathlib import Path

import pytest

from backend.config_loader import ConfigLoader

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def e2e_config_loader(tmp_path: Path) -> ConfigLoader:
    workspace_root = tmp_path / ".mitra"
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "[python]\nPYTHON=\n"
        f"[paths]\nWORKSPACE_ROOT={workspace_root}\n"
        f"SESSION_LOG_DIR={workspace_root / 'logs'}\n"
        "[upload]\nMAX_FILE_SIZE_MB=200\n"
        "ALLOWED_EXTENSIONS=.csv,.xls,.xlsx\n"
        "MINI_DATA_SAMPLE_ROWS=1000\nCHUNK_SIZE_ROWS=50000\n"
        "RECENT_UPLOAD_LIMIT=5\nMIN_ROWS=10\nNULL_THRESHOLD=0.8\n"
        "PII_PATTERNS=[\"(?i)email\"]\nMETADATA_MATCH_MIN_OVERLAP=0.5\n"
        "[pipeline]\nTRAIN_TEST_SPLIT=0.8\nMAX_ML_MODELS=10\n"
        "MAX_HPT_TRIALS=5\n"
        "[llm_models]\nOPENAI_BASE_MODEL=openai/gpt-5.1\n"
        "ANTHROPIC_BASE_MODEL=anthropic/claude-sonnet-4-5-20250929\n"
        "GEMINI_BASE_MODEL=gemini/gemini-3-pro\n"
        "[metadata_agent]\nCLASSIFICATION_UNIQUE_THRESHOLD=0.05\n"
        "CATEGORICAL_UNIQUE_RATIO=0.05\nLLM_MAX_RETRIES=3\n"
        "METADATA_CONTEXT_CHAR_LIMIT=20000\n"
        "[training_api]\n"
        f"MODEL_LIBRARY_ROOT={REPO_ROOT / 'model_library'}\n"
        "DEFAULT_EXECUTION_MODE=local\nSESSION_OUTPUT_DIR=training\n"
        "METADATA_CANDIDATES=reports/metadata.json,metadata.json\n"
        "MODEL_CONFIG_CANDIDATES=model_config.json,reports/model_config.json\n"
        "TRAIN_CANDIDATES=data/train.csv,train.csv\n"
        "TEST_CANDIDATES=data/test.csv,test.csv\n"
        "RUN_STATUS_FILENAME=training_run.json\n"
        "MANIFEST_FILENAME=training_jobs.json\n"
        "SUMMARY_FILENAME=training_summary.json\n"
        "MAX_CONCURRENT_RUNS=2\nRAY_TIMEOUT_SEC=5\n",
        encoding="utf-8",
    )
    return ConfigLoader(config_path=config_path, repo_root=REPO_ROOT)
