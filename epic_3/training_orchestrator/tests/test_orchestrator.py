from __future__ import annotations

import json
from pathlib import Path

import pytest

from epic_3.training_orchestrator.errors import MissingDataSplitError
from epic_3.training_orchestrator.orchestrator import TrainingOrchestrator

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_LIBRARY = REPO_ROOT / "model_library"
FIXTURES = REPO_ROOT / "epic_3" / "model_selection" / "fixtures"


def test_prepare_writes_manifest_and_model_directories(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("x,y\n1,a\n", encoding="utf-8")
    test.write_text("x,y\n2,b\n", encoding="utf-8")
    session = tmp_path / ".mitra" / "session-123"

    manifest = TrainingOrchestrator(MODEL_LIBRARY).prepare(
        session_id="session-123",
        metadata_path=FIXTURES / "iris_metadata.json",
        model_config_path=FIXTURES / "expected_iris_model_config.json",
        train_path=train,
        test_path=test,
        session_dir=session,
    )

    output = session / "training_jobs.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert manifest.total_jobs == 5
    assert payload["session_id"] == "session-123"
    assert [job["status"] for job in payload["jobs"]] == ["queued"] * 5
    assert (session / "model_001").is_dir()
    assert (session / "model_005").is_dir()


def test_prepare_rejects_missing_epic2_split(tmp_path: Path) -> None:
    with pytest.raises(MissingDataSplitError, match="train split"):
        TrainingOrchestrator(MODEL_LIBRARY).prepare(
            session_id="session-123",
            metadata_path=FIXTURES / "iris_metadata.json",
            model_config_path=FIXTURES / "expected_iris_model_config.json",
            train_path=tmp_path / "missing-train.csv",
            test_path=tmp_path / "missing-test.csv",
            session_dir=tmp_path / "session",
        )
