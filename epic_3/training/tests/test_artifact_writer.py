from __future__ import annotations

import json
from pathlib import Path

from epic_3.training.artifact_writer import write_training_result
from epic_3.training.contracts import TrainingResult


def test_writes_training_result_atomically(tmp_path: Path) -> None:
    result = TrainingResult(
        model_id="model_001",
        model_name="RandomForestClassifier",
        status="completed",
        metrics={"validation_score": 0.9},
        model_path=str(tmp_path / "model.pkl"),
        training_time_sec=1.2,
        error=None,
    )

    path = write_training_result(result, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "train_metrics.json"
    assert payload["status"] == "completed"
    assert not list(tmp_path.glob("*.tmp"))
