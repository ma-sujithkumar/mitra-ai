from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from backend.agents.training_orchestrator.orchestrator import TrainingOrchestrator

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_LIBRARY = REPO_ROOT / "model_library"


def _write_split(path: Path, X, y, feature_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*feature_names, "species"])
        for features, target in zip(X, y, strict=True):
            writer.writerow([*features, target])


def test_prepare_train_and_aggregate_one_iris_model(tmp_path: Path) -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_split(train_path, X_train, y_train, list(iris.feature_names))
    _write_split(test_path, X_test, y_test, list(iris.feature_names))

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "problem_type": "classification",
                "data_format": "tabular",
                "output_cols": ["species"],
            }
        ),
        encoding="utf-8",
    )
    model_config_path = tmp_path / "model_config.json"
    model_config_path.write_text(
        json.dumps(
            [
                {
                    "name": "RandomForestClassifier",
                    "model_name": "RandomForestClassifier",
                    "task_type": "classification",
                    "priority": 1,
                    "rationale": "Iris end-to-end smoke test",
                }
            ]
        ),
        encoding="utf-8",
    )

    session_dir = tmp_path / ".mitra" / "iris-session"
    summary = TrainingOrchestrator(MODEL_LIBRARY).prepare_and_execute_local(
        session_id="iris-session",
        metadata_path=metadata_path,
        model_config_path=model_config_path,
        train_path=train_path,
        test_path=test_path,
        session_dir=session_dir,
        target_column="species",
    )

    assert summary.status == "completed"
    assert summary.completed == 1
    assert summary.failed == 0
    assert summary.models[0].validation_score is not None
    assert summary.models[0].validation_score >= 0.80
    assert Path(summary.models[0].model_path or "").is_file()
    assert (session_dir / "model_001" / "train_metrics.json").is_file()
    assert (session_dir / "training_summary.json").is_file()
    manifest = json.loads(
        (session_dir / "training_jobs.json").read_text(encoding="utf-8")
    )
    assert manifest["jobs"][0]["status"] == "completed"
