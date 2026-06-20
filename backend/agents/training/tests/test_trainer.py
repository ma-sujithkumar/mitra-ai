from __future__ import annotations

import json
from pathlib import Path

from backend.agents.training.trainer import LocalTrainingWorker
from backend.agents.training_orchestrator.contracts import TrainingJob


def _classification_job(train: Path, test: Path, output: Path) -> TrainingJob:
    return TrainingJob(
        model_id="model_001",
        model_name="RandomForestClassifier",
        task_type="classification",
        data_format="tabular",
        trainer_type="tabular_classification",
        parameters={
            "n_estimators": 30,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": 1,
        },
        train_path=str(train),
        test_path=str(test),
        output_dir=str(output),
        priority=1,
        rationale="Iris smoke test",
    )


def test_trains_iris_model_and_writes_artifacts(
    model_library_root: Path,
    iris_csv_splits: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    train_path, test_path = iris_csv_splits
    output_dir = tmp_path / "model_001"
    worker = LocalTrainingWorker(model_library_root)

    result = worker.run(_classification_job(train_path, test_path, output_dir))

    assert result.status == "completed", result.error
    assert result.error is None
    assert result.metrics["validation_score"] >= 0.80
    assert Path(result.model_path or "").is_file()
    metrics_path = output_dir / "train_metrics.json"
    assert metrics_path.is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["status"] == "completed"


def test_trains_regression_model(
    model_library_root: Path,
    regression_csv_splits: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    train_path, test_path = regression_csv_splits
    job = TrainingJob(
        model_id="model_002",
        model_name="LinearRegression",
        task_type="regression",
        data_format="tabular",
        trainer_type="tabular_regression",
        parameters={"fit_intercept": True, "n_jobs": 1},
        train_path=str(train_path),
        test_path=str(test_path),
        output_dir=str(tmp_path / "model_002"),
        priority=1,
    )

    result = LocalTrainingWorker(model_library_root).run(job)

    assert result.status == "completed", result.error
    assert result.metrics["primary_metric"] == "r2"
    assert "rmse" in result.metrics["validation"]


def test_returns_structured_failure_and_metrics_file(
    model_library_root: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "failed"
    job = _classification_job(
        tmp_path / "missing_train.csv",
        tmp_path / "missing_test.csv",
        output_dir,
    )

    result = LocalTrainingWorker(model_library_root).run(job)

    assert result.status == "failed"
    assert result.model_path is None
    assert "does not exist" in (result.error or "")
    payload = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
