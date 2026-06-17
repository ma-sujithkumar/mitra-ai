from __future__ import annotations

from pathlib import Path

import pytest

from epic_3.training.contracts import TrainingResult
from epic_3.training_orchestrator.contracts import TrainingJob, TrainingJobManifest
from epic_3.training_orchestrator.errors import ResultAggregationError
from epic_3.training_orchestrator.result_aggregator import TrainingResultAggregator


def _job(model_id: str, model_name: str, tmp_path: Path, priority: int) -> TrainingJob:
    return TrainingJob(
        model_id=model_id,
        model_name=model_name,
        task_type="classification",
        data_format="tabular",
        trainer_type="tabular_classification",
        parameters={},
        train_path=str(tmp_path / "train.csv"),
        test_path=str(tmp_path / "test.csv"),
        output_dir=str(tmp_path / model_id),
        priority=priority,
    )


def _manifest(tmp_path: Path) -> TrainingJobManifest:
    return TrainingJobManifest(
        session_id="session-1",
        problem_type="classification",
        data_format="tabular",
        total_jobs=2,
        jobs=[
            _job("model_001", "RandomForestClassifier", tmp_path, 1),
            _job("model_002", "LogisticRegression", tmp_path, 2),
        ],
    )


def _completed(model_id: str, model_name: str, tmp_path: Path) -> TrainingResult:
    model_path = tmp_path / model_id / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    return TrainingResult(
        model_id=model_id,
        model_name=model_name,
        status="completed",
        metrics={"validation_score": 0.91},
        model_path=str(model_path),
        training_time_sec=1.0,
        error=None,
    )


def test_aggregates_in_manifest_order_and_writes_summary(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    results = [
        TrainingResult(
            model_id="model_002",
            model_name="LogisticRegression",
            status="failed",
            metrics={},
            model_path=None,
            training_time_sec=0.2,
            error="training failed",
        ),
        _completed("model_001", "RandomForestClassifier", tmp_path),
    ]

    aggregator = TrainingResultAggregator()
    summary = aggregator.build(manifest=manifest, results=results)
    path = aggregator.write(summary, tmp_path / "training_summary.json")

    assert summary.status == "partial_failure"
    assert summary.completed == 1
    assert summary.failed == 1
    assert [item.model_id for item in summary.models] == ["model_001", "model_002"]
    assert summary.models[0].validation_score == pytest.approx(0.91)
    assert path.is_file()


def test_rejects_missing_result(tmp_path: Path) -> None:
    with pytest.raises(ResultAggregationError, match="missing model_id"):
        TrainingResultAggregator().build(
            manifest=_manifest(tmp_path),
            results=[_completed("model_001", "RandomForestClassifier", tmp_path)],
        )


def test_rejects_duplicate_result_ids(tmp_path: Path) -> None:
    first = _completed("model_001", "RandomForestClassifier", tmp_path)
    with pytest.raises(ResultAggregationError, match="duplicate model_id"):
        TrainingResultAggregator().build(
            manifest=_manifest(tmp_path),
            results=[first, first],
        )
