from __future__ import annotations

import json
from pathlib import Path

import pytest

from epic_3.training.contracts import TrainingResult
from epic_3.training_orchestrator.contracts import TrainingJob, TrainingJobManifest
from epic_3.training_orchestrator.errors import MissingDataSplitError
from epic_3.training_orchestrator.orchestrator import TrainingOrchestrator

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_LIBRARY = REPO_ROOT / "model_library"
FIXTURES = REPO_ROOT / "epic_3" / "model_selection" / "fixtures"


class _StubWorker:
    def __init__(
        self,
        *,
        fail_ids: set[str] | None = None,
        raise_ids: set[str] | None = None,
    ) -> None:
        self.fail_ids = fail_ids or set()
        self.raise_ids = raise_ids or set()
        self.seen_statuses: list[str] = []

    def run(self, job: TrainingJob) -> TrainingResult:
        self.seen_statuses.append(job.status)
        if job.model_id in self.raise_ids:
            raise RuntimeError("worker crashed")
        if job.model_id in self.fail_ids:
            return TrainingResult(
                model_id=job.model_id,
                model_name=job.model_name,
                status="failed",
                metrics={},
                model_path=None,
                training_time_sec=0.1,
                error="expected failure",
            )

        model_path = Path(job.output_dir) / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"fake-model")
        return TrainingResult(
            model_id=job.model_id,
            model_name=job.model_name,
            status="completed",
            metrics={"validation_score": 0.9 + (job.priority / 100)},
            model_path=str(model_path),
            training_time_sec=0.2,
            error=None,
        )


def _manual_manifest(tmp_path: Path, count: int = 3) -> TrainingJobManifest:
    jobs = []
    for index in range(1, count + 1):
        model_id = f"model_{index:03d}"
        output_dir = tmp_path / "session" / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        jobs.append(
            TrainingJob(
                model_id=model_id,
                model_name=f"Model{index}",
                task_type="classification",
                data_format="tabular",
                trainer_type="tabular_classification",
                parameters={},
                train_path=str(tmp_path / "train.csv"),
                test_path=str(tmp_path / "test.csv"),
                output_dir=str(output_dir),
                priority=index,
            )
        )
    return TrainingJobManifest(
        session_id="session-local",
        problem_type="classification",
        data_format="tabular",
        total_jobs=len(jobs),
        jobs=jobs,
    )


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


def test_execute_local_updates_status_and_writes_summary(tmp_path: Path) -> None:
    manifest = _manual_manifest(tmp_path)
    manifest_path = tmp_path / "session" / "training_jobs.json"
    worker = _StubWorker()

    summary = TrainingOrchestrator(MODEL_LIBRARY).execute_local(
        manifest,
        worker=worker,
        manifest_path=manifest_path,
    )

    assert worker.seen_statuses == ["running", "running", "running"]
    assert summary.status == "completed"
    assert summary.completed == 3
    assert summary.failed == 0
    assert [job.status for job in manifest.jobs] == ["completed"] * 3
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [job["status"] for job in persisted["jobs"]] == ["completed"] * 3
    assert (tmp_path / "session" / "training_summary.json").is_file()


def test_execute_local_isolates_partial_failure(tmp_path: Path) -> None:
    manifest = _manual_manifest(tmp_path)
    worker = _StubWorker(fail_ids={"model_002"})

    summary = TrainingOrchestrator(MODEL_LIBRARY).execute_local(
        manifest,
        worker=worker,
    )

    assert summary.status == "partial_failure"
    assert summary.completed == 2
    assert summary.failed == 1
    assert [job.status for job in manifest.jobs] == [
        "completed",
        "failed",
        "completed",
    ]


def test_execute_local_converts_worker_exception_and_continues(tmp_path: Path) -> None:
    manifest = _manual_manifest(tmp_path)
    worker = _StubWorker(raise_ids={"model_001"})

    summary = TrainingOrchestrator(MODEL_LIBRARY).execute_local(
        manifest,
        worker=worker,
    )

    assert summary.completed == 2
    assert summary.failed == 1
    failed = summary.models[0]
    assert failed.status == "failed"
    assert "worker crashed" in (failed.error or "")
    assert (tmp_path / "session" / "model_001" / "train_metrics.json").is_file()


def test_execute_local_reports_all_failed_session(tmp_path: Path) -> None:
    manifest = _manual_manifest(tmp_path, count=2)
    worker = _StubWorker(fail_ids={"model_001", "model_002"})

    summary = TrainingOrchestrator(MODEL_LIBRARY).execute_local(
        manifest,
        worker=worker,
    )

    assert summary.status == "failed"
    assert summary.completed == 0
    assert summary.failed == 2
    assert [job.status for job in manifest.jobs] == ["failed", "failed"]
