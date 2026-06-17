from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from epic_3.training.contracts import TrainingResult
from epic_3.training_orchestrator.contracts import TrainingJob, TrainingJobManifest
from epic_3.training_orchestrator.orchestrator import TrainingOrchestrator

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_LIBRARY = REPO_ROOT / "model_library"


def _manifest(tmp_path: Path, count: int = 3) -> TrainingJobManifest:
    jobs: list[TrainingJob] = []
    session_dir = tmp_path / "session"
    for index in range(1, count + 1):
        model_id = f"model_{index:03d}"
        output_dir = session_dir / model_id
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
        session_id="session-ray",
        problem_type="classification",
        data_format="tabular",
        total_jobs=count,
        jobs=jobs,
    )


def _completed(job: TrainingJob, score: float) -> TrainingResult:
    model_path = Path(job.output_dir) / "model.pkl"
    model_path.write_bytes(b"model")
    return TrainingResult(
        model_id=job.model_id,
        model_name=job.model_name,
        status="completed",
        metrics={"validation_score": score},
        model_path=str(model_path),
        training_time_sec=0.25,
        error=None,
    )


def _failed(job: TrainingJob, error: str) -> TrainingResult:
    return TrainingResult(
        model_id=job.model_id,
        model_name=job.model_name,
        status="failed",
        metrics={},
        model_path=None,
        training_time_sec=0.0,
        error=error,
    )


class _FakeRayExecutor:
    def __init__(
        self,
        results: list[TrainingResult] | None = None,
        *,
        start_error: Exception | None = None,
        collect_error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.start_error = start_error
        self.collect_error = collect_error
        self.started = False
        self.closed = False
        self.submitted_jobs: list[TrainingJob] = []
        self.submitted_statuses: list[str] = []
        self.resource_overrides: Any = None
        self.timeout_sec: float | None = None

    def start(self) -> dict[str, bool]:
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        return {"ready": True}

    def submit_all(
        self,
        jobs: list[TrainingJob],
        *,
        resource_overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        self.submitted_jobs = list(jobs)
        self.submitted_statuses = [job.status for job in jobs]
        self.resource_overrides = resource_overrides
        return [job.model_id for job in jobs]

    def collect(
        self,
        handles: list[str],
        *,
        timeout_sec: float | None = None,
    ) -> list[TrainingResult]:
        assert handles == [job.model_id for job in self.submitted_jobs]
        self.timeout_sec = timeout_sec
        if self.collect_error is not None:
            raise self.collect_error
        return list(self.results)

    def close(self) -> None:
        self.closed = True


def test_execute_ray_maps_completion_order_back_to_manifest_order(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    executor = _FakeRayExecutor(
        [
            _completed(manifest.jobs[2], 0.93),
            _completed(manifest.jobs[0], 0.91),
            _completed(manifest.jobs[1], 0.92),
        ]
    )
    manifest_path = tmp_path / "session" / "training_jobs.json"

    summary = TrainingOrchestrator(MODEL_LIBRARY).execute_ray(
        manifest,
        executor=executor,
        manifest_path=manifest_path,
        timeout_sec=17.0,
        resource_overrides={"model_001": {"num_cpus": 2}},
    )

    assert executor.started is True
    assert executor.closed is True
    assert executor.timeout_sec == 17.0
    assert executor.resource_overrides == {"model_001": {"num_cpus": 2}}
    assert executor.submitted_statuses == ["running"] * 3
    assert [item.model_id for item in summary.models] == [
        "model_001",
        "model_002",
        "model_003",
    ]
    assert [item.validation_score for item in summary.models] == [0.91, 0.92, 0.93]
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [job["status"] for job in persisted["jobs"]] == ["completed"] * 3
    assert (tmp_path / "session" / "training_summary.json").is_file()


def test_execute_ray_isolates_worker_failure(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    executor = _FakeRayExecutor(
        [
            _completed(manifest.jobs[0], 0.91),
            _failed(manifest.jobs[1], "Ray task failed"),
            _completed(manifest.jobs[2], 0.93),
        ]
    )

    summary = TrainingOrchestrator(MODEL_LIBRARY).execute_ray(
        manifest,
        executor=executor,
    )

    assert summary.status == "partial_failure"
    assert summary.completed == 2
    assert summary.failed == 1
    assert [job.status for job in manifest.jobs] == [
        "completed",
        "failed",
        "completed",
    ]


def test_execute_ray_marks_missing_timeout_result_failed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    executor = _FakeRayExecutor(
        [
            _completed(manifest.jobs[0], 0.91),
            _failed(manifest.jobs[1], "Ray task timed out and was cancelled"),
            # model_003 is deliberately missing to verify full-session output.
        ]
    )

    summary = TrainingOrchestrator(MODEL_LIBRARY).execute_ray(
        manifest,
        executor=executor,
    )

    assert summary.completed == 1
    assert summary.failed == 2
    assert "timed out" in (summary.models[1].error or "")
    assert "no result" in (summary.models[2].error or "")


def test_execute_ray_converts_executor_start_failure_to_complete_summary(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, count=2)
    executor = _FakeRayExecutor(start_error=RuntimeError("cluster unavailable"))

    summary = TrainingOrchestrator(MODEL_LIBRARY).execute_ray(
        manifest,
        executor=executor,
    )

    assert executor.closed is True
    assert summary.status == "failed"
    assert summary.completed == 0
    assert summary.failed == 2
    assert all("cluster unavailable" in (item.error or "") for item in summary.models)
    assert [job.status for job in manifest.jobs] == ["failed", "failed"]


def test_execute_ray_can_leave_injected_executor_open(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, count=1)
    executor = _FakeRayExecutor([_completed(manifest.jobs[0], 0.95)])

    TrainingOrchestrator(MODEL_LIBRARY).execute_ray(
        manifest,
        executor=executor,
        close_executor=False,
    )

    assert executor.started is True
    assert executor.closed is False
