from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from epic_3.events import TrainingEventBus
from epic_3.training.contracts import TrainingResult
from epic_3.training_orchestrator import TrainingJob, TrainingOrchestrator

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_LIBRARY = REPO_ROOT / "model_library"


def _prepare_manifest(tmp_path: Path, bus: TrainingEventBus):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_path.write_text("feature,target\n1,0\n", encoding="utf-8")
    test_path.write_text("feature,target\n2,1\n", encoding="utf-8")

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "problem_type": "classification",
                "data_format": "tabular",
                "output_cols": ["target"],
            }
        ),
        encoding="utf-8",
    )
    model_config_path = tmp_path / "model_config.json"
    model_config_path.write_text(
        json.dumps(
            [
                {
                    "name": "LogisticRegression",
                    "model_name": "LogisticRegression",
                    "task_type": "classification",
                    "priority": 1,
                },
                {
                    "name": "RandomForestClassifier",
                    "model_name": "RandomForestClassifier",
                    "task_type": "classification",
                    "priority": 2,
                },
            ]
        ),
        encoding="utf-8",
    )

    orchestrator = TrainingOrchestrator(MODEL_LIBRARY, event_sink=bus)
    manifest = orchestrator.prepare(
        session_id="event-session",
        metadata_path=metadata_path,
        model_config_path=model_config_path,
        train_path=train_path,
        test_path=test_path,
        session_dir=tmp_path / "session",
    )
    return orchestrator, manifest


class _LocalWorker:
    def run(self, job: TrainingJob) -> TrainingResult:
        if job.model_id == "model_002":
            return TrainingResult(
                model_id=job.model_id,
                model_name=job.model_name,
                status="failed",
                metrics={},
                model_path=None,
                training_time_sec=0.2,
                error="simulated failure",
            )

        model_path = Path(job.output_dir) / "model.pkl"
        model_path.write_bytes(b"model")
        return TrainingResult(
            model_id=job.model_id,
            model_name=job.model_name,
            status="completed",
            metrics={"validation_score": 0.91},
            model_path=str(model_path),
            training_time_sec=0.1,
            error=None,
        )


class _CallbackRayExecutor:
    def __init__(self, results: list[TrainingResult]) -> None:
        self.results = results
        self.closed = False

    def start(self) -> dict[str, bool]:
        return {"ready": True}

    def submit_all(
        self,
        jobs: list[TrainingJob],
        *,
        resource_overrides: dict[str, Any] | None = None,
    ) -> list[str]:
        del resource_overrides
        return [job.model_id for job in jobs]

    def collect(
        self,
        handles: list[str],
        *,
        timeout_sec: float | None = None,
        on_result: Callable[[TrainingResult], None] | None = None,
    ) -> list[TrainingResult]:
        del handles, timeout_sec
        for result in self.results:
            if on_result:
                on_result(result)
        return list(self.results)

    def close(self) -> None:
        self.closed = True


def _completed(job: TrainingJob, score: float) -> TrainingResult:
    model_path = Path(job.output_dir) / "model.pkl"
    model_path.write_bytes(b"model")
    return TrainingResult(
        model_id=job.model_id,
        model_name=job.model_name,
        status="completed",
        metrics={"validation_score": score},
        model_path=str(model_path),
        training_time_sec=0.2,
        error=None,
    )


def _failed(job: TrainingJob, message: str) -> TrainingResult:
    return TrainingResult(
        model_id=job.model_id,
        model_name=job.model_name,
        status="failed",
        metrics={},
        model_path=None,
        training_time_sec=0.0,
        error=message,
    )


def test_local_training_emits_full_lifecycle_and_summary(tmp_path: Path) -> None:
    bus = TrainingEventBus()
    orchestrator, manifest = _prepare_manifest(tmp_path, bus)

    summary = orchestrator.execute_local(manifest, worker=_LocalWorker())
    events = bus.history("event-session")

    assert summary.status == "partial_failure"
    assert [event.status for event in events] == [
        "queued",
        "queued",
        "running",
        "completed",
        "running",
        "failed",
        "all_completed",
    ]
    completed = next(event for event in events if event.status == "completed")
    assert completed.details["validation_score"] == 0.91
    failed = next(event for event in events if event.status == "failed")
    assert failed.level == "error"
    assert failed.details["error"] == "simulated failure"


def test_ray_training_streams_completion_order_and_timeout(tmp_path: Path) -> None:
    bus = TrainingEventBus()
    orchestrator, manifest = _prepare_manifest(tmp_path, bus)
    executor = _CallbackRayExecutor(
        [
            _failed(manifest.jobs[1], "Ray task timed out and was cancelled"),
            _completed(manifest.jobs[0], 0.94),
        ]
    )

    summary = orchestrator.execute_ray(manifest, executor=executor)
    events = bus.history("event-session")

    assert summary.status == "partial_failure"
    assert executor.closed is True
    assert sum(event.status == "submitted" for event in events) == 2
    assert sum(event.status == "running" for event in events) == 2
    assert sum(event.status == "timed_out" for event in events) == 1
    assert sum(event.status == "completed" for event in events) == 1
    assert events[-1].status == "all_completed"
    terminal_models = [
        event.model_id
        for event in events
        if event.status in {"completed", "timed_out"}
    ]
    assert terminal_models == ["model_002", "model_001"]


def test_broken_event_sink_never_breaks_training(tmp_path: Path) -> None:
    class BrokenSink:
        def emit(self, event: object) -> None:
            del event
            raise RuntimeError("browser disconnected")

        def close_session(self, session_id: str) -> None:
            del session_id
            raise RuntimeError("browser disconnected")

        def reset_session(self, session_id: str, *, clear_history: bool = True) -> None:
            del session_id, clear_history
            raise RuntimeError("browser disconnected")

    bus = TrainingEventBus()
    _, manifest = _prepare_manifest(tmp_path, bus)
    orchestrator = TrainingOrchestrator(MODEL_LIBRARY, event_sink=BrokenSink())

    summary = orchestrator.execute_local(manifest, worker=_LocalWorker())

    assert summary.completed == 1
    assert summary.failed == 1
