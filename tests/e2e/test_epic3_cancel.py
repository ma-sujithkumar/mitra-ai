from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any, Callable

from backend.agents.training.contracts import TrainingResult

from .helpers import build_training_client, prepare_session, wait_for_terminal


class BlockingRayExecutor:
    def __init__(self) -> None:
        self.jobs: list[Any] = []
        self.collect_started = Event()
        self.cancelled = Event()
        self.closed = False

    def start(self) -> dict[str, bool]:
        return {"ready": True}

    def submit_all(self, jobs: list[Any], **kwargs: Any) -> list[str]:
        del kwargs
        self.jobs = list(jobs)
        return [job.model_id for job in self.jobs]

    def collect(
        self,
        handles: list[str],
        *,
        timeout_sec: float | None = None,
        on_result: Callable[[TrainingResult], None] | None = None,
    ) -> list[TrainingResult]:
        del handles, timeout_sec, on_result
        self.collect_started.set()
        self.cancelled.wait(timeout=5)
        return [
            TrainingResult(
                model_id=job.model_id,
                model_name=job.model_name,
                status="failed",
                metrics={},
                model_path=None,
                training_time_sec=0.0,
                error="Ray task cancelled",
            )
            for job in self.jobs
        ]

    def cancel_all(self, *, force: bool = True) -> int:
        del force
        self.cancelled.set()
        return len(self.jobs)

    def close(self) -> None:
        self.closed = True
        self.cancelled.set()


def test_cancel_marks_active_models_and_run_state_persistently(e2e_config_loader) -> None:
    session_id = "e2e_cancel"
    session_path = prepare_session(
        e2e_config_loader,
        session_id=session_id,
        metadata={
            "problem_type": "classification",
            "data_format": "tabular",
            "output_cols": ["target"],
        },
        model_config=[
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
        ],
        train_header=["feature", "target"],
        train_rows=[[0.0, 0], [1.0, 1], [2.0, 1], [3.0, 0]],
        test_rows=[[0.5, 0], [1.5, 1]],
    )
    executor = BlockingRayExecutor()
    client, service, _ = build_training_client(
        e2e_config_loader,
        executor_factory=lambda model_root, target: executor,
    )

    start = client.post(
        "/api/training/start",
        json={
            "session_id": session_id,
            "target_column": "target",
            "execution_mode": "ray",
        },
    )
    assert start.status_code == 202
    assert executor.collect_started.wait(timeout=3)

    running_response = client.get(f"/api/training/status/{session_id}")
    assert running_response.status_code == 200
    running_state = running_response.json()
    assert running_state["status"] == "running"
    assert running_state["job_status_counts"] == {"running": 2}
    assert [item["model_id"] for item in running_state["model_states"]] == [
        "model_001",
        "model_002",
    ]

    cancel = client.post(f"/api/training/cancel/{session_id}")
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"
    assert cancel.json()["cancelled_jobs"] == 2

    final_state = wait_for_terminal(client, session_id)
    assert final_state["status"] == "cancelled"
    assert final_state["cancellation_requested"] is True
    assert final_state["job_status_counts"] == {"cancelled": 2}
    assert {item["status"] for item in final_state["model_states"]} == {"cancelled"}

    persisted_status = (session_path / "training" / "training_run.json").read_text(
        encoding="utf-8"
    )
    assert '"status": "cancelled"' in persisted_status
    assert '"cancelled": 2' in persisted_status
    client.close()
    service.shutdown()
