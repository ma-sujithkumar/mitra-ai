from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from backend.agents.training.contracts import TrainingResult

from .helpers import build_training_client, prepare_session, wait_for_terminal


class ScenarioRayExecutor:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.jobs: list[Any] = []
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
        del handles, timeout_sec
        results: list[TrainingResult] = []
        for index, job in enumerate(reversed(self.jobs)):
            should_fail = index == 0
            if should_fail:
                error = (
                    "Ray task timed out and was cancelled"
                    if self.scenario == "timeout"
                    else "Ray task failed: synthetic worker failure"
                )
                result = TrainingResult(
                    model_id=job.model_id,
                    model_name=job.model_name,
                    status="failed",
                    metrics={},
                    model_path=None,
                    training_time_sec=0.0,
                    error=error,
                )
            else:
                model_path = Path(job.output_dir) / "model.pkl"
                model_path.write_bytes(b"fake-model")
                result = TrainingResult(
                    model_id=job.model_id,
                    model_name=job.model_name,
                    status="completed",
                    metrics={"validation_score": 0.91, "primary_metric": "accuracy"},
                    model_path=str(model_path),
                    training_time_sec=0.1,
                    error=None,
                )
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    def cancel_all(self, *, force: bool = True) -> int:
        del force
        return len(self.jobs)

    def close(self) -> None:
        self.closed = True


def prepare_ray_session(config_loader, session_id: str) -> Path:
    return prepare_session(
        config_loader,
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


@pytest.mark.parametrize(
    ("scenario", "expected_terminal_model_status"),
    [("partial", "failed"), ("timeout", "timed_out")],
)
def test_ray_partial_failure_and_timeout_are_persisted(
    e2e_config_loader,
    scenario: str,
    expected_terminal_model_status: str,
) -> None:
    session_id = f"e2e_ray_{scenario}"
    session_path = prepare_ray_session(e2e_config_loader, session_id)
    executor = ScenarioRayExecutor(scenario)
    client, service, event_bus = build_training_client(
        e2e_config_loader,
        executor_factory=lambda model_root, target: executor,
    )

    response = client.post(
        "/api/training/start",
        json={
            "session_id": session_id,
            "target_column": "target",
            "execution_mode": "ray",
            "timeout_sec": 0.1,
        },
    )
    assert response.status_code == 202
    final_state = wait_for_terminal(client, session_id)

    assert final_state["status"] == "partial_failure"
    assert final_state["completed_models"] == 1
    assert final_state["failed_models"] == 1
    assert sorted(item["status"] for item in final_state["model_states"]) == sorted(
        ["completed", expected_terminal_model_status]
    )

    manifest = json.loads(
        (session_path / "training" / "training_jobs.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (session_path / "training" / "training_summary.json").read_text(encoding="utf-8")
    )
    assert [item["model_id"] for item in summary["models"]] == [
        item["model_id"] for item in manifest["jobs"]
    ]
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert any(event.status == expected_terminal_model_status for event in event_bus.history(session_id))
    assert executor.closed is True
    client.close()
    service.shutdown()
