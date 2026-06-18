"""Local, single-job training worker for Epic-3.

Ray wrapping, SSE events, Page-2 UI, and session-level result aggregation are
intentionally outside this work item.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from epic_3.training_orchestrator.contracts import TrainingJob

from .artifact_writer import (
    model_artifact_path,
    prepare_output_directory,
    write_training_result,
)
from .contracts import TrainingResult
from .data_loader import load_training_data
from .library_adapter import MLKitTrainingAdapter


class LocalTrainingWorker:
    """Train and evaluate exactly one ``TrainingJob`` without Ray."""

    def __init__(
        self,
        model_library_root: str | Path,
        *,
        target_column: str | None = None,
    ) -> None:
        self.adapter = MLKitTrainingAdapter(model_library_root)
        self.target_column = target_column

    def run(self, job: TrainingJob) -> TrainingResult:
        started = perf_counter()
        output_dir = prepare_output_directory(job.output_dir)

        try:
            data = load_training_data(
                job.train_path,
                job.test_path,
                target_column=self.target_column,
            )
            artifacts = self.adapter.train_and_evaluate(
                job=job,
                data=data,
                model_path=model_artifact_path(output_dir),
            )
            result = TrainingResult(
                model_id=job.model_id,
                model_name=job.model_name,
                status="completed",
                metrics=artifacts.metrics,
                model_path=str(artifacts.model_path),
                training_time_sec=perf_counter() - started,
                error=None,
            )
        except Exception as exc:
            result = TrainingResult(
                model_id=job.model_id,
                model_name=job.model_name,
                status="failed",
                metrics={},
                model_path=None,
                training_time_sec=perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

        write_training_result(
            result,
            output_dir,
            extra={
                "task_type": job.task_type,
                "data_format": job.data_format,
                "trainer_type": job.trainer_type,
            },
        )
        return result


def train_job(
    job: TrainingJob,
    *,
    model_library_root: str | Path,
    target_column: str | None = None,
) -> TrainingResult:
    """Functional entry point suitable for a future Ray remote wrapper."""

    return LocalTrainingWorker(
        model_library_root,
        target_column=target_column,
    ).run(job)
