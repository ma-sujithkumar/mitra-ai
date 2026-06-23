"""Ray-serializable worker entry point for one Epic-3 training job."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from backend.agents.training.contracts import TrainingResult
from backend.agents.training.trainer import train_job
from backend.agents.training_orchestrator.contracts import TrainingJob


def execute_training_job(
    job_payload: dict[str, Any],
    *,
    model_library_root: str,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Execute one job and return a JSON-compatible ``TrainingResult``.

    The local worker already turns model/data errors into failed results.  This
    outer guard also catches environment or deserialization failures occurring
    before the local worker can create its normal artifact.
    """

    started = perf_counter()
    model_id = str(job_payload.get("model_id", "model_000"))
    model_name = str(job_payload.get("model_name", "unknown"))

    try:
        job = TrainingJob.model_validate(job_payload)
        result = train_job(
            job,
            model_library_root=Path(model_library_root),
            target_column=target_column,
        )
    except Exception as exc:
        result = TrainingResult(
            model_id=model_id,
            model_name=model_name,
            status="failed",
            metrics={},
            model_path=None,
            training_time_sec=perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    return result.model_dump(mode="json")
