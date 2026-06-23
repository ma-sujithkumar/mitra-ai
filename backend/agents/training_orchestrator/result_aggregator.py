"""Aggregate per-model ``TrainingResult`` values into a session summary."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from backend.agents.training.contracts import TrainingResult

from .contracts import (
    SummaryStatus,
    TrainingJobManifest,
    TrainingSummary,
    TrainingSummaryItem,
)
from .errors import ResultAggregationError


class TrainingResultAggregator:
    """Validate worker results and preserve manifest order in the summary."""

    def build(
        self,
        *,
        manifest: TrainingJobManifest,
        results: Iterable[TrainingResult],
    ) -> TrainingSummary:
        result_list = list(results)
        result_ids = [result.model_id for result in result_list]
        if len(result_ids) != len(set(result_ids)):
            raise ResultAggregationError("training results contain duplicate model_id values")

        expected = {job.model_id: job for job in manifest.jobs}
        unexpected = sorted(set(result_ids) - set(expected))
        if unexpected:
            raise ResultAggregationError(
                f"training results contain unknown model_id values: {unexpected}"
            )

        missing = sorted(set(expected) - set(result_ids))
        if missing:
            raise ResultAggregationError(
                f"training results are missing model_id values: {missing}"
            )

        by_id = {result.model_id: result for result in result_list}
        items: list[TrainingSummaryItem] = []
        for job in manifest.jobs:
            result = by_id[job.model_id]
            if result.model_name != job.model_name:
                raise ResultAggregationError(
                    f"result model_name mismatch for {job.model_id}: "
                    f"expected '{job.model_name}', got '{result.model_name}'"
                )
            validation_score = result.metrics.get("validation_score")
            if validation_score is not None:
                try:
                    validation_score = float(validation_score)
                except (TypeError, ValueError) as exc:
                    raise ResultAggregationError(
                        f"validation_score for {job.model_id} must be numeric"
                    ) from exc

            items.append(
                TrainingSummaryItem(
                    model_id=result.model_id,
                    model_name=result.model_name,
                    status=result.status,
                    metrics=result.metrics,
                    validation_score=validation_score,
                    model_path=result.model_path,
                    training_time_sec=result.training_time_sec,
                    error=result.error,
                )
            )

        completed = sum(item.status == "completed" for item in items)
        failed = len(items) - completed
        status: SummaryStatus
        if failed == 0:
            status = "completed"
        elif completed == 0:
            status = "failed"
        else:
            status = "partial_failure"

        return TrainingSummary(
            session_id=manifest.session_id,
            status=status,
            total_models=len(items),
            completed=completed,
            failed=failed,
            models=items,
        )

    def write(self, summary: TrainingSummary, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(destination, summary.model_dump(mode="json"))
        return destination

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        temp_name: str | None = None
        try:
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except (OSError, TypeError, ValueError) as exc:
            if temp_name:
                try:
                    Path(temp_name).unlink(missing_ok=True)
                except OSError:
                    pass
            raise ResultAggregationError(
                f"unable to write training summary {path}: {exc}"
            ) from exc
