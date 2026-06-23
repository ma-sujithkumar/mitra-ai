"""Ray-backed parallel executor for Epic-3 training jobs.

This module owns Ray lifecycle, resource assignment, parallel submission,
completion-order collection, timeout handling, cancellation, and health
reporting. The training orchestrator remains responsible for manifest status
updates and ``training_summary.json`` generation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from time import monotonic
from typing import Any

logger = logging.getLogger(__name__)

import ray

from backend.agents.training.contracts import TrainingResult
from backend.agents.training_orchestrator.contracts import TrainingJob

from .config import RaySettings
from .contracts import RayHealth, RayJobHandle, RayResourceRequest
from .errors import (
    RayExecutionError,
    RayInitializationError,
    RaySubmissionError,
)
from .resources import RayResourcePolicy
from .worker import execute_training_job


class RayExecutor:
    """Submit and collect independent model-training tasks through Ray."""

    def __init__(
        self,
        model_library_root: str | Path,
        *,
        target_column: str | None = None,
        settings: RaySettings | None = None,
        resource_policy: RayResourcePolicy | None = None,
        ray_module: Any | None = None,
    ) -> None:
        self.model_library_root = str(Path(model_library_root).expanduser().resolve())
        self.target_column = target_column
        self.settings = settings or RaySettings.from_project_config()
        self.resource_policy = resource_policy or RayResourcePolicy.from_settings(
            self.settings
        )
        self.ray = ray_module or ray
        self.mode = "uninitialized"
        self.owns_runtime = False
        self.remote_worker: Any | None = None
        self.active_handles: dict[Any, RayJobHandle] = {}
        self.last_error: str | None = None

    def start(self) -> RayHealth:
        """Start a fresh local Ray runtime for this training run.

        Any previously running Ray instance (from a prior training turn or a
        background `ray start`) is shut down first so stale IDLE workers do not
        accumulate and exhaust RAM across judge feedback turns.  The cluster is
        always owned by this executor and is stopped in close().
        """
        # Shut down any leftover Ray cluster so we start completely clean.
        # Stale IDLE workers each hold a Python interpreter + loaded dataset in
        # memory; without this, 2-3 judge turns exhaust all available RAM.
        if bool(self.ray.is_initialized()):
            logger.info("=> RayExecutor.start: stale Ray instance detected, shutting it down before reinit")
            try:
                self.ray.shutdown()
            except Exception as shutdown_exc:
                logger.debug("=> RayExecutor.start: shutdown of stale instance failed (non-fatal): %s", shutdown_exc)

        external_error: Exception | None = None
        if self.settings.address is not None:
            try:
                self.ray.init(
                    address=self.settings.address,
                    namespace=self.settings.namespace,
                    ignore_reinit_error=True,
                )
                self.mode = "external"
                self.owns_runtime = False
                self._ensure_remote_worker()
                self.last_error = None
                return self.health()
            except Exception as exc:
                external_error = exc

        try:
            self.ray.init(
                num_cpus=self.settings.resolved_local_num_cpus(),
                namespace=self.settings.namespace,
                ignore_reinit_error=False,
                include_dashboard=self.settings.include_dashboard,
            )
            self.mode = "local"
            self.owns_runtime = True
            self._ensure_remote_worker()
            self.last_error = None
            return self.health()
        except Exception as exc:
            local_detail = f"local startup failed: {type(exc).__name__}: {exc}"
            detail = local_detail
            if external_error is not None:
                detail = (
                    "external connection failed: "
                    f"{type(external_error).__name__}: {external_error}; "
                    f"{local_detail}"
                )
            self.last_error = detail
            self.mode = "unavailable"
            raise RayInitializationError(detail) from exc

    def health(self) -> RayHealth:
        """Return a non-throwing snapshot for API health reporting."""

        initialized = bool(self.ray.is_initialized())
        cluster_resources: dict[str, float] = {}
        available_resources: dict[str, float] = {}
        error = self.last_error

        if initialized:
            try:
                cluster_resources = self._normalize_resources(
                    self.ray.cluster_resources()
                )
                available_resources = self._normalize_resources(
                    self.ray.available_resources()
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        mode = self.mode
        if initialized and mode in {"uninitialized", "unavailable"}:
            mode = "external"

        return RayHealth(
            ready=initialized and error is None,
            initialized=initialized,
            mode=mode,
            cluster_resources=cluster_resources,
            available_resources=available_resources,
            active_jobs=len(self.active_handles),
            error=error,
        )

    def submit(
        self,
        job: TrainingJob,
        *,
        resources: RayResourceRequest | Mapping[str, Any] | None = None,
    ) -> RayJobHandle:
        """Submit one training job and return its local tracking handle."""

        self._ensure_started()
        remote_worker = self._ensure_remote_worker()
        cluster_resources = self._normalize_resources(self.ray.cluster_resources())
        resolved_resources = self.resource_policy.resolve(
            job,
            cluster_resources=cluster_resources,
            override=resources,
        )

        remote_options: dict[str, Any] = {
            "num_cpus": resolved_resources.num_cpus,
            "num_gpus": resolved_resources.num_gpus,
            "name": f"mitra-{job.model_id}-{job.model_name}",
        }
        if resolved_resources.memory_bytes > 0:
            remote_options["memory"] = resolved_resources.memory_bytes

        try:
            object_ref = remote_worker.options(**remote_options).remote(
                job.model_dump(mode="json"),
                model_library_root=self.model_library_root,
                target_column=self.target_column,
            )
        except Exception as exc:
            raise RaySubmissionError(
                f"Unable to submit {job.model_id}/{job.model_name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        handle = RayJobHandle.create(
            ref=object_ref,
            job=job,
            resources=resolved_resources,
        )
        self.active_handles[object_ref] = handle
        return handle

    def submit_all(
        self,
        jobs: Iterable[TrainingJob],
        *,
        resource_overrides: Mapping[
            str, RayResourceRequest | Mapping[str, Any]
        ] | None = None,
    ) -> list[RayJobHandle]:
        """Submit jobs in priority order without waiting for earlier jobs."""

        ordered_jobs = sorted(
            list(jobs),
            key=lambda item: (item.priority, item.model_id),
        )
        overrides = resource_overrides or {}
        return [
            self.submit(job, resources=overrides.get(job.model_id))
            for job in ordered_jobs
        ]

    def collect(
        self,
        handles: Sequence[RayJobHandle],
        *,
        timeout_sec: float | None = None,
        on_result: Callable[[TrainingResult], None] | None = None,
    ) -> list[TrainingResult]:
        """Collect in completion order while isolating task-level failures."""

        if not handles:
            return []
        self._ensure_started()

        pending = {handle.ref: handle for handle in handles}
        results: list[TrainingResult] = []
        effective_timeout = (
            timeout_sec
            if timeout_sec is not None
            else self.settings.job_timeout_sec
        )
        deadline = monotonic() + effective_timeout

        while pending:
            remaining_time = max(0.0, deadline - monotonic())
            if remaining_time <= 0.0:
                timed_out = self._timeout_pending(pending)
                results.extend(timed_out)
                self._notify_results(timed_out, on_result)
                break

            try:
                ready_refs, _ = self.ray.wait(
                    list(pending),
                    num_returns=1,
                    timeout=remaining_time,
                )
            except Exception as exc:
                failed = self._fail_pending(
                    pending,
                    f"Ray wait failed: {type(exc).__name__}: {exc}",
                )
                results.extend(failed)
                self._notify_results(failed, on_result)
                break

            if not ready_refs:
                timed_out = self._timeout_pending(pending)
                results.extend(timed_out)
                self._notify_results(timed_out, on_result)
                break

            object_ref = ready_refs[0]
            handle = pending.pop(object_ref)
            result = self._collect_one(handle)
            results.append(result)
            self._notify_result(result, on_result)
            self.active_handles.pop(object_ref, None)

        return results


    @staticmethod
    def _notify_result(
        result: TrainingResult,
        callback: Callable[[TrainingResult], None] | None,
    ) -> None:
        if callback is None:
            return
        try:
            callback(result)
        except Exception:
            # Progress reporting is best-effort and must not alter execution.
            pass

    @classmethod
    def _notify_results(
        cls,
        results: Sequence[TrainingResult],
        callback: Callable[[TrainingResult], None] | None,
    ) -> None:
        for result in results:
            cls._notify_result(result, callback)

    def run_all(
        self,
        jobs: Iterable[TrainingJob],
        *,
        timeout_sec: float | None = None,
        resource_overrides: Mapping[
            str, RayResourceRequest | Mapping[str, Any]
        ] | None = None,
        on_result: Callable[[TrainingResult], None] | None = None,
    ) -> list[TrainingResult]:
        """Submit every job, then collect one result per submitted job."""

        handles = self.submit_all(jobs, resource_overrides=resource_overrides)
        return self.collect(
            handles,
            timeout_sec=timeout_sec,
            on_result=on_result,
        )

    def cancel(self, handle: RayJobHandle, *, force: bool = True) -> bool:
        """Cancel one active task and remove its local tracking state."""

        if handle.ref not in self.active_handles:
            return False
        cancellation_succeeded = True
        try:
            self.ray.cancel(handle.ref, force=force)
        except Exception as exc:
            cancellation_succeeded = False
            self.last_error = (
                f"Ray cancellation failed for {handle.job.model_id}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            self.active_handles.pop(handle.ref, None)
        return cancellation_succeeded

    def cancel_all(self, *, force: bool = True) -> int:
        """Cancel every task currently owned by this executor."""

        cancelled_count = 0
        for handle in list(self.active_handles.values()):
            cancelled_count += int(self.cancel(handle, force=force))
        return cancelled_count

    def close(self) -> None:
        """Cancel active jobs and shut down Ray unconditionally.

        Ray is always shut down (not just when owns_runtime=True) so IDLE worker
        processes are freed after every training turn.  Stale workers each consume
        hundreds of MB; without cleanup they accumulate across judge feedback turns
        and exhaust available RAM (swap included) within 2-3 turns.
        """
        self.cancel_all(force=True)
        if bool(self.ray.is_initialized()):
            logger.info("=> RayExecutor.close: shutting down Ray (mode=%s)", self.mode)
            try:
                self.ray.shutdown()
            except Exception as shutdown_exc:
                logger.debug("=> RayExecutor.close: ray.shutdown() raised (non-fatal): %s", shutdown_exc)
        self.mode = "uninitialized"
        self.owns_runtime = False
        self.remote_worker = None

    def __enter__(self) -> "RayExecutor":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _collect_one(self, handle: RayJobHandle) -> TrainingResult:
        try:
            payload = self.ray.get(handle.ref)
            result = TrainingResult.model_validate(payload)
            self._validate_result(handle.job, result)
            return result
        except Exception as exc:
            return self._failed_result(
                handle.job,
                f"Ray task failed: {type(exc).__name__}: {exc}",
            )

    def _ensure_started(self) -> None:
        if not bool(self.ray.is_initialized()):
            self.start()

    def _ensure_remote_worker(self) -> Any:
        if self.remote_worker is None:
            self.remote_worker = self.ray.remote(execute_training_job)
        return self.remote_worker

    def _timeout_pending(
        self,
        pending: dict[Any, RayJobHandle],
    ) -> list[TrainingResult]:
        results: list[TrainingResult] = []
        for object_ref, handle in list(pending.items()):
            try:
                self.ray.cancel(object_ref, force=True)
            except Exception:
                pass
            results.append(
                self._failed_result(
                    handle.job,
                    "Ray task timed out and was cancelled",
                )
            )
            self.active_handles.pop(object_ref, None)
            pending.pop(object_ref, None)
        return results

    def _fail_pending(
        self,
        pending: dict[Any, RayJobHandle],
        error: str,
    ) -> list[TrainingResult]:
        results: list[TrainingResult] = []
        for object_ref, handle in list(pending.items()):
            results.append(self._failed_result(handle.job, error))
            self.active_handles.pop(object_ref, None)
            pending.pop(object_ref, None)
        return results

    @staticmethod
    def _failed_result(job: TrainingJob, error: str) -> TrainingResult:
        return TrainingResult(
            model_id=job.model_id,
            model_name=job.model_name,
            status="failed",
            metrics={},
            model_path=None,
            training_time_sec=0.0,
            error=error,
        )

    @staticmethod
    def _validate_result(job: TrainingJob, result: TrainingResult) -> None:
        if result.model_id != job.model_id:
            raise RayExecutionError(
                f"worker returned model_id '{result.model_id}' for '{job.model_id}'"
            )
        if result.model_name != job.model_name:
            raise RayExecutionError(
                f"worker returned model_name '{result.model_name}' for "
                f"'{job.model_name}'"
            )

    @staticmethod
    def _normalize_resources(
        payload: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in dict(payload or {}).items()
            if isinstance(value, (int, float))
        }
