from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.ray_wrapper.config import RaySettings
from backend.agents.training.contracts import TrainingResult
from backend.agents.training_orchestrator.contracts import TrainingJob


@dataclass(frozen=True)
class FakeObjectRef:
    value: int


class FakeRemoteFunction:
    def __init__(
        self,
        ray: "FakeRay",
        function: Callable[..., Any],
        options: dict[str, Any] | None = None,
    ) -> None:
        self.ray = ray
        self.function = function
        self._options = dict(options or {})

    def options(self, **kwargs: Any) -> "FakeRemoteFunction":
        merged = dict(self._options)
        merged.update(kwargs)
        return FakeRemoteFunction(self.ray, self.function, merged)

    def remote(self, *args: Any, **kwargs: Any) -> FakeObjectRef:
        ref = FakeObjectRef(self.ray.next_ref)
        self.ray.next_ref += 1
        self.ray.submissions.append(
            {
                "ref": ref,
                "args": args,
                "kwargs": kwargs,
                "options": dict(self._options),
            }
        )
        try:
            payload = self.ray.responder(*args, **kwargs)
        except Exception as exc:  # Store task failure for ray.get().
            payload = exc
        self.ray.values[ref] = payload
        return ref


class FakeRay:
    def __init__(
        self,
        *,
        initialized: bool = False,
        fail_external: bool = False,
        cluster_resources: dict[str, float] | None = None,
        available_resources: dict[str, float] | None = None,
        responder: Callable[..., Any] | None = None,
        never_ready: bool = False,
    ) -> None:
        self.initialized = initialized
        self.fail_external = fail_external
        self._cluster_resources = cluster_resources or {"CPU": 4.0, "GPU": 0.0}
        self._available_resources = available_resources or dict(self._cluster_resources)
        self.responder = responder or self._default_responder
        self.never_ready = never_ready
        self.next_ref = 1
        self.values: dict[FakeObjectRef, Any] = {}
        self.submissions: list[dict[str, Any]] = []
        self.init_calls: list[dict[str, Any]] = []
        self.cancelled: list[tuple[FakeObjectRef, bool]] = []
        self.shutdown_calls = 0

    def is_initialized(self) -> bool:
        return self.initialized

    def init(self, **kwargs: Any) -> None:
        self.init_calls.append(dict(kwargs))
        if kwargs.get("address") and self.fail_external:
            raise ConnectionError("no external Ray cluster")
        self.initialized = True

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.initialized = False

    def cluster_resources(self) -> dict[str, float]:
        return dict(self._cluster_resources)

    def available_resources(self) -> dict[str, float]:
        return dict(self._available_resources)

    def remote(self, function: Callable[..., Any]) -> FakeRemoteFunction:
        return FakeRemoteFunction(self, function)

    def wait(
        self,
        refs: list[FakeObjectRef],
        *,
        num_returns: int,
        timeout: float | None,
    ) -> tuple[list[FakeObjectRef], list[FakeObjectRef]]:
        if self.never_ready:
            return [], refs
        ready = refs[:num_returns]
        return ready, refs[num_returns:]

    def get(self, ref: FakeObjectRef) -> Any:
        value = self.values[ref]
        if isinstance(value, Exception):
            raise value
        return value

    def cancel(self, ref: FakeObjectRef, *, force: bool) -> None:
        self.cancelled.append((ref, force))

    @staticmethod
    def _default_responder(
        job_payload: dict[str, Any],
        *,
        model_library_root: str,
        target_column: str | None,
    ) -> dict[str, Any]:
        del model_library_root, target_column
        model_id = job_payload["model_id"]
        model_path = str(Path(job_payload["output_dir"]) / "model.pkl")
        return TrainingResult(
            model_id=model_id,
            model_name=job_payload["model_name"],
            status="completed",
            metrics={"validation_score": 0.9},
            model_path=model_path,
            training_time_sec=0.1,
            error=None,
        ).model_dump(mode="json")


@pytest.fixture
def ray_settings() -> RaySettings:
    return RaySettings(
        address="auto",
        namespace="mitra-epic3-test",
        local_num_cpus=3,
        include_dashboard=False,
        job_timeout_sec=1.0,
        default_cpus_per_job=1.0,
        image_cpus_per_job=2.0,
        image_gpus_per_job=1.0,
        default_memory_gb=0.0,
    )


@pytest.fixture
def model_library_root() -> Path:
    return REPO_ROOT / "model_library"


@pytest.fixture
def training_jobs(tmp_path: Path) -> list[TrainingJob]:
    jobs: list[TrainingJob] = []
    for index, name in enumerate(
        ["RandomForestClassifier", "LogisticRegression"],
        start=1,
    ):
        output = tmp_path / "session" / f"model_{index:03d}"
        output.mkdir(parents=True)
        jobs.append(
            TrainingJob(
                model_id=f"model_{index:03d}",
                model_name=name,
                task_type="classification",
                data_format="tabular",
                trainer_type="tabular_classification",
                parameters={},
                train_path=str(tmp_path / "train.csv"),
                test_path=str(tmp_path / "test.csv"),
                output_dir=str(output),
                priority=index,
            )
        )
    return jobs
