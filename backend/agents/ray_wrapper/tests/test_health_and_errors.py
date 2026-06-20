from __future__ import annotations

import pytest

from backend.agents.ray_wrapper.errors import RayInitializationError
from backend.agents.ray_wrapper.executor import RayExecutor

from .conftest import FakeRay


class AlwaysFailRay(FakeRay):
    def init(self, **kwargs):
        self.init_calls.append(dict(kwargs))
        raise RuntimeError("startup blocked")


def test_initialization_error_contains_external_and_local_failures(
    model_library_root,
    ray_settings,
) -> None:
    ray = AlwaysFailRay(initialized=False)
    executor = RayExecutor(model_library_root, ray_module=ray, settings=ray_settings)

    with pytest.raises(RayInitializationError, match="external connection failed"):
        executor.start()

    health = executor.health()
    assert health.ready is False
    assert health.mode == "unavailable"
    assert "startup blocked" in (health.error or "")


def test_health_reports_cluster_resources(model_library_root, ray_settings) -> None:
    ray = FakeRay(
        initialized=True,
        cluster_resources={"CPU": 8, "GPU": 1, "node:abc": 1},
        available_resources={"CPU": 5.5, "GPU": 1},
    )
    executor = RayExecutor(model_library_root, ray_module=ray, settings=ray_settings)
    executor.start()

    health = executor.health()

    assert health.ready is True
    assert health.cluster_resources["CPU"] == 8.0
    assert health.available_resources["CPU"] == 5.5
