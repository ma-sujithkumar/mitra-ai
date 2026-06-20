from __future__ import annotations

from backend.agents.ray_wrapper.executor import RayExecutor

from .conftest import FakeRay


def test_start_falls_back_from_external_to_local(model_library_root, ray_settings) -> None:
    ray = FakeRay(initialized=False, fail_external=True)
    executor = RayExecutor(
        model_library_root,
        ray_module=ray,
        settings=ray_settings,
    )

    health = executor.start()

    assert health.ready is True
    assert health.mode == "local"
    assert ray.init_calls[0]["address"] == "auto"
    assert ray.init_calls[1]["num_cpus"] == 3

    executor.close()
    assert ray.shutdown_calls == 1


def test_existing_runtime_is_not_shutdown(model_library_root, ray_settings) -> None:
    ray = FakeRay(initialized=True)
    executor = RayExecutor(model_library_root, ray_module=ray, settings=ray_settings)

    assert executor.start().mode == "external"
    executor.close()

    assert ray.shutdown_calls == 0
    assert ray.initialized is True


def test_submit_all_uses_priority_and_resource_options(
    model_library_root,
    training_jobs,
    ray_settings,
) -> None:
    ray = FakeRay(initialized=True, cluster_resources={"CPU": 4, "GPU": 0})
    executor = RayExecutor(model_library_root, ray_module=ray, settings=ray_settings)

    handles = executor.submit_all(reversed(training_jobs))

    assert [handle.job.model_id for handle in handles] == ["model_001", "model_002"]
    assert len(ray.submissions) == 2
    assert ray.submissions[0]["options"]["num_cpus"] == 1.0
    assert ray.submissions[0]["options"]["num_gpus"] == 0.0
    assert ray.submissions[0]["options"]["name"].startswith("mitra-model_001")
    assert executor.health().active_jobs == 2


def test_collect_returns_two_completed_results(
    model_library_root,
    training_jobs,
    ray_settings,
) -> None:
    ray = FakeRay(initialized=True)
    executor = RayExecutor(model_library_root, ray_module=ray, settings=ray_settings)

    handles = executor.submit_all(training_jobs)
    results = executor.collect(handles)

    assert [item.model_id for item in results] == ["model_001", "model_002"]
    assert [item.status for item in results] == ["completed", "completed"]
    assert executor.health().active_jobs == 0


def test_collect_isolates_remote_failure(
    model_library_root,
    training_jobs,
    ray_settings,
) -> None:
    def responder(job_payload, **kwargs):
        del kwargs
        if job_payload["model_id"] == "model_001":
            raise RuntimeError("simulated Ray worker crash")
        return FakeRay._default_responder(
            job_payload,
            model_library_root="ignored",
            target_column=None,
        )

    ray = FakeRay(initialized=True, responder=responder)
    executor = RayExecutor(model_library_root, ray_module=ray, settings=ray_settings)

    results = executor.run_all(training_jobs)

    assert results[0].status == "failed"
    assert "simulated Ray worker crash" in (results[0].error or "")
    assert results[1].status == "completed"


def test_timeout_cancels_every_pending_job(
    model_library_root,
    training_jobs,
    ray_settings,
) -> None:
    ray = FakeRay(initialized=True, never_ready=True)
    executor = RayExecutor(model_library_root, ray_module=ray, settings=ray_settings)
    handles = executor.submit_all(training_jobs)

    results = executor.collect(handles, timeout_sec=0.01)

    assert len(results) == 2
    assert all(item.status == "failed" for item in results)
    assert all("timed out" in (item.error or "") for item in results)
    assert len(ray.cancelled) == 2
    assert executor.health().active_jobs == 0


def test_cancel_all_clears_active_handles(
    model_library_root,
    training_jobs,
    ray_settings,
) -> None:
    ray = FakeRay(initialized=True)
    executor = RayExecutor(model_library_root, ray_module=ray, settings=ray_settings)
    executor.submit_all(training_jobs)

    assert executor.cancel_all() == 2
    assert executor.cancel_all() == 0
    assert len(ray.cancelled) == 2
