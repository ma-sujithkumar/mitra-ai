from __future__ import annotations

import pytest

from backend.agents.ray_wrapper.contracts import RayResourceRequest
from backend.agents.ray_wrapper.config import RaySettings
from backend.agents.ray_wrapper.resources import RayResourcePolicy
from backend.agents.training_orchestrator.contracts import TrainingJob


def _policy() -> RayResourcePolicy:
    settings = RaySettings(
        address=None,
        namespace="test",
        local_num_cpus=2,
        include_dashboard=False,
        job_timeout_sec=30.0,
        default_cpus_per_job=1.0,
        image_cpus_per_job=2.0,
        image_gpus_per_job=1.0,
        default_memory_gb=0.0,
    )
    return RayResourcePolicy.from_settings(settings)


def _image_job(tmp_path) -> TrainingJob:
    return TrainingJob(
        model_id="model_001",
        model_name="PyTorchCNNClassifier",
        task_type="classification",
        data_format="image",
        trainer_type="image_classification",
        parameters={},
        train_path=str(tmp_path / "train.npz"),
        test_path=str(tmp_path / "test.npz"),
        output_dir=str(tmp_path / "model_001"),
        priority=1,
    )


def test_image_job_uses_gpu_when_available(tmp_path) -> None:
    request = _policy().resolve(
        _image_job(tmp_path),
        cluster_resources={"CPU": 8, "GPU": 2},
    )
    assert request.num_cpus == pytest.approx(2.0)
    assert request.num_gpus == pytest.approx(1.0)


def test_image_job_falls_back_to_cpu_without_gpu(tmp_path) -> None:
    request = _policy().resolve(
        _image_job(tmp_path),
        cluster_resources={"CPU": 4, "GPU": 0},
    )
    assert request.num_gpus == 0.0
    assert request.num_cpus == pytest.approx(2.0)


def test_override_is_validated_and_cpu_is_capped(tmp_path) -> None:
    request = _policy().resolve(
        _image_job(tmp_path),
        cluster_resources={"CPU": 2, "GPU": 1},
        override=RayResourceRequest(
            num_cpus=16,
            num_gpus=1,
            memory_bytes=1024,
        ),
    )
    assert request.num_cpus == pytest.approx(2.0)
    assert request.num_gpus == pytest.approx(1.0)
    assert request.memory_bytes == 1024
