"""Resource policy for Epic-3 Ray training jobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from epic_3.training_orchestrator.contracts import TrainingJob

from .config import RaySettings
from .contracts import RayResourceRequest


class RayResourcePolicy(BaseModel):
    """Cluster-aware resource policy for tabular and image training jobs."""

    model_config = ConfigDict(extra="forbid")

    default_cpus: float = Field(gt=0.0)
    image_cpus: float = Field(gt=0.0)
    image_gpus: float = Field(ge=0.0)
    default_memory_gb: float = Field(ge=0.0)

    @classmethod
    def from_settings(cls, settings: RaySettings) -> "RayResourcePolicy":
        return cls(
            default_cpus=settings.default_cpus_per_job,
            image_cpus=settings.image_cpus_per_job,
            image_gpus=settings.image_gpus_per_job,
            default_memory_gb=settings.default_memory_gb,
        )

    def resolve(
        self,
        job: TrainingJob,
        *,
        cluster_resources: Mapping[str, Any] | None = None,
        override: RayResourceRequest | Mapping[str, Any] | None = None,
    ) -> RayResourceRequest:
        """Return a schedulable resource request for one training job."""

        resources = dict(cluster_resources or {})
        total_cpu = float(resources.get("CPU", 0.0) or 0.0)
        total_gpu = float(resources.get("GPU", 0.0) or 0.0)

        request = self._default_request(job) if override is None else self._parse(override)

        requested_cpu = request.num_cpus
        if total_cpu > 0.0:
            requested_cpu = min(requested_cpu, total_cpu)
            requested_cpu = max(requested_cpu, min(0.1, total_cpu))

        # GPU jobs fall back to CPU when the cluster has no GPU. If a cluster
        # exposes fractional GPU capacity, use only the available amount.
        requested_gpu = 0.0
        if request.num_gpus > 0.0 and total_gpu > 0.0:
            requested_gpu = min(request.num_gpus, total_gpu)

        return RayResourceRequest(
            num_cpus=requested_cpu,
            num_gpus=requested_gpu,
            memory_bytes=request.memory_bytes,
        )

    def _default_request(self, job: TrainingJob) -> RayResourceRequest:
        is_image_job = job.trainer_type == "image_classification"
        requested_cpus = self.image_cpus if is_image_job else self.default_cpus
        requested_gpus = self.image_gpus if is_image_job else 0.0
        return RayResourceRequest(
            num_cpus=requested_cpus,
            num_gpus=requested_gpus,
            memory_bytes=int(self.default_memory_gb * (1024**3)),
        )

    @staticmethod
    def _parse(
        override: RayResourceRequest | Mapping[str, Any],
    ) -> RayResourceRequest:
        if isinstance(override, RayResourceRequest):
            return override
        return RayResourceRequest.model_validate(override)
