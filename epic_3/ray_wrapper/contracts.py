"""Typed contracts for Epic-3 Ray health, resources, and job handles."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from epic_3.training_orchestrator.contracts import TrainingJob

RayRuntimeMode = Literal["external", "local", "uninitialized", "unavailable"]


class RayResourceRequest(BaseModel):
    """Resources requested by one remote training task."""

    model_config = ConfigDict(extra="forbid")

    num_cpus: float = Field(default=1.0, gt=0.0)
    num_gpus: float = Field(default=0.0, ge=0.0)
    memory_bytes: int = Field(default=0, ge=0)


class RayHealth(BaseModel):
    """Serializable health snapshot suitable for a FastAPI ``/health`` route."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    initialized: bool
    mode: RayRuntimeMode
    cluster_resources: dict[str, float] = Field(default_factory=dict)
    available_resources: dict[str, float] = Field(default_factory=dict)
    active_jobs: int = Field(default=0, ge=0)
    error: str | None = None


@dataclass(slots=True)
class RayJobHandle:
    """Local bookkeeping for one submitted Ray object reference."""

    ref: Any
    job: TrainingJob
    resources: RayResourceRequest
    submitted_at: float

    @classmethod
    def create(
        cls,
        *,
        ref: Any,
        job: TrainingJob,
        resources: RayResourceRequest,
    ) -> "RayJobHandle":
        return cls(
            ref=ref,
            job=job,
            resources=resources,
            submitted_at=monotonic(),
        )
