from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ExecutionMode = Literal["ray", "local"]
TrainingRunStatus = Literal[
    "created",
    "running",
    "completed",
    "partial_failure",
    "failed",
    "cancelled",
]
TrainingModelStatus = Literal[
    "queued",
    "submitted",
    "running",
    "completed",
    "failed",
    "timed_out",
    "cancelled",
]


class TrainingStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    metadata_path: str | None = None
    model_config_path: str | None = None
    train_path: str | None = None
    test_path: str | None = None
    target_column: str | None = None
    problem_type: Literal["classification", "regression", "unsupervised", "auto"] | None = None
    allow_fallback_artifacts: bool = True
    execution_mode: ExecutionMode | None = None
    timeout_sec: float | None = Field(default=None, gt=0.0)


class TrainingStartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: TrainingRunStatus
    execution_mode: ExecutionMode
    status_url: str
    events_url: str


class TrainingModelState(BaseModel):
    """Persisted per-model state mirrored from orchestrator lifecycle events."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(pattern=r"^model_\d{3}$")
    model_name: str
    status: TrainingModelStatus
    pct: int = Field(default=0, ge=0, le=100)
    updated_at: datetime
    validation_score: float | None = None
    model_path: str | None = None
    training_time_sec: float | None = Field(default=None, ge=0.0)
    error: str | None = None


class TrainingStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: TrainingRunStatus
    execution_mode: ExecutionMode
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancellation_requested: bool = False
    cancelled_jobs: int = 0
    manifest_path: str | None = None
    summary_path: str | None = None
    total_models: int | None = None
    completed_models: int | None = None
    failed_models: int | None = None
    job_status_counts: dict[str, int] = Field(default_factory=dict)
    model_states: list[TrainingModelState] = Field(default_factory=list)
    error: str | None = None


class TrainingCancelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: TrainingRunStatus
    cancellation_requested: bool
    cancelled_jobs: int
