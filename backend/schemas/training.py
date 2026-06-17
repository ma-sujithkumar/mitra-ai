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


class TrainingStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    metadata_path: str | None = None
    model_config_path: str | None = None
    train_path: str | None = None
    test_path: str | None = None
    target_column: str | None = None
    execution_mode: ExecutionMode | None = None
    timeout_sec: float | None = Field(default=None, gt=0.0)


class TrainingStartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: TrainingRunStatus
    execution_mode: ExecutionMode
    status_url: str
    events_url: str


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
    completed_models: int | None = None
    failed_models: int | None = None
    error: str | None = None


class TrainingCancelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: TrainingRunStatus
    cancellation_requested: bool
    cancelled_jobs: int
