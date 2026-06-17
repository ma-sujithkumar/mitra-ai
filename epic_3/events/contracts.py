"""Typed event contracts for Epic-3 training progress."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TrainingEventLevel = Literal["info", "warn", "error"]
TrainingEventStatus = Literal[
    "queued",
    "submitted",
    "running",
    "completed",
    "failed",
    "timed_out",
    "cancelled",
    "all_completed",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TrainingEvent(BaseModel):
    """One JSON/SSE-safe lifecycle event for a training session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    stage: Literal["training"] = "training"
    level: TrainingEventLevel = "info"
    msg: str = Field(min_length=1)
    pct: int = Field(default=0, ge=0, le=100)
    status: TrainingEventStatus
    ts: datetime = Field(default_factory=_utc_now)
    sequence: int = Field(default=0, ge=0)
    model_id: str | None = Field(default=None, pattern=r"^model_\d{3}$")
    model_name: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
