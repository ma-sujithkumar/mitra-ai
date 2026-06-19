"""Typed result contract produced by the local training worker.

This module belongs to Onkar's Epic-3 training-pipeline work.  It deliberately
contains no Ray, SSE, UI, or orchestration logic.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TrainingStatus = Literal["completed", "failed"]


class TrainingResult(BaseModel):
    """Stable result returned after executing one ``TrainingJob`` locally."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(pattern=r"^model_\d{3}$")
    model_name: str
    status: TrainingStatus
    metrics: dict[str, Any] = Field(default_factory=dict)
    model_path: str | None = None
    training_time_sec: float = Field(ge=0.0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "TrainingResult":
        if self.status == "completed":
            if not self.model_path:
                raise ValueError("completed results must include model_path")
            if self.error is not None:
                raise ValueError("completed results must not include error")
        else:
            if not self.error:
                raise ValueError("failed results must include error")
            if self.model_path is not None:
                raise ValueError("failed results must not include model_path")
        return self
