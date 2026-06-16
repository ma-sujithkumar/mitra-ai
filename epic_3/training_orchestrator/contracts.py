"""Typed contracts shared between Subhasis's orchestrator and Onkar's trainer.

This module intentionally stops at the job boundary.  It does not train a model,
start Ray, or calculate metrics; those responsibilities belong to the training
executor that will consume :class:`TrainingJob`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskType = Literal["classification", "regression"]
DataFormat = Literal["tabular", "image"]
TrainerType = Literal[
    "tabular_classification",
    "image_classification",
    "tabular_regression",
]
JobStatus = Literal["queued"]


class OrchestratorMetadata(BaseModel):
    """Metadata fields required to route a selected model."""

    model_config = ConfigDict(extra="allow")

    problem_type: Literal["classification", "regression", "unsupervised"]
    data_format: DataFormat = "tabular"
    output_cols: list[str] = Field(default_factory=list)


class SelectedModelConfig(BaseModel):
    """Compatibility view of one model_config.json entry.

    New model-selection output contains both ``name`` and ``model_name``.  The
    pre-validator accepts either field so the router can still consume an older
    file while always exposing a single exact registry key downstream.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    model_name: str
    task_type: TaskType
    priority: int = Field(ge=1)
    rationale: str = ""
    default_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    hp_space: dict[str, Any] = Field(default_factory=dict)
    source: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_model_name(cls, value: Any) -> Any:
        if isinstance(value, dict):
            payload = dict(value)
            payload.setdefault("model_name", payload.get("name"))
            return payload
        return value


class TrainingJob(BaseModel):
    """The stable hand-off contract consumed by the training/Ray layer."""

    model_id: str = Field(pattern=r"^model_\d{3}$")
    model_name: str
    task_type: TaskType
    data_format: DataFormat
    trainer_type: TrainerType
    parameters: dict[str, Any] = Field(default_factory=dict)
    train_path: str
    test_path: str
    output_dir: str
    priority: int = Field(ge=1)
    rationale: str = ""
    status: JobStatus = "queued"
    source: str = "model_library/ml_kit.py::MODEL_REGISTRY"


class TrainingJobManifest(BaseModel):
    """Artifact written by the orchestrator before any training starts."""

    session_id: str
    problem_type: TaskType
    data_format: DataFormat
    total_jobs: int = Field(ge=1)
    jobs: list[TrainingJob]

    @model_validator(mode="after")
    def validate_total(self) -> "TrainingJobManifest":
        if self.total_jobs != len(self.jobs):
            raise ValueError("total_jobs must equal the number of jobs")
        return self
