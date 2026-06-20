"""Typed contracts for Epic-3 model routing and training orchestration.

``TrainingJob`` is produced by Subhasis's router and consumed by Onkar's
training worker.  The summary contracts in this module are produced after the
worker results have been collected.
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
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
SummaryStatus = Literal["completed", "partial_failure", "failed"]


class OrchestratorMetadata(BaseModel):
    """Metadata fields required to route a selected model."""

    model_config = ConfigDict(extra="allow")

    problem_type: Literal["classification", "regression", "unsupervised"]
    data_format: DataFormat = "tabular"
    output_cols: list[str] = Field(default_factory=list)


class SelectedModelConfig(BaseModel):
    """Compatibility view of one ``model_config.json`` entry.

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
    """Stable hand-off contract consumed by the training/Ray layer."""

    model_config = ConfigDict(validate_assignment=True)

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
    """Artifact written by the orchestrator and updated during execution."""

    model_config = ConfigDict(validate_assignment=True)

    session_id: str
    problem_type: TaskType
    data_format: DataFormat
    total_jobs: int = Field(ge=1)
    jobs: list[TrainingJob]

    @model_validator(mode="after")
    def validate_total(self) -> "TrainingJobManifest":
        if self.total_jobs != len(self.jobs):
            raise ValueError("total_jobs must equal the number of jobs")
        model_ids = [job.model_id for job in self.jobs]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("training job model_id values must be unique")
        return self


class TrainingSummaryItem(BaseModel):
    """One normalized result included in ``training_summary.json``."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(pattern=r"^model_\d{3}$")
    model_name: str
    status: Literal["completed", "failed"]
    metrics: dict[str, Any] = Field(default_factory=dict)
    validation_score: float | None = None
    model_path: str | None = None
    training_time_sec: float = Field(ge=0.0)
    error: str | None = None


class TrainingSummary(BaseModel):
    """Session-level result consumed by downstream Epic-3/Epic-4 stages."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: SummaryStatus
    total_models: int = Field(ge=1)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    models: list[TrainingSummaryItem]

    @model_validator(mode="after")
    def validate_counts(self) -> "TrainingSummary":
        if self.total_models != len(self.models):
            raise ValueError("total_models must equal the number of model results")
        actual_completed = sum(item.status == "completed" for item in self.models)
        actual_failed = sum(item.status == "failed" for item in self.models)
        if self.completed != actual_completed or self.failed != actual_failed:
            raise ValueError("completed/failed counts do not match model results")
        if self.completed + self.failed != self.total_models:
            raise ValueError("every model must be completed or failed")

        expected_status: SummaryStatus
        if self.failed == 0:
            expected_status = "completed"
        elif self.completed == 0:
            expected_status = "failed"
        else:
            expected_status = "partial_failure"
        if self.status != expected_status:
            raise ValueError(
                f"summary status must be '{expected_status}' for the current counts"
            )
        return self
