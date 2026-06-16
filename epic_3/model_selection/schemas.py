"""Pydantic contracts for model selection inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TaskType = Literal["classification", "regression"]
DataFormat = Literal["tabular", "image"]


class MetadataInput(BaseModel):
    """Subset of metadata.json consumed by model selection.

    ``extra='allow'`` keeps this component forward-compatible with metadata
    fields owned by other epics.
    """

    model_config = ConfigDict(extra="allow")

    problem_type: Literal["classification", "regression", "unsupervised"]
    data_format: DataFormat = "tabular"
    output_cols: list[str] = Field(default_factory=list)
    input_cols: list[str] = Field(default_factory=list)
    drop_cols: list[str] = Field(default_factory=list)
    col_types: dict[str, str] = Field(default_factory=dict)
    row_count: int = Field(default=0, ge=0)
    col_count: int = Field(default=0, ge=0)
    class_balance: dict[str, float | int] = Field(default_factory=dict)
    user_description: str = ""

    @field_validator("output_cols", "input_cols", "drop_cols")
    @classmethod
    def unique_columns(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class EngineeredFeature(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    formula: str = ""


class FeatureSelectionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    keep: list[str] = Field(default_factory=list)
    drop: list[str] = Field(default_factory=list)
    engineered: list[EngineeredFeature] = Field(default_factory=list)
    rationale: dict[str, str] = Field(default_factory=dict)


class DatasetProfile(BaseModel):
    task_type: TaskType
    data_format: DataFormat
    row_count: int
    original_col_count: int
    selected_feature_count: int
    numeric_feature_count: int
    categorical_feature_count: int
    class_count: int | None = None
    imbalance_ratio: float | None = None
    user_description: str = ""


class ModelDescriptor(BaseModel):
    """One exact entry discovered from MLKit.MODEL_REGISTRY."""

    model_name: str
    task_type: TaskType
    wrapper_class: str
    import_module: str
    default_hyperparameters: dict[str, Any] = Field(default_factory=dict)


class RankedSuggestion(BaseModel):
    """Internal suggestion emitted by either the LLM or deterministic ranker."""

    model_name: str
    rationale: str = ""
    score: float = 0.0


class ModelCandidate(BaseModel):
    """One entry written to model_config.json.

    ``model_name`` is the exact MLKit registry key used by downstream training.
    ``name`` is retained as a compatibility alias for older consumers.
    """

    name: str
    model_name: str
    task_type: TaskType
    priority: int = Field(ge=1)
    rationale: str
    selection_score: float
    default_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    hp_space: dict[str, Any] = Field(default_factory=dict)
    source: str = "model_library/ml_kit.py::MODEL_REGISTRY"


class SelectionReport(BaseModel):
    task_type: TaskType
    data_format: DataFormat
    selected_count: int
    available_model_count: int
    selection_mode: Literal["llm", "deterministic", "llm_with_fallback"]
    invalid_llm_models: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
