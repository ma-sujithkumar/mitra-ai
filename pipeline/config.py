from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ImputationConfig(BaseModel):
    null_drop_threshold: float = Field(ge=0.0, le=1.0)
    knn_neighbors: int = Field(gt=0)
    iterative_max_iter: int = Field(gt=0)
    # Spec §4 "Null detection in categoricals": these tokens stay as literal
    # category labels for categorical/binary columns and are never treated as
    # missing data. The orchestrator also passes them as `na_values` exceptions
    # so pandas does not coerce them at CSV-load time.
    categorical_null_literals: list[str] = Field(
        default_factory=lambda: ["None", "NA", "N/A", "na", "n/a", "none", "NaN"]
    )


class OutlierConfig(BaseModel):
    iqr_multiplier: float = Field(gt=0)
    zscore_threshold: float = Field(gt=0)
    isolation_contamination: float = Field(gt=0, lt=0.5)
    default_action: str


class FeatureCreationConfig(BaseModel):
    max_created_features: int = Field(gt=0)
    equal_width_bins: int = Field(gt=1)
    quantile_bins: int = Field(gt=1)


class FeatureSelectionConfig(BaseModel):
    correlation_threshold: float = Field(ge=0, le=1)
    mi_threshold: float = Field(ge=0)
    variance_threshold: float = Field(ge=0)
    lasso_alpha: float = Field(gt=0)
    rf_n_estimators: int = Field(gt=0)
    pca_variance_retained: float = Field(gt=0, le=1)
    top_k_features: int = Field(gt=0)
    cluster_cut_threshold: float = Field(gt=0, lt=1)
    linear_baseline_k: int = Field(gt=0)


class ScalingConfig(BaseModel):
    power_transformer_method: str


class PipelineSettings(BaseModel):
    max_tool_retries: int = Field(gt=0)
    random_state: int
    max_workers: int = Field(gt=0)
    task_infer_nunique_threshold: int = Field(gt=0)
    downstream_model_hint: Literal["linear", "tree"] = "tree"


class ValidationSettings(BaseModel):
    min_rationale_chars: int = Field(ge=0)
    min_alternatives: int = Field(ge=0)
    lazy_response_threshold: float = Field(ge=0.0, le=1.0)
    boilerplate_denylist: list[str] = Field(default_factory=list)


class LlmConfig(BaseModel):
    max_tokens: int = Field(gt=0)
    api_key_env_var: str = Field(min_length=1, default="OPENAI_API_KEY")
    api_key: str = Field(min_length=1)
    base_url: str | None = None  # optional; if set, used as the OpenAI-compatible endpoint


class ConfigSchema(BaseModel):
    imputation: ImputationConfig
    outlier: OutlierConfig
    feature_creation: FeatureCreationConfig
    feature_selection: FeatureSelectionConfig
    scaling: ScalingConfig
    pipeline: PipelineSettings
    validation: ValidationSettings
    llm: LlmConfig


def load_config(path: str | Path) -> ConfigSchema:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ConfigSchema(**raw)
