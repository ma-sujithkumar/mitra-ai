from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ImputationConfig(BaseModel):
    null_drop_threshold: float = Field(ge=0.0, le=1.0)
    knn_neighbors: int = Field(gt=0)
    iterative_max_iter: int = Field(gt=0)


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


class ScalingConfig(BaseModel):
    power_transformer_method: str


class PipelineSettings(BaseModel):
    max_tool_retries: int = Field(gt=0)
    random_state: int
    max_workers: int = Field(gt=0)


class ConfigSchema(BaseModel):
    imputation: ImputationConfig
    outlier: OutlierConfig
    feature_creation: FeatureCreationConfig
    feature_selection: FeatureSelectionConfig
    scaling: ScalingConfig
    pipeline: PipelineSettings


def load_config(path: str | Path) -> ConfigSchema:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ConfigSchema(**raw)
