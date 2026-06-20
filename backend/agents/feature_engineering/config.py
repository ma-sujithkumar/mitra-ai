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
    lazy_min_batch_size: int = Field(default=3, ge=1)
    raw_log_max_chars: int = Field(default=60000, ge=1024)
    boilerplate_denylist: list[str] = Field(default_factory=list)


class LlmConfig(BaseModel):
    max_tokens: int = Field(gt=0)
    # Credentials are NOT sourced from config.yaml anymore -- they come from the
    # resolved LlmSettings (.env via LlmSettingsResolver), same as every other
    # agent. These fields are kept optional only for backward compatibility and
    # are ignored by the orchestrator.
    api_key_env_var: str = Field(default="OPENAI_API_KEY")
    api_key: str = ""
    base_url: str | None = None


class PathsConfig(BaseModel):
    # Root for precomputed feature-selection stat artifacts (.mitra/<run_id>/stats).
    workspace_root: str = ".mitra"


class FeatureStatsConfig(BaseModel):
    # Whether the PCA artifact also materialises the transformed components on disk.
    keep_pca_components: bool = False
    # Cap on base columns used to build pairwise correlation/MI artifacts on wide data.
    max_corr_pairs: int = Field(gt=0, default=200)


class ReportConfig(BaseModel):
    # Default off: the report is written from a deterministic template (no LLM call).
    use_llm: bool = False


class ConfigSchema(BaseModel):
    imputation: ImputationConfig
    outlier: OutlierConfig
    feature_creation: FeatureCreationConfig
    feature_selection: FeatureSelectionConfig
    scaling: ScalingConfig
    pipeline: PipelineSettings
    validation: ValidationSettings
    llm: LlmConfig
    # New blocks default to safe values so older config.yaml files still validate.
    paths: PathsConfig = Field(default_factory=PathsConfig)
    feature_stats: FeatureStatsConfig = Field(default_factory=FeatureStatsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


def load_config(path: str | Path) -> ConfigSchema:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ConfigSchema(**raw)
