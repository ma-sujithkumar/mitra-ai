"""Typed data carrier threaded through every stage of the SHAP explainability pipeline.

architecture.md Section 4/5: SessionContext holds the facts produced by each pipeline
stage (loaders, validators, explainers, exporters) so later stages and the metadata
exporter can read them without re-deriving anything or relying on global state. This
module only defines the data shape - it does not load models, compute SHAP values, or
export artifacts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ExecutionStatus(Enum):
    """Lifecycle status of a single SHAP pipeline run (spec.md Sec 18, Sec 20)."""

    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"


class ModelNameValidationStatus(Enum):
    """Outcome of comparing the supplied model_name against the detected model type
    (spec.md Sec 8 Rules 1-4)."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNDETECTABLE = "undetectable"
    UNSUPPORTED = "unsupported"


@dataclass
class SessionContext:
    """Mutable, pipeline-scoped state for a single SHAP explainability execution.

    Required fields are the Sec 4 input contract. Every other field starts as None
    or empty and is populated by exactly one downstream stage; no stage overwrites a
    field already populated by an earlier stage.

    Attributes:
        session_id: Unique execution identifier from the input contract (Sec 4.1).
        supplied_model_name: model_name supplied by Epic 3, treated as metadata only
            (Sec 4.2, Sec 8).
        pickle_file_path: Path to the trained model artifact (Sec 4.3).
        engineered_dataset_path: Path to the Epic 2 engineered dataset (Sec 4.4).
        created_at: UTC timestamp the SessionContext was constructed.
        execution_status: Current lifecycle status of the run.
        detected_model_type: Model class name detected from the loaded artifact
            (Sec 8), populated by the model loading stage.
        model_name_validation_status: Result of comparing supplied_model_name against
            detected_model_type (Sec 8 Rules 1-4), populated by the model validator.
        model_name_validation_message: Human-readable detail for the validation
            result above.
        target_column_name: Name of the column excluded from SHAP processing, if any
            (Sec 11), populated by the schema validator.
        feature_names: Ordered feature names used for SHAP processing, dynamically
            resolved from the engineered dataset (Sec 13), populated by the schema
            validator.
        num_samples: Row count of the feature-only dataset used for SHAP processing.
        num_features: Length of feature_names.
        explainer_name: Name of the SHAP explainer selected for the detected model
            type (Sec 14), populated by the explainer factory.
        shap_values: Raw SHAP values computed for the feature-only dataset, populated
            by the SHAP service. Left untyped (Any) since this module does not define
            or depend on the SHAP computation's concrete return type.
        global_feature_importance: Aggregated per-feature importance table (Sec
            17.1), populated by the SHAP service.
        feature_shap_mapping: Long-form per-record/per-feature SHAP table (Sec 17.2),
            populated by the SHAP service.
        warnings: Non-terminating warning messages accumulated across stages (e.g.
            Sec 8 Rule 2 model-name mismatch).
        error_message: Terminating failure message, set only when execution_status
            becomes FAILED.
        extra_metadata: Additional key-value facts any stage wants to surface in
            metadata.json without requiring a new SessionContext field for every
            future addition (Sec 24 Extensibility).
    """

    session_id: str
    supplied_model_name: str
    pickle_file_path: str
    engineered_dataset_path: str

    created_at: datetime = field(default_factory=datetime.utcnow)
    execution_status: ExecutionStatus = ExecutionStatus.RUNNING

    detected_model_type: Optional[str] = None
    model_name_validation_status: Optional[ModelNameValidationStatus] = None
    model_name_validation_message: Optional[str] = None

    target_column_name: Optional[str] = None
    feature_names: Optional[list[str]] = None
    num_samples: Optional[int] = None
    num_features: Optional[int] = None

    explainer_name: Optional[str] = None

    shap_values: Optional[Any] = None
    global_feature_importance: Optional[Any] = None
    feature_shap_mapping: Optional[Any] = None

    warnings: list[str] = field(default_factory=list)
    error_message: Optional[str] = None

    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def add_warning(self, warning_message: str) -> None:
        """Records a non-terminating warning and promotes the run's status.

        Used for cases such as Sec 8 Rule 2 (model-name mismatch), where execution
        must continue but the discrepancy still needs to surface in metadata.json.

        Args:
            warning_message: Human-readable description of the warning condition.
        """
        self.warnings.append(warning_message)
        if self.execution_status == ExecutionStatus.RUNNING:
            self.execution_status = ExecutionStatus.WARNING

    def mark_failed(self, failure_message: str) -> None:
        """Marks the run as terminated by a validation or execution failure (Sec 20).

        Args:
            failure_message: Human-readable description of why execution stopped.
        """
        self.execution_status = ExecutionStatus.FAILED
        self.error_message = failure_message

    def mark_success(self) -> None:
        """Marks the run as completed.

        If warnings were already recorded, the status remains WARNING rather than
        being overwritten to SUCCESS, since spec.md Sec 18 treats a model-name
        mismatch run as a successful-but-flagged execution, not a clean success.
        """
        if self.execution_status == ExecutionStatus.RUNNING:
            self.execution_status = ExecutionStatus.SUCCESS

    def has_failed(self) -> bool:
        """Returns True if the run has been marked as a terminating failure."""
        return self.execution_status == ExecutionStatus.FAILED
