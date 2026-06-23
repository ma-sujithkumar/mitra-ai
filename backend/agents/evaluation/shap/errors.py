"""Structured exception hierarchy for the SHAP Explainability Module.

All domain failures raised by the pipeline inherit from SHAPModuleError so callers
can catch the base type for blanket handling or catch specific subtypes for
fine-grained recovery (e.g. MetadataExporter always receives a populated failure
message regardless of which stage terminated).
"""


class SHAPModuleError(RuntimeError):
    """Base exception for all SHAP Explainability Module failures."""


class ModelLoadError(SHAPModuleError):
    """Raised when the model artifact cannot be loaded or deserialized."""


class DatasetLoadError(SHAPModuleError):
    """Raised when the engineered dataset cannot be loaded or validated."""


class ModelValidationError(SHAPModuleError):
    """Raised when model type detection fails (Rule 3) or the detected type is
    unsupported (Rule 4) per spec.md Section 8. Rule 2 (name mismatch) is a
    non-terminating warning and never raises this error."""


class SchemaValidationError(SHAPModuleError):
    """Raised when dataset and model feature schemas are incompatible (spec.md
    Sec 12): feature count mismatch, feature name mismatch, or zero features
    remain after target column exclusion."""


class SHAPExecutionError(SHAPModuleError):
    """Raised when explainer construction fails, shap_values() call fails,
    SHAP value shape normalization cannot produce a canonical form, or the
    model family has no explainer mapping entry in model_type_detection.json."""


class ExportError(SHAPModuleError):
    """Raised when a CSV or JSON artifact cannot be written to disk.

    Wraps OSError from file I/O so the pipeline failure path can catch all
    domain failures uniformly via SHAPModuleError."""


class VisualizationError(SHAPModuleError):
    """Raised when a plot cannot be generated or saved to disk.

    Wraps exceptions from shap.summary_plot(), plt.savefig(), or matplotlib
    so the pipeline failure path can catch all domain failures uniformly
    via SHAPModuleError."""
