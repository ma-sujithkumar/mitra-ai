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
