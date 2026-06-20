"""Typed errors emitted by the MITRA model-selection component."""


class ModelSelectionError(RuntimeError):
    """Base class for model-selection failures."""


class ModelLibraryCatalogError(ModelSelectionError):
    """Raised when the model-library registry/config cannot be read safely."""


class UnsupportedProblemTypeError(ModelSelectionError):
    """Raised when the current model library has no model for the requested task."""
