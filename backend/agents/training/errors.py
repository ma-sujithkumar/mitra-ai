"""Errors raised internally by the local training pipeline."""


class TrainingPipelineError(RuntimeError):
    """Base error for deterministic training-pipeline failures."""


class TrainingDataError(TrainingPipelineError):
    """Raised when a train/test artifact cannot be loaded or validated."""


class ModelLibraryExecutionError(TrainingPipelineError):
    """Raised when the selected model cannot be executed through MLKit."""


class ArtifactWriteError(TrainingPipelineError):
    """Raised when a model or metrics artifact cannot be persisted."""
