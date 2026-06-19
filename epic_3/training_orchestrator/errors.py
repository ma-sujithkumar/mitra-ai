"""Errors raised by Epic-3 model routing and training orchestration."""


class TrainingOrchestratorError(RuntimeError):
    """Base error for the Epic-3 training-orchestrator boundary."""


class InvalidModelConfigError(TrainingOrchestratorError):
    """Raised when model_config.json is empty, malformed, or inconsistent."""


class ModelRoutingError(TrainingOrchestratorError):
    """Raised when a selected model cannot be routed to a supported trainer."""


class MissingDataSplitError(TrainingOrchestratorError):
    """Raised when the train/test artifacts from Epic-2 are unavailable."""


class TrainingExecutionError(TrainingOrchestratorError):
    """Raised when the training worker violates the orchestration contract."""


class ResultAggregationError(TrainingOrchestratorError):
    """Raised when per-model results cannot form a valid session summary."""
