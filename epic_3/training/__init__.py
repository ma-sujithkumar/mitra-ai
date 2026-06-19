"""Epic-3 local training pipeline owned by Onkar."""

from .contracts import TrainingResult
from .trainer import LocalTrainingWorker, train_job

__all__ = ["LocalTrainingWorker", "TrainingResult", "train_job"]
