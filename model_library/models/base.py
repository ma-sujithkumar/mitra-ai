# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.data_bundle import DataBundle


class BaseModel(ABC):
    """Abstract base class for all 60 MLKit model wrappers.

    Concrete subclasses must implement train() and predict().
    save() and load() use pickle with CPU-safe state so instances are
    safe to pass across Ray workers.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model: Any = None

    @abstractmethod
    def train(self, data: DataBundle) -> None:
        """Fit the underlying model on data.common.{X_train, y_train}."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X. Shape handling is done internally."""

    def _merge_hyperparameter_overrides(self, data: DataBundle) -> dict:
        """Return config merged with any per-call DataBundle.hyperparameters."""
        merged = dict(self.config)
        merged.update(data.hyperparameters)
        return merged

    def save(self, path: str) -> None:
        """Pickle the model to disk. PyTorch models are moved to CPU first."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        model_to_save = self._prepare_for_serialization()
        with open(path, "wb") as file_handle:
            pickle.dump(model_to_save, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """Load a pickled model from disk into self.model."""
        with open(path, "rb") as file_handle:
            self.model = pickle.load(file_handle)

    def _prepare_for_serialization(self) -> Any:
        """Override in PyTorch subclasses to move to CPU before pickling."""
        return self.model
