import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from core.data_bundle import DataBundle
from models.base import BaseModel


class KMeansWrapper(BaseModel):
    """KMeans clustering wrapper. Ignores y during training; predict returns cluster labels."""

    def train(self, data: DataBundle) -> None:
        params = self._merge_hyperparameter_overrides(data)
        self.model = KMeans(
            n_clusters=int(params.get("n_clusters", 8)),
            random_state=int(params.get("random_state", 42)),
            n_init="auto",
        )
        self.model.fit(data.common.X_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MiniBatchKMeansWrapper(BaseModel):
    """MiniBatchKMeans clustering wrapper. Faster than KMeans for large datasets."""

    def train(self, data: DataBundle) -> None:
        params = self._merge_hyperparameter_overrides(data)
        self.model = MiniBatchKMeans(
            n_clusters=int(params.get("n_clusters", 8)),
            random_state=int(params.get("random_state", 42)),
            n_init="auto",
        )
        self.model.fit(data.common.X_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
