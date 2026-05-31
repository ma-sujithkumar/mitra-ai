# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
import logging

import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    Ridge,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor

from core.data_bundle import DataBundle
from models.base import BaseModel


logger = logging.getLogger(__name__)


# Maps config.yaml base_estimator string values to sklearn regressor classes.
ESTIMATOR_CLASS_MAP: dict = {
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "SVR": SVR,
    "LinearSVR": LinearSVR,
    "Ridge": Ridge,
    "LinearRegression": LinearRegression,
}


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    """Guarantee the input array is 2D (n_samples, n_features)."""
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


class LinearRegressionWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = LinearRegression(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class RidgeWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = Ridge(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class LassoWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = Lasso(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class ElasticNetWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = ElasticNet(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class LarsWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = Lars(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class LassoLarsWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = LassoLars(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class OrthogonalMatchingPursuitWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = OrthogonalMatchingPursuit(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class BayesianRidgeWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = BayesianRidge(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class ARDRegressionWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = ARDRegression(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class SGDRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = SGDRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class PassiveAggressiveRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = PassiveAggressiveRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class SVRWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = SVR(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class NuSVRWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = NuSVR(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class LinearSVRWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = LinearSVR(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class KNeighborsRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = KNeighborsRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class RadiusNeighborsRegressorWrapper(BaseModel):
    """RadiusNeighbors can fail if a test point has no neighbors within the radius.
    Falls back to mean of training targets on radius failure."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.fallback_value: float = 0.0

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.fallback_value = float(np.mean(data.common.y_train))
        self.model = RadiusNeighborsRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            predictions = self.model.predict(_ensure_2d(X))
        except ValueError as radius_error:
            logger.warning(
                "=> RadiusNeighborsRegressor: radius error during predict (%s). "
                "Falling back to training mean %.4f for all samples.",
                radius_error,
                self.fallback_value,
            )
            return np.full(X.shape[0], self.fallback_value, dtype=float)
        # sklearn returns NaN for samples with no neighbors in radius instead of raising
        nan_mask = np.isnan(predictions)
        if np.any(nan_mask):
            logger.warning(
                "=> RadiusNeighborsRegressor: %d sample(s) had no neighbors within radius. "
                "Replacing NaN with training mean %.4f.",
                nan_mask.sum(),
                self.fallback_value,
            )
            predictions[nan_mask] = self.fallback_value
        return predictions


class DecisionTreeRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = DecisionTreeRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class RandomForestRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = RandomForestRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class ExtraTreesRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = ExtraTreesRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class GradientBoostingRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = GradientBoostingRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class AdaBoostRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = AdaBoostRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class HistGradientBoostingRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = HistGradientBoostingRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class MLPRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        if "hidden_layer_sizes" in merged_config:
            merged_config["hidden_layer_sizes"] = tuple(
                merged_config["hidden_layer_sizes"]
            )
        self.model = MLPRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class BaggingRegressorWrapper(BaseModel):
    """BaggingRegressor wraps another estimator given by base_estimator in config."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        base_estimator_name = merged_config.pop("base_estimator", "DecisionTreeRegressor")
        base_estimator_class = ESTIMATOR_CLASS_MAP.get(
            base_estimator_name, DecisionTreeRegressor
        )
        self.model = BaggingRegressor(
            estimator=base_estimator_class(), **merged_config
        )
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class DummyRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        # Remove null values — DummyRegressor does not accept None for constant/quantile
        cleaned_config = {
            key: value
            for key, value in merged_config.items()
            if value is not None
        }
        self.model = DummyRegressor(**cleaned_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class HuberRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = HuberRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class TheilSenRegressorWrapper(BaseModel):
    """TheilSenRegressor with n_subsamples capped in config.yaml to avoid O(n^2) hangs."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        logger.info(
            "=> TheilSenRegressor: training with n_subsamples=%s "
            "(capped to avoid long runtimes).",
            merged_config.get("n_subsamples"),
        )
        self.model = TheilSenRegressor(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))
