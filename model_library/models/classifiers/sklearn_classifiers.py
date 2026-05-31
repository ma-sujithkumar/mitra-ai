# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from core.data_bundle import DataBundle
from models.base import BaseModel


logger = logging.getLogger(__name__)


# Maps config.yaml base_estimator string values to sklearn classes.
# Used by BaggingClassifier and CalibratedClassifierCV to avoid if-else ladders.
ESTIMATOR_CLASS_MAP: dict = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "LinearSVC": LinearSVC,
    "SVC": SVC,
    "LogisticRegression": LogisticRegression,
    "SGDClassifier": SGDClassifier,
}


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    """Guarantee the input array is 2D (n_samples, n_features)."""
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


class LogisticRegressionWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = LogisticRegression(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class RidgeClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = RidgeClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class SGDClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = SGDClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class SVCWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = SVC(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class LinearSVCWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = LinearSVC(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class NuSVCWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = NuSVC(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class DecisionTreeClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = DecisionTreeClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class RandomForestClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = RandomForestClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class ExtraTreesClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = ExtraTreesClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class GradientBoostingClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = GradientBoostingClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class AdaBoostClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = AdaBoostClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class HistGradientBoostingClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = HistGradientBoostingClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class KNeighborsClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = KNeighborsClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class RadiusNeighborsClassifierWrapper(BaseModel):
    """RadiusNeighbors can fail if a test point has no neighbors within the radius.
    The wrapper catches this per-sample and falls back to the majority training class."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.majority_class: int = 0

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        X_train_2d = _ensure_2d(data.common.X_train)
        self.majority_class = int(
            np.bincount(data.common.y_train.astype(int)).argmax()
        )
        self.model = RadiusNeighborsClassifier(**merged_config)
        self.model.fit(X_train_2d, data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            return self.model.predict(_ensure_2d(X))
        except ValueError as radius_error:
            logger.warning(
                "=> RadiusNeighborsClassifier: radius error during predict (%s). "
                "Falling back to majority class %d for all samples.",
                radius_error,
                self.majority_class,
            )
            return np.full(X.shape[0], self.majority_class, dtype=int)


class GaussianNBWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = GaussianNB(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class MultinomialNBWrapper(BaseModel):
    """MultinomialNB requires non-negative inputs. Negative values are clipped to 0."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _clip_non_negative(self, X: np.ndarray, context: str) -> np.ndarray:
        X_array = _ensure_2d(X)
        if np.any(X_array < 0):
            logger.warning(
                "=> MultinomialNB: negative values found in %s. "
                "Clipping to 0 (MultinomialNB requires non-negative input).",
                context,
            )
            X_array = np.clip(X_array, 0, None)
        return X_array

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        X_train_clean = self._clip_non_negative(data.common.X_train, "X_train")
        self.model = MultinomialNB(**merged_config)
        self.model.fit(X_train_clean, data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean = self._clip_non_negative(X, "X_test")
        return self.model.predict(X_clean)


class ComplementNBWrapper(BaseModel):
    """ComplementNB requires non-negative inputs. Negative values are clipped to 0."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _clip_non_negative(self, X: np.ndarray, context: str) -> np.ndarray:
        X_array = _ensure_2d(X)
        if np.any(X_array < 0):
            logger.warning(
                "=> ComplementNB: negative values found in %s. Clipping to 0.",
                context,
            )
            X_array = np.clip(X_array, 0, None)
        return X_array

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        X_train_clean = self._clip_non_negative(data.common.X_train, "X_train")
        self.model = ComplementNB(**merged_config)
        self.model.fit(X_train_clean, data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean = self._clip_non_negative(X, "X_test")
        return self.model.predict(X_clean)


class BernoulliNBWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = BernoulliNB(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class CategoricalNBWrapper(BaseModel):
    """CategoricalNB requires non-negative integer inputs.
    Negative values are clipped to 0. Test inputs are also clipped to the
    per-feature max category seen in training — prevents index-out-of-bounds
    when test samples contain unseen category values."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.max_category_per_feature: np.ndarray = np.array([])

    def _prepare_input(self, X: np.ndarray, context: str) -> np.ndarray:
        X_array = _ensure_2d(X)
        if np.any(X_array < 0):
            logger.warning(
                "=> CategoricalNB: negative values found in %s. Clipping to 0.",
                context,
            )
            X_array = np.clip(X_array, 0, None)
        X_int = X_array.astype(int)
        # Clip unseen categories to training max to avoid index-out-of-bounds at predict
        if self.max_category_per_feature.size > 0:
            X_int = np.minimum(X_int, self.max_category_per_feature)
        return X_int

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        X_train_clean = self._prepare_input(data.common.X_train, "X_train")
        self.max_category_per_feature = X_train_clean.max(axis=0)
        self.model = CategoricalNB(**merged_config)
        self.model.fit(X_train_clean, data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean = self._prepare_input(X, "X_test")
        return self.model.predict(X_clean)


class MLPClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        # hidden_layer_sizes must be a tuple for sklearn
        if "hidden_layer_sizes" in merged_config:
            merged_config["hidden_layer_sizes"] = tuple(
                merged_config["hidden_layer_sizes"]
            )
        self.model = MLPClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class PassiveAggressiveClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = PassiveAggressiveClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class QuadraticDiscriminantAnalysisWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = QuadraticDiscriminantAnalysis(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class LinearDiscriminantAnalysisWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = LinearDiscriminantAnalysis(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class BaggingClassifierWrapper(BaseModel):
    """BaggingClassifier wraps another estimator given by base_estimator in config."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        base_estimator_name = merged_config.pop("base_estimator", "DecisionTreeClassifier")
        base_estimator_class = ESTIMATOR_CLASS_MAP.get(
            base_estimator_name, DecisionTreeClassifier
        )
        self.model = BaggingClassifier(
            estimator=base_estimator_class(), **merged_config
        )
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class DummyClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = DummyClassifier(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class NearestCentroidWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = NearestCentroid(**merged_config)
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))


class CalibratedClassifierCVWrapper(BaseModel):
    """CalibratedClassifierCV wraps another estimator. base_estimator from config."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        base_estimator_name = merged_config.pop("base_estimator", "LinearSVC")
        base_estimator_class = ESTIMATOR_CLASS_MAP.get(base_estimator_name, LinearSVC)
        self.model = CalibratedClassifierCV(
            estimator=base_estimator_class(), **merged_config
        )
        self.model.fit(_ensure_2d(data.common.X_train), data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_ensure_2d(X))
