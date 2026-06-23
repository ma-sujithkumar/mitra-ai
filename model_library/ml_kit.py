#!/home/sujithma/venv/bin/python
# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
import logging
from typing import Optional

import numpy as np

from core.config_loader import load_config
from core.data_bundle import DataBundle
from core.validators import validate_data_bundle, validate_model_name
from models.base import BaseModel
from models.classifiers.pytorch_classifiers import (
    PyTorchCNNClassifierWrapper,
    PyTorchFCNNClassifierWrapper,
)
from models.classifiers.sklearn_classifiers import (
    AdaBoostClassifierWrapper,
    BaggingClassifierWrapper,
    BernoulliNBWrapper,
    CalibratedClassifierCVWrapper,
    CategoricalNBWrapper,
    ComplementNBWrapper,
    DecisionTreeClassifierWrapper,
    DummyClassifierWrapper,
    ExtraTreesClassifierWrapper,
    GaussianNBWrapper,
    GradientBoostingClassifierWrapper,
    HistGradientBoostingClassifierWrapper,
    KNeighborsClassifierWrapper,
    LinearDiscriminantAnalysisWrapper,
    LinearSVCWrapper,
    LogisticRegressionWrapper,
    MLPClassifierWrapper,
    MultinomialNBWrapper,
    NearestCentroidWrapper,
    NuSVCWrapper,
    PassiveAggressiveClassifierWrapper,
    QuadraticDiscriminantAnalysisWrapper,
    RadiusNeighborsClassifierWrapper,
    RandomForestClassifierWrapper,
    RidgeClassifierWrapper,
    SGDClassifierWrapper,
    SVCWrapper,
)
from models.classifiers.xgboost_classifiers import XGBClassifierWrapper
from models.regressors.pytorch_regressors import (
    PyTorchCNNRegressorWrapper,
    PyTorchFCNNRegressorWrapper,
)
from models.regressors.sklearn_regressors import (
    ARDRegressionWrapper,
    AdaBoostRegressorWrapper,
    BaggingRegressorWrapper,
    BayesianRidgeWrapper,
    DecisionTreeRegressorWrapper,
    DummyRegressorWrapper,
    ElasticNetWrapper,
    ExtraTreesRegressorWrapper,
    GradientBoostingRegressorWrapper,
    HistGradientBoostingRegressorWrapper,
    HuberRegressorWrapper,
    KNeighborsRegressorWrapper,
    LarsWrapper,
    LassoLarsWrapper,
    LassoWrapper,
    LinearRegressionWrapper,
    LinearSVRWrapper,
    MLPRegressorWrapper,
    NuSVRWrapper,
    OrthogonalMatchingPursuitWrapper,
    PassiveAggressiveRegressorWrapper,
    RadiusNeighborsRegressorWrapper,
    RandomForestRegressorWrapper,
    RidgeWrapper,
    SGDRegressorWrapper,
    SVRWrapper,
    TheilSenRegressorWrapper,
)
from models.regressors.xgboost_regressors import XGBRegressorWrapper
from models.clustering.sklearn_clusterers import KMeansWrapper, MiniBatchKMeansWrapper


logger = logging.getLogger(__name__)


# Registry mapping model_name string to its wrapper class.
# No if-else — all dispatch is done via this dict.
MODEL_REGISTRY: dict = {
    # Classifiers
    "LogisticRegression": LogisticRegressionWrapper,
    "RidgeClassifier": RidgeClassifierWrapper,
    "SGDClassifier": SGDClassifierWrapper,
    "SVC": SVCWrapper,
    "LinearSVC": LinearSVCWrapper,
    "NuSVC": NuSVCWrapper,
    "DecisionTreeClassifier": DecisionTreeClassifierWrapper,
    "RandomForestClassifier": RandomForestClassifierWrapper,
    "ExtraTreesClassifier": ExtraTreesClassifierWrapper,
    "GradientBoostingClassifier": GradientBoostingClassifierWrapper,
    "AdaBoostClassifier": AdaBoostClassifierWrapper,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifierWrapper,
    "KNeighborsClassifier": KNeighborsClassifierWrapper,
    "RadiusNeighborsClassifier": RadiusNeighborsClassifierWrapper,
    "GaussianNB": GaussianNBWrapper,
    "MultinomialNB": MultinomialNBWrapper,
    "ComplementNB": ComplementNBWrapper,
    "BernoulliNB": BernoulliNBWrapper,
    "CategoricalNB": CategoricalNBWrapper,
    "MLPClassifier": MLPClassifierWrapper,
    "PassiveAggressiveClassifier": PassiveAggressiveClassifierWrapper,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysisWrapper,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysisWrapper,
    "BaggingClassifier": BaggingClassifierWrapper,
    "DummyClassifier": DummyClassifierWrapper,
    "NearestCentroid": NearestCentroidWrapper,
    "CalibratedClassifierCV": CalibratedClassifierCVWrapper,
    "XGBClassifier": XGBClassifierWrapper,
    "PyTorchFCNNClassifier": PyTorchFCNNClassifierWrapper,
    "PyTorchCNNClassifier": PyTorchCNNClassifierWrapper,
    # Regressors
    "LinearRegression": LinearRegressionWrapper,
    "Ridge": RidgeWrapper,
    "Lasso": LassoWrapper,
    "ElasticNet": ElasticNetWrapper,
    "Lars": LarsWrapper,
    "LassoLars": LassoLarsWrapper,
    "OrthogonalMatchingPursuit": OrthogonalMatchingPursuitWrapper,
    "BayesianRidge": BayesianRidgeWrapper,
    "ARDRegression": ARDRegressionWrapper,
    "SGDRegressor": SGDRegressorWrapper,
    "PassiveAggressiveRegressor": PassiveAggressiveRegressorWrapper,
    "SVR": SVRWrapper,
    "NuSVR": NuSVRWrapper,
    "LinearSVR": LinearSVRWrapper,
    "KNeighborsRegressor": KNeighborsRegressorWrapper,
    "RadiusNeighborsRegressor": RadiusNeighborsRegressorWrapper,
    "DecisionTreeRegressor": DecisionTreeRegressorWrapper,
    "RandomForestRegressor": RandomForestRegressorWrapper,
    "ExtraTreesRegressor": ExtraTreesRegressorWrapper,
    "GradientBoostingRegressor": GradientBoostingRegressorWrapper,
    "AdaBoostRegressor": AdaBoostRegressorWrapper,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressorWrapper,
    "MLPRegressor": MLPRegressorWrapper,
    "BaggingRegressor": BaggingRegressorWrapper,
    "XGBRegressor": XGBRegressorWrapper,
    "PyTorchFCNNRegressor": PyTorchFCNNRegressorWrapper,
    "PyTorchCNNRegressor": PyTorchCNNRegressorWrapper,
    "DummyRegressor": DummyRegressorWrapper,
    "HuberRegressor": HuberRegressorWrapper,
    "TheilSenRegressor": TheilSenRegressorWrapper,
    # Clustering
    "KMeans": KMeansWrapper,
    "MiniBatchKMeans": MiniBatchKMeansWrapper,
}

TRAINING_MODE_FULL_TRAIN = "full_train"
TRAINING_MODE_FINE_TUNE = "fine_tune"


class MLKit:
    """Single entry point for instantiating, training, and testing any of the 60 ML models.

    Designed for agentic use: model_name is a string, data is a DataBundle,
    and the instance is fully pickle-serializable for Ray worker dispatch.

    Example:
        from ml_kit import MLKit
        from core.data_bundle import CommonData, DataBundle
        import numpy as np

        common = CommonData(X_train, y_train, X_test, y_test)
        data = DataBundle(common=common)
        kit = MLKit(model_name="XGBClassifier", data=data)
        kit.train()
        y_pred = kit.test()
    """

    def __init__(
        self,
        model_name: str,
        data: DataBundle,
        preloaded_model: Optional[str] = None,
        training_mode: str = TRAINING_MODE_FINE_TUNE,
    ) -> None:
        """
        Args:
            model_name: Exact name of one of the 60 registered models (see README.md).
            data: DataBundle containing CommonData arrays and optional hyperparameter overrides.
            preloaded_model: Optional path to a .pkl file to load before training.
            training_mode: 'fine_tune' continues from the loaded state (default).
                           'full_train' reinitializes the model weights before training.
        """
        validate_model_name(model_name, training_mode)
        validate_data_bundle(data)

        self.model_name = model_name
        self.data = data
        self.preloaded_model_path = preloaded_model
        self.training_mode = training_mode

        model_config = load_config(model_name)
        wrapper_class = MODEL_REGISTRY[model_name]
        self.model: BaseModel = wrapper_class(model_config)

        if preloaded_model is not None:
            logger.info(
                "=> MLKit: loading preloaded model from '%s' (training_mode=%s).",
                preloaded_model,
                training_mode,
            )
            self.model.load(preloaded_model)

    def train(self) -> None:
        """Train the model on data.common.{X_train, y_train}.

        When training_mode='full_train' and a preloaded model was provided,
        the wrapper is re-instantiated first to reset weights before fitting.
        When training_mode='fine_tune', training continues from the loaded state.
        """
        if (
            self.training_mode == TRAINING_MODE_FULL_TRAIN
            and self.preloaded_model_path is not None
        ):
            logger.info(
                "=> MLKit: training_mode=full_train — reinitializing %s before fitting.",
                self.model_name,
            )
            model_config = load_config(self.model_name)
            wrapper_class = MODEL_REGISTRY[self.model_name]
            self.model = wrapper_class(model_config)

        logger.info("=> MLKit: training %s.", self.model_name)
        self.model.train(self.data)
        logger.info("=> MLKit: training complete for %s.", self.model_name)

    def test(self) -> np.ndarray:
        """Run inference on data.common.X_test and return predictions.

        Returns:
            y_pred: np.ndarray of predicted labels (classification) or values (regression).
            Pass y_pred to metrics.evaluators.compute_metrics() to get a MetricsResult.
        """
        logger.info("=> MLKit: running inference for %s.", self.model_name)
        return self.model.predict(self.data.common.X_test)

    def save(self, path: str) -> None:
        """Pickle the trained model to disk. PyTorch models are CPU-safe."""
        logger.info("=> MLKit: saving %s to '%s'.", self.model_name, path)
        self.model.save(path)

    @classmethod
    def load_from(
        cls,
        model_name: str,
        path: str,
        data: DataBundle,
        training_mode: str = TRAINING_MODE_FINE_TUNE,
    ) -> "MLKit":
        """Convenience constructor: creates MLKit with a preloaded model.

        Args:
            model_name: Name of the model to load.
            path: Path to the .pkl file.
            data: DataBundle for subsequent train/test calls.
            training_mode: See MLKit.__init__.

        Returns:
            A ready-to-use MLKit instance with the model loaded.
        """
        return cls(
            model_name=model_name,
            data=data,
            preloaded_model=path,
            training_mode=training_mode,
        )
