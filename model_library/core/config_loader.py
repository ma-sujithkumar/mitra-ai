# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
import configparser
import os
from typing import Optional

import yaml


EXPECTED_MODELS = [
    "LogisticRegression", "RidgeClassifier", "SGDClassifier", "SVC",
    "LinearSVC", "NuSVC", "DecisionTreeClassifier", "RandomForestClassifier",
    "ExtraTreesClassifier", "GradientBoostingClassifier", "AdaBoostClassifier",
    "HistGradientBoostingClassifier", "KNeighborsClassifier",
    "RadiusNeighborsClassifier", "GaussianNB", "MultinomialNB", "ComplementNB",
    "BernoulliNB", "CategoricalNB", "MLPClassifier",
    "PassiveAggressiveClassifier", "QuadraticDiscriminantAnalysis",
    "LinearDiscriminantAnalysis", "BaggingClassifier", "DummyClassifier",
    "NearestCentroid", "CalibratedClassifierCV", "XGBClassifier",
    "PyTorchFCNNClassifier", "PyTorchCNNClassifier",
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "Lars", "LassoLars",
    "OrthogonalMatchingPursuit", "BayesianRidge", "ARDRegression",
    "SGDRegressor", "PassiveAggressiveRegressor", "SVR", "NuSVR",
    "LinearSVR", "KNeighborsRegressor", "RadiusNeighborsRegressor",
    "DecisionTreeRegressor", "RandomForestRegressor", "ExtraTreesRegressor",
    "GradientBoostingRegressor", "AdaBoostRegressor",
    "HistGradientBoostingRegressor", "MLPRegressor", "BaggingRegressor",
    "XGBRegressor", "PyTorchFCNNRegressor", "PyTorchCNNRegressor",
    "DummyRegressor", "HuberRegressor", "TheilSenRegressor",
    # Clustering
    "KMeans", "MiniBatchKMeans",
]


class ConfigValidationError(Exception):
    """Raised when config.yaml is missing required model keys."""


def _resolve_config_yaml_path() -> str:
    """Reads config.ini to locate config.yaml, resolving relative to project root."""
    ini_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.ini")
    ini_path = os.path.normpath(ini_path)

    parser = configparser.ConfigParser()
    parser.read(ini_path)

    relative_yaml_path = parser.get("paths", "config_yaml")
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, relative_yaml_path)


def load_config(model_name: Optional[str] = None) -> dict:
    """Load and validate config.yaml.

    Args:
        model_name: If provided, returns only that model's config dict.
                    If None, returns the full models dict.

    Returns:
        A dict of hyperparameters for the requested model, or all models.

    Raises:
        ConfigValidationError: If any of the 60 expected model keys are absent.
        FileNotFoundError: If config.yaml cannot be found at the resolved path.
    """
    config_yaml_path = _resolve_config_yaml_path()

    with open(config_yaml_path, "r") as config_file:
        raw_config = yaml.safe_load(config_file)

    all_models = raw_config.get("models", {})

    missing_models = [name for name in EXPECTED_MODELS if name not in all_models]
    if missing_models:
        raise ConfigValidationError(
            f"config.yaml is missing entries for {len(missing_models)} model(s): "
            f"{missing_models}. Add these keys under 'models:' in config.yaml."
        )

    if model_name is not None:
        if model_name not in all_models:
            raise ConfigValidationError(
                f"Model '{model_name}' not found in config.yaml. "
                f"Available models: {sorted(all_models.keys())}"
            )
        return dict(all_models[model_name])

    return {name: dict(params) for name, params in all_models.items()}
