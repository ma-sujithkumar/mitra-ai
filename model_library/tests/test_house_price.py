#!/home/sujithma/venv/bin/python
# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
"""Runner script: trains and evaluates all 30 regressors on California Housing.

Usage:
    python tests/test_house_price.py
    python tests/test_house_price.py -v                  # verbose / debug logging
    python tests/test_house_price.py --max-samples 1000  # use more training samples
"""
import argparse
import logging
import os
import sys

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_bundle import CommonData, DataBundle
from metrics.evaluators import compute_metrics
from ml_kit import MLKit


REGRESSOR_NAMES = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "Lars",
    "LassoLars",
    "OrthogonalMatchingPursuit",
    "BayesianRidge",
    "ARDRegression",
    "SGDRegressor",
    "PassiveAggressiveRegressor",
    "SVR",
    "NuSVR",
    "LinearSVR",
    "KNeighborsRegressor",
    "RadiusNeighborsRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "AdaBoostRegressor",
    "HistGradientBoostingRegressor",
    "MLPRegressor",
    "BaggingRegressor",
    "XGBRegressor",
    "PyTorchFCNNRegressor",
    "PyTorchCNNRegressor",
    "DummyRegressor",
    "HuberRegressor",
    "TheilSenRegressor",
]

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42

# PyTorch epochs passed via DataBundle.hyperparameters — overrides config.yaml default.
PYTORCH_MODEL_NAMES = {"PyTorchFCNNRegressor", "PyTorchCNNRegressor"}
TEST_PYTORCH_EPOCHS = 2

DEFAULT_MAX_SAMPLES = 100


def load_california_housing_as_numpy(max_samples: int) -> tuple:
    """Load California Housing, split train/test, standardize features.

    The test suite is responsible for splitting per the spec.
    Standardization keeps distance-based and gradient models convergent.
    """
    housing_dataset = fetch_california_housing()
    X_all = housing_dataset.data.astype(np.float32)
    y_all = housing_dataset.target.astype(np.float32)

    if max_samples > 0 and max_samples < X_all.shape[0]:
        X_all = X_all[:max_samples]
        y_all = y_all[:max_samples]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_all, y_all, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_scaled, y_train_raw, X_test_scaled, y_test_raw


def run_all_regressors(verbose: bool, max_samples: int) -> None:
    """Load California Housing, run each regressor, print MetricsResult."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s => %(message)s",
    )
    logger = logging.getLogger("test_house_price")

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    logger.info("=> Loading California Housing dataset (max_samples=%d).", max_samples)
    X_train, y_train, X_test, y_test = load_california_housing_as_numpy(max_samples)
    logger.info("=> Dataset ready: X_train=%s, X_test=%s", X_train.shape, X_test.shape)

    results_passed = []
    results_failed = []

    for regressor_name in REGRESSOR_NAMES:
        logger.info("=> Starting regressor: %s", regressor_name)
        try:
            common_data = CommonData(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            hyperparameter_overrides = (
                {"epochs": TEST_PYTORCH_EPOCHS}
                if regressor_name in PYTORCH_MODEL_NAMES
                else {}
            )
            data_bundle = DataBundle(common=common_data, hyperparameters=hyperparameter_overrides)
            kit = MLKit(model_name=regressor_name, data=data_bundle)
            kit.train()
            y_pred = kit.test()
            metrics_result = compute_metrics(
                y_true=y_test,
                y_pred=y_pred,
                task_type="regression",
                model_name=regressor_name,
            )
            print(metrics_result)
            print()
            results_passed.append(regressor_name)
        except Exception as run_error:
            logger.error(
                "=> FAILED [%s]: %s", regressor_name, run_error, exc_info=verbose
            )
            results_failed.append((regressor_name, str(run_error)))

    print("=" * 60)
    print(f"=> Passed: {len(results_passed)}/{len(REGRESSOR_NAMES)}")
    if results_failed:
        print(f"=> Failed: {len(results_failed)}")
        for failed_name, error_msg in results_failed:
            print(f"   - {failed_name}: {error_msg}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all regressors on California Housing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Max total samples before splitting (default: {DEFAULT_MAX_SAMPLES}, 0 = full dataset)",
    )
    parsed_args = parser.parse_args()
    run_all_regressors(verbose=parsed_args.verbose, max_samples=parsed_args.max_samples)


if __name__ == "__main__":
    main()
