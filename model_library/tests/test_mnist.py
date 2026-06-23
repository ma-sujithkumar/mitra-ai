#!/home/sujithma/venv/bin/python
# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
"""Runner script: trains and evaluates all 30 classifiers on MNIST.

Usage:
    python tests/test_mnist.py
    python tests/test_mnist.py -v                  # verbose / debug logging
    python tests/test_mnist.py --max-samples 5000  # use more training samples
"""
import argparse
import logging
import os
import sys

import numpy as np
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_bundle import CommonData, DataBundle
from metrics.evaluators import compute_metrics
from ml_kit import MLKit


CLASSIFIER_NAMES = [
    "LogisticRegression",
    "RidgeClassifier",
    "SGDClassifier",
    "SVC",
    "LinearSVC",
    "NuSVC",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "AdaBoostClassifier",
    "HistGradientBoostingClassifier",
    "KNeighborsClassifier",
    "RadiusNeighborsClassifier",
    "GaussianNB",
    "MultinomialNB",
    "ComplementNB",
    "BernoulliNB",
    "CategoricalNB",
    "MLPClassifier",
    "PassiveAggressiveClassifier",
    "QuadraticDiscriminantAnalysis",
    "LinearDiscriminantAnalysis",
    "BaggingClassifier",
    "DummyClassifier",
    "NearestCentroid",
    "CalibratedClassifierCV",
    "XGBClassifier",
    "PyTorchFCNNClassifier",
    "PyTorchCNNClassifier",
]

DATA_DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")

# PyTorch epochs passed via DataBundle.hyperparameters — overrides config.yaml default.
PYTORCH_MODEL_NAMES = {"PyTorchFCNNClassifier", "PyTorchCNNClassifier"}
TEST_PYTORCH_EPOCHS = 2

DEFAULT_MAX_SAMPLES = 100


def load_mnist_as_numpy() -> tuple:
    """Download MNIST via torchvision and return flat numpy arrays (normalized to [0,1])."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=DATA_DOWNLOAD_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=DATA_DOWNLOAD_DIR, train=False, download=True, transform=transform
    )

    X_train_raw = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train_raw = train_dataset.targets.numpy()
    X_test_raw = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test_raw = test_dataset.targets.numpy()

    return X_train_raw, y_train_raw, X_test_raw, y_test_raw


def run_all_classifiers(verbose: bool, max_samples: int) -> None:
    """Load MNIST, run each classifier, print MetricsResult."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s => %(message)s",
    )
    logger = logging.getLogger("test_mnist")

    os.makedirs(DATA_DOWNLOAD_DIR, exist_ok=True)

    logger.info("=> Loading MNIST dataset.")
    X_train, y_train, X_test, y_test = load_mnist_as_numpy()

    if max_samples > 0 and max_samples < X_train.shape[0]:
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
        logger.info("=> Subsampled train set to %d samples.", max_samples)

    logger.info("=> MNIST ready: X_train=%s, X_test=%s", X_train.shape, X_test.shape)

    results_passed = []
    results_failed = []

    for classifier_name in CLASSIFIER_NAMES:
        logger.info("=> Starting classifier: %s", classifier_name)
        try:
            common_data = CommonData(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            hyperparameter_overrides = (
                {"epochs": TEST_PYTORCH_EPOCHS}
                if classifier_name in PYTORCH_MODEL_NAMES
                else {}
            )
            data_bundle = DataBundle(common=common_data, hyperparameters=hyperparameter_overrides)
            kit = MLKit(model_name=classifier_name, data=data_bundle)
            kit.train()
            y_pred = kit.test()
            metrics_result = compute_metrics(
                y_true=y_test,
                y_pred=y_pred,
                task_type="classification",
                model_name=classifier_name,
            )
            print(metrics_result)
            print()
            results_passed.append(classifier_name)
        except Exception as run_error:
            logger.error(
                "=> FAILED [%s]: %s", classifier_name, run_error, exc_info=verbose
            )
            results_failed.append((classifier_name, str(run_error)))

    print("=" * 60)
    print(f"=> Passed: {len(results_passed)}/{len(CLASSIFIER_NAMES)}")
    if results_failed:
        print(f"=> Failed: {len(results_failed)}")
        for failed_name, error_msg in results_failed:
            print(f"   - {failed_name}: {error_msg}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all classifiers on MNIST")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Max training samples (default: {DEFAULT_MAX_SAMPLES}, 0 = full dataset)",
    )
    parsed_args = parser.parse_args()
    run_all_classifiers(verbose=parsed_args.verbose, max_samples=parsed_args.max_samples)


if __name__ == "__main__":
    main()
