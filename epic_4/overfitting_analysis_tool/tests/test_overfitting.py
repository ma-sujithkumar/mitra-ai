import argparse
import json
import logging
import os
import sys
import tempfile

import numpy as np
from sklearn.datasets import make_classification, make_regression

# Bootstrap: add tool root to sys.path so we can import OverfittingAnalyzer directly.
_TOOL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _TOOL_ROOT not in sys.path:
    sys.path.insert(0, _TOOL_ROOT)

from overfitting_analysis import OverfittingAnalyzer

logger = logging.getLogger("test_overfitting")

REQUIRED_OUTPUT_KEYS = [
    "model_name", "model_type", "is_overfitted", "gap_threshold",
    "primary_metric", "gaps", "rel_rmse_gap", "train_metrics",
    "test_metrics", "k_fold_cross_validation_results", "cv_skipped_reason",
]

KFOLD_REQUIRED_KEYS = ["k", "scoring", "per_fold_scores", "mean", "std", "train_vs_cv_gap"]


def _save_dataset(output_dir: str, filename: str, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> str:
    """Save arrays to a .npz file and return its path."""
    npz_path = os.path.join(output_dir, filename)
    np.savez(npz_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return npz_path + ".npz"


def _write_input_json(output_dir: str, filename: str, model_type: str, model_name: str,
                      dataset_path: str) -> str:
    """Write a minimal input JSON and return its path."""
    json_path = os.path.join(output_dir, filename)
    payload = {
        "model_type": model_type,
        "model_name": model_name,
        "dataset_path": dataset_path,
    }
    with open(json_path, "w") as json_file:
        json.dump(payload, json_file)
    return json_path


def _assert_output_schema(result: dict, model_type: str, k_folds: int) -> None:
    """Assert that all required keys are present and K-fold results are well-formed."""
    for key in REQUIRED_OUTPUT_KEYS:
        assert key in result, f"Output JSON missing key: '{key}'"

    gaps = result["gaps"]
    assert isinstance(gaps, dict) and len(gaps) > 0, "gaps dict must be non-empty"

    if model_type == "classification":
        assert "accuracy" in gaps, "Classification gaps must include 'accuracy'"
        assert result["rel_rmse_gap"] is None, "rel_rmse_gap must be null for classification"
    else:
        assert "r2" in gaps, "Regression gaps must include 'r2'"
        assert result["rel_rmse_gap"] is not None, "rel_rmse_gap must be non-null for regression"

    kfold = result["k_fold_cross_validation_results"]
    if kfold is not None:
        for key in KFOLD_REQUIRED_KEYS:
            assert key in kfold, f"K-fold result missing key: '{key}'"
        assert len(kfold["per_fold_scores"]) == k_folds, (
            f"Expected {k_folds} fold scores, got {len(kfold['per_fold_scores'])}"
        )
        assert isinstance(kfold["mean"], float), "K-fold mean must be float"
        assert isinstance(kfold["std"], float), "K-fold std must be float"


def run_classification_overfit_case(work_dir: str, verbose: bool) -> None:
    """
    Overfit case: DecisionTreeClassifier with no depth limit trained on small dataset.
    High train accuracy, poor test accuracy => is_overfitted should be True.
    """
    logger.info("=> [Classification/Overfit] Starting test case.")

    rng = np.random.default_rng(0)
    num_train = 60
    num_test = 40
    num_features = 10
    num_classes = 2

    X_all, y_all = make_classification(
        n_samples=num_train + num_test,
        n_features=num_features,
        n_informative=3,
        n_redundant=2,
        n_classes=num_classes,
        random_state=7,
    )
    X_train = X_all[:num_train].astype(np.float32)
    y_train = y_all[:num_train]
    X_test = X_all[num_train:].astype(np.float32)
    y_test = y_all[num_train:]

    dataset_path = _save_dataset(work_dir, "clf_overfit_data", X_train, y_train, X_test, y_test)
    input_json_path = _write_input_json(
        work_dir, "clf_overfit_input.json",
        "classification", "DecisionTreeClassifier", dataset_path,
    )
    output_dir = os.path.join(work_dir, "clf_overfit_output")

    analyzer = OverfittingAnalyzer(
        input_json_path=input_json_path,
        output_dir=output_dir,
        verbose=verbose,
    )
    result = analyzer.run()

    _assert_output_schema(result, "classification", k_folds=5)
    assert result["is_overfitted"] is True, (
        f"Overfit DecisionTree should be flagged. gaps={result['gaps']}"
    )
    logger.info("=> [Classification/Overfit] PASSED. is_overfitted=%s gaps=%s", result["is_overfitted"], result["gaps"])


def run_classification_wellfit_case(work_dir: str, verbose: bool) -> None:
    """
    Well-fit case: LogisticRegression on a well-separated dataset.
    Train and test metrics should be close => is_overfitted should be False.
    """
    logger.info("=> [Classification/WellFit] Starting test case.")

    num_train = 200
    num_test = 100

    X_all, y_all = make_classification(
        n_samples=num_train + num_test,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        class_sep=2.0,
        random_state=42,
    )
    X_train = X_all[:num_train].astype(np.float32)
    y_train = y_all[:num_train]
    X_test = X_all[num_train:].astype(np.float32)
    y_test = y_all[num_train:]

    dataset_path = _save_dataset(work_dir, "clf_wellfit_data", X_train, y_train, X_test, y_test)
    input_json_path = _write_input_json(
        work_dir, "clf_wellfit_input.json",
        "classification", "LogisticRegression", dataset_path,
    )
    output_dir = os.path.join(work_dir, "clf_wellfit_output")

    analyzer = OverfittingAnalyzer(
        input_json_path=input_json_path,
        output_dir=output_dir,
        verbose=verbose,
    )
    result = analyzer.run()

    _assert_output_schema(result, "classification", k_folds=5)
    assert result["is_overfitted"] is False, (
        f"LogisticRegression on well-separated data should not be flagged. gaps={result['gaps']}"
    )
    logger.info("=> [Classification/WellFit] PASSED. is_overfitted=%s gaps=%s", result["is_overfitted"], result["gaps"])


def run_regression_case(work_dir: str, verbose: bool) -> None:
    """
    Regression case: LinearRegression on a clean dataset.
    Checks schema, rel_rmse_gap presence, and verdict plausibility.
    """
    logger.info("=> [Regression/WellFit] Starting test case.")

    num_train = 200
    num_test = 80

    X_all, y_all = make_regression(
        n_samples=num_train + num_test,
        n_features=10,
        n_informative=7,
        noise=5.0,
        random_state=42,
    )
    X_train = X_all[:num_train].astype(np.float32)
    y_train = y_all[:num_train].astype(np.float32)
    X_test = X_all[num_train:].astype(np.float32)
    y_test = y_all[num_train:].astype(np.float32)

    dataset_path = _save_dataset(work_dir, "reg_wellfit_data", X_train, y_train, X_test, y_test)
    input_json_path = _write_input_json(
        work_dir, "reg_wellfit_input.json",
        "regression", "LinearRegression", dataset_path,
    )
    output_dir = os.path.join(work_dir, "reg_wellfit_output")

    analyzer = OverfittingAnalyzer(
        input_json_path=input_json_path,
        output_dir=output_dir,
        verbose=verbose,
    )
    result = analyzer.run()

    _assert_output_schema(result, "regression", k_folds=5)
    assert isinstance(result["rel_rmse_gap"], float), "rel_rmse_gap must be float for regression"
    assert "r2" in result["gaps"], "Regression gaps must contain r2"
    logger.info(
        "=> [Regression/WellFit] PASSED. is_overfitted=%s rel_rmse_gap=%.4f gaps=%s",
        result["is_overfitted"], result["rel_rmse_gap"], result["gaps"],
    )


def run_kfold_reproducibility_case(work_dir: str, verbose: bool) -> None:
    """
    Reproducibility: two runs with same random_state must produce identical per_fold_scores.
    """
    logger.info("=> [KFold/Reproducibility] Starting test case.")

    num_train = 150
    num_test = 50

    X_all, y_all = make_classification(
        n_samples=num_train + num_test,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=99,
    )
    X_train = X_all[:num_train].astype(np.float32)
    y_train = y_all[:num_train]
    X_test = X_all[num_train:].astype(np.float32)
    y_test = y_all[num_train:]

    dataset_path = _save_dataset(work_dir, "kfold_repro_data", X_train, y_train, X_test, y_test)
    input_json_path = _write_input_json(
        work_dir, "kfold_repro_input.json",
        "classification", "LogisticRegression", dataset_path,
    )

    results = []
    for run_index in range(2):
        output_dir = os.path.join(work_dir, f"kfold_repro_output_{run_index}")
        analyzer = OverfittingAnalyzer(
            input_json_path=input_json_path,
            output_dir=output_dir,
            verbose=verbose,
        )
        results.append(analyzer.run())

    scores_run_0 = results[0]["k_fold_cross_validation_results"]["per_fold_scores"]
    scores_run_1 = results[1]["k_fold_cross_validation_results"]["per_fold_scores"]

    assert scores_run_0 == scores_run_1, (
        f"K-fold scores must be reproducible.\nRun 0: {scores_run_0}\nRun 1: {scores_run_1}"
    )
    logger.info("=> [KFold/Reproducibility] PASSED. Scores: %s", scores_run_0)


def run_all_tests(verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s => %(message)s",
    )

    with tempfile.TemporaryDirectory() as work_dir:
        pass_count = 0
        fail_count = 0
        test_cases = [
            ("Classification/Overfit", run_classification_overfit_case),
            ("Classification/WellFit", run_classification_wellfit_case),
            ("Regression/WellFit", run_regression_case),
            ("KFold/Reproducibility", run_kfold_reproducibility_case),
        ]

        # Use a persistent tmp dir per session so intermediate files are inspectable.
        run_dir = tempfile.mkdtemp(prefix="overfitting_test_")
        logger.info("=> Test working directory: %s", run_dir)

        for test_name, test_func in test_cases:
            try:
                test_func(run_dir, verbose)
                logger.info("=> PASS: %s", test_name)
                pass_count += 1
            except AssertionError as assertion_err:
                logger.error("=> FAIL: %s -- %s", test_name, assertion_err)
                fail_count += 1
            except Exception as unexpected_err:
                logger.error("=> ERROR: %s -- %s", test_name, unexpected_err, exc_info=True)
                fail_count += 1

        logger.info(
            "=> Test results: %d passed, %d failed out of %d total.",
            pass_count, fail_count, len(test_cases),
        )
        if fail_count > 0:
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run overfitting analysis tool test suite.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Enable debug-level logging.")
    parsed_args = parser.parse_args()
    run_all_tests(verbose=parsed_args.verbose)


if __name__ == "__main__":
    main()
