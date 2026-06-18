import argparse
import logging
import os

import numpy as np
import pandas as pd

TASK_TYPE_CLASSIFICATION = "classification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converts the hadijomaa/dataset2vec raw 'datasets/<name>/' "
        "directories (CSV-no-header predictors/labels/folds files) into this "
        "project's *.npz corpus format (X_train, y_train, X_test, y_test, task_type)."
    )
    parser.add_argument(
        "--raw-datasets-dir", required=True, type=str,
        help="path to the cloned repo's 'datasets/' directory",
    )
    parser.add_argument(
        "--output-corpus-dir", required=True, type=str, help="output directory for *.npz files"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args()


def convert_one_dataset(dataset_dir: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-level train/test split from folds_py.dat: fold==1 rows are the held-out
    test fold, fold==0 rows are train (validation_folds_py.dat is an internal HPO
    validation flag from the original paper's experiments -- not needed for our
    train/test-only Optuna trial scheme, so those rows stay in our training set)."""
    predictors = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}_py.dat"), header=None)
    feature_matrix = np.asarray(predictors, dtype=np.float64)

    labels = pd.read_csv(os.path.join(dataset_dir, "labels_py.dat"), header=None)
    target_vector = np.asarray(labels, dtype=np.float64).reshape(-1)

    folds = pd.read_csv(os.path.join(dataset_dir, "folds_py.dat"), header=None)[0]
    fold_array = np.asarray(folds)

    train_mask = fold_array == 0
    test_mask = fold_array == 1

    X_train = feature_matrix[train_mask]
    y_train = target_vector[train_mask]
    X_test = feature_matrix[test_mask]
    y_test = target_vector[test_mask]
    return X_train, y_train, X_test, y_test


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    os.makedirs(args.output_corpus_dir, exist_ok=True)
    dataset_names = sorted(
        entry for entry in os.listdir(args.raw_datasets_dir)
        if os.path.isdir(os.path.join(args.raw_datasets_dir, entry))
    )

    n_converted = 0
    n_skipped = 0
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(args.raw_datasets_dir, dataset_name)
        X_train, y_train, X_test, y_test = convert_one_dataset(dataset_dir, dataset_name)

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(
                "=> skipping '%s': empty train (%d rows) or test (%d rows) split.",
                dataset_name, len(X_train), len(X_test),
            )
            n_skipped += 1
            continue

        output_path = os.path.join(args.output_corpus_dir, f"{dataset_name}.npz")
        np.savez(
            output_path,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            task_type=TASK_TYPE_CLASSIFICATION,
        )
        n_converted += 1
        logger.info(
            "=> converted '%s': X_train=%s X_test=%s n_classes=%d.",
            dataset_name, X_train.shape, X_test.shape, len(np.unique(y_train)),
        )

    print(f"=> converted {n_converted} dataset(s), skipped {n_skipped}, written to '{args.output_corpus_dir}'.")


if __name__ == "__main__":
    main()
