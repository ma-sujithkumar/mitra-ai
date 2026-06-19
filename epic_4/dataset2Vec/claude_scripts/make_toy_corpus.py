import os

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "toy_corpus")


def write_dataset(dataset_id: str, X: np.ndarray, y: np.ndarray) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_id}.npz")
    np.savez(output_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"=> wrote {output_path} (X_train shape={X_train.shape})")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    iris = load_iris()
    write_dataset("iris", iris.data, iris.target)

    wine = load_wine()
    write_dataset("wine", wine.data, wine.target)

    breast_cancer = load_breast_cancer()
    write_dataset("breast_cancer", breast_cancer.data, breast_cancer.target)

    diabetes = load_diabetes()
    write_dataset("diabetes", diabetes.data, diabetes.target)

    synthetic_X, synthetic_y = make_classification(
        n_samples=300, n_features=25, n_informative=10, n_classes=4, random_state=42
    )
    write_dataset("synthetic_blob", synthetic_X, synthetic_y)


if __name__ == "__main__":
    main()
