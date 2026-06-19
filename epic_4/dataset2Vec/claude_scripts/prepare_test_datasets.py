#!/usr/bin/env python
import os
import sys
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

VENV_ROOT = Path("/home/sujithma/venv/bin/python").parent.parent
sys.path.insert(0, str(VENV_ROOT / "lib" / "python3.11" / "site-packages"))

OUTPUT_DIR = Path(__file__).parent.parent / "test_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_dataset(dataset_name: str, X: np.ndarray, y: np.ndarray, task_type: str = "classification") -> None:
    """Prepare a dataset in NPZ format with train/test split."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    output_path = OUTPUT_DIR / f"{dataset_name}.npz"
    np.savez(
        output_path,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.int32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.int32),
        task_type=np.array(task_type, dtype=object)
    )
    print(f"=> Created {output_path}: X_train={X_train.shape}, X_test={X_test.shape}")

if __name__ == "__main__":
    print("=> Preparing test datasets...")

    # Load sklearn datasets
    iris = load_iris()
    prepare_dataset("test-iris", iris.data, iris.target)

    wine = load_wine()
    prepare_dataset("test-wine", wine.data, wine.target)

    breast_cancer = load_breast_cancer()
    prepare_dataset("test-breast-cancer", breast_cancer.data, breast_cancer.target)

    digits = load_digits()
    prepare_dataset("test-digits", digits.data, digits.target)

    print(f"=> All test datasets ready in {OUTPUT_DIR}")
    print(f"=> Test datasets: {sorted([f.stem for f in OUTPUT_DIR.glob('*.npz')])}")
