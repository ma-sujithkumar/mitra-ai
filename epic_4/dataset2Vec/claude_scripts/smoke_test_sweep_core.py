import logging
import os
import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from d2v_core.sweep import classify_model_family, execute_model_trial, metrics_result_to_dict

sys.path.insert(0, "/home/sujithma/mitra/model_library")
from core.data_bundle import CommonData

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    common = CommonData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    assert classify_model_family("LogisticRegression") == "sklearn"
    assert classify_model_family("RandomForestClassifier") == "sklearn"
    assert classify_model_family("XGBClassifier") == "xgboost"
    assert classify_model_family("PyTorchFCNNClassifier") == "pytorch"

    for model_name, hyperparameters in [
        ("LogisticRegression", {"C": 1.0, "max_iter": 500}),
        ("RandomForestClassifier", {"n_estimators": 100, "max_depth": 5}),
        ("XGBClassifier", {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}),
    ]:
        metrics = execute_model_trial(model_name, hyperparameters, common, task_type="classification")
        metrics_dict = metrics_result_to_dict(metrics)
        print(f"=> {model_name}: {metrics_dict}")
        assert metrics_dict["accuracy"] > 0.7, (model_name, metrics_dict)

    print("=> smoke test passed: execute_model_trial works for sklearn + xgboost families.")


if __name__ == "__main__":
    main()
