"""
Shared pytest fixtures loading all mock data from tests/mock_data/.
Reused by unit tests and integration tests to avoid duplication.
"""

import json
import os
import sys

import pytest

# Ensure the repo root is importable so absolute backend.* imports resolve even
# when pytest is invoked from inside this package directory.
# tests -> judge -> evaluation -> agents -> backend -> repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), "mock_data")


def _load_json(filename: str) -> dict:
    filepath = os.path.join(MOCK_DATA_DIR, filename)
    with open(filepath, "r") as json_file:
        return json.load(json_file)


@pytest.fixture(scope="session")
def classification_judge_input_dict() -> dict:
    return _load_json("judge_input_classification.json")


@pytest.fixture(scope="session")
def regression_judge_input_dict() -> dict:
    return _load_json("judge_input_regression.json")


@pytest.fixture(scope="session")
def overfitting_logistic_dict() -> dict:
    return _load_json("overfitting_analysis_LogisticRegression.json")


@pytest.fixture(scope="session")
def shap_rf_dict() -> dict:
    return _load_json("shap_summary_RandomForestClassifier.json")


@pytest.fixture(scope="session")
def hyperparam_gbc_dict() -> dict:
    return _load_json("hyperparam_sensitivity_GradientBoostingClassifier.json")


@pytest.fixture(scope="session")
def judge_config() -> dict:
    from backend.agents.evaluation.judge.config_loader import load_judge_config
    return load_judge_config()
