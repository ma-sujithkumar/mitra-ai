"""Unit tests for backend.agents.evaluation.shap.loaders.model_loader."""

import json
import pickle
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from backend.agents.evaluation.shap.errors import ModelLoadError
from backend.agents.evaluation.shap.loaders.model_loader import LoadedModel, ModelLoader
from backend.agents.evaluation.shap.utils.logger import ExecutionLogger

# ---------------------------------------------------------------------------
# Minimal training data used across all model-creation helpers
# ---------------------------------------------------------------------------
_TRAINING_FEATURES_ARRAY = np.array(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
)
_TRAINING_LABELS_ARRAY = np.array([0, 1, 0, 1, 0])
_TRAINING_FEATURES_DATAFRAME = pd.DataFrame(
    {
        "feature_alpha": [1.0, 3.0, 5.0, 7.0, 9.0],
        "feature_beta": [2.0, 4.0, 6.0, 8.0, 10.0],
    }
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_test_logger(tmp_path: Path) -> ExecutionLogger:
    """Creates an ExecutionLogger writing to a unique temp log file."""
    return ExecutionLogger(
        session_id=f"test-{uuid.uuid4().hex}",
        log_file_path=tmp_path / "logs" / "execution.log",
    )


def _write_model_type_config(directory: Path) -> Path:
    """Writes a self-contained model type detection JSON config to directory."""
    config_directory = directory / "config"
    config_directory.mkdir(parents=True, exist_ok=True)
    config_path = config_directory / "model_type_detection.json"
    detection_map = {
        "class_name_to_model_family": {
            "XGBClassifier": "XGBoost",
            "XGBRegressor": "XGBoost",
            "RandomForestClassifier": "RandomForest",
            "RandomForestRegressor": "RandomForest",
            "LGBMClassifier": "LightGBM",
            "LGBMRegressor": "LightGBM",
            "CatBoostClassifier": "CatBoost",
            "CatBoostRegressor": "CatBoost",
            "LogisticRegression": "LogisticRegression",
        }
    }
    config_path.write_text(json.dumps(detection_map), encoding="utf-8")
    return config_path


def _save_as_pickle(model_object: Any, directory: Path, filename: str = "model.pkl") -> Path:
    model_path = directory / filename
    with open(model_path, "wb") as file_handle:
        pickle.dump(model_object, file_handle)
    return model_path


def _save_as_joblib(model_object: Any, directory: Path, filename: str = "model.joblib") -> Path:
    model_path = directory / filename
    joblib.dump(model_object, model_path)
    return model_path


def _train_random_forest_on_array() -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=1, random_state=42)
    model.fit(_TRAINING_FEATURES_ARRAY, _TRAINING_LABELS_ARRAY)
    return model


def _train_random_forest_on_dataframe() -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=1, random_state=42)
    model.fit(_TRAINING_FEATURES_DATAFRAME, _TRAINING_LABELS_ARRAY)
    return model


def _train_logistic_regression() -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(_TRAINING_FEATURES_ARRAY, _TRAINING_LABELS_ARRAY)
    return model


def _train_xgboost() -> XGBClassifier:
    model = XGBClassifier(n_estimators=1, random_state=42, verbosity=0)
    model.fit(_TRAINING_FEATURES_ARRAY, _TRAINING_LABELS_ARRAY)
    return model


def _train_lightgbm() -> LGBMClassifier:
    model = LGBMClassifier(n_estimators=1, random_state=42, verbose=-1)
    model.fit(_TRAINING_FEATURES_ARRAY, _TRAINING_LABELS_ARRAY)
    return model


def _train_catboost() -> CatBoostClassifier:
    model = CatBoostClassifier(iterations=1, random_seed=42, verbose=False)
    model.fit(_TRAINING_FEATURES_ARRAY, _TRAINING_LABELS_ARRAY)
    return model


# ---------------------------------------------------------------------------
# Model family detection: one per Phase 1 supported type
# ---------------------------------------------------------------------------

def test_load_random_forest_pickle_detects_random_forest_family(tmp_path: Path) -> None:
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_random_forest_on_array(), tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.model_family == "RandomForest"
    assert loaded_model.detected_class_name == "RandomForestClassifier"


def test_load_logistic_regression_pickle_detects_logistic_regression_family(tmp_path: Path) -> None:
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_logistic_regression(), tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.model_family == "LogisticRegression"
    assert loaded_model.detected_class_name == "LogisticRegression"


def test_load_xgboost_pickle_detects_xgboost_family(tmp_path: Path) -> None:
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_xgboost(), tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.model_family == "XGBoost"
    assert loaded_model.detected_class_name == "XGBClassifier"


def test_load_lightgbm_pickle_detects_lightgbm_family(tmp_path: Path) -> None:
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_lightgbm(), tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.model_family == "LightGBM"
    assert loaded_model.detected_class_name == "LGBMClassifier"


def test_load_catboost_pickle_detects_catboost_family(tmp_path: Path) -> None:
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_catboost(), tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.model_family == "CatBoost"
    assert loaded_model.detected_class_name == "CatBoostClassifier"


# ---------------------------------------------------------------------------
# Serialization format detection
# ---------------------------------------------------------------------------

def test_load_pkl_extension_records_pickle_format(tmp_path: Path) -> None:
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_random_forest_on_array(), tmp_path, "model.pkl")
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.serialization_format == "pickle"


def test_load_joblib_file_records_joblib_format(tmp_path: Path) -> None:
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_joblib(_train_random_forest_on_array(), tmp_path, "model.joblib")
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.serialization_format == "joblib"


def test_joblib_extension_prefers_joblib_over_pickle_format(tmp_path: Path) -> None:
    """A file with .joblib extension should succeed via the joblib primary path."""
    config_path = _write_model_type_config(tmp_path)
    # Save with joblib under a .joblib extension — the loader's primary attempt should succeed.
    model_path = _save_as_joblib(_train_logistic_regression(), tmp_path, "model.joblib")
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.serialization_format == "joblib"
    assert loaded_model.model_family == "LogisticRegression"


def test_pickle_content_with_joblib_extension_loads_successfully(tmp_path: Path) -> None:
    """joblib.load() is a superset of pickle.load() and can transparently read pickle files.

    A pickle-serialized file with a .joblib extension is loaded via the joblib
    primary path (extension drives format ordering), which succeeds because joblib
    can deserialize standard pickle streams.
    """
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(
        _train_random_forest_on_array(), tmp_path, "model.joblib"
    )
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.model_family == "RandomForest"
    assert loaded_model.model_object is not None


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_missing_file_raises_model_load_error(tmp_path: Path) -> None:
    config_path = _write_model_type_config(tmp_path)
    nonexistent_path = tmp_path / "does_not_exist.pkl"
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    with pytest.raises(ModelLoadError, match="does not exist"):
        loader.load(nonexistent_path)


def test_corrupted_file_raises_model_load_error(tmp_path: Path) -> None:
    config_path = _write_model_type_config(tmp_path)
    corrupted_path = tmp_path / "corrupted.pkl"
    corrupted_path.write_bytes(b"this is not a valid pickle or joblib file")
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    with pytest.raises(ModelLoadError):
        loader.load(corrupted_path)


def test_unsupported_model_class_sets_family_to_none(tmp_path: Path) -> None:
    """An object whose class is not in the detection map gets model_family=None."""
    config_path = _write_model_type_config(tmp_path)
    # A plain dict is not in the supported class map.
    model_path = _save_as_pickle({"some": "data"}, tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.model_family is None
    assert loaded_model.detected_class_name == "dict"


# ---------------------------------------------------------------------------
# Feature name extraction
# ---------------------------------------------------------------------------

def test_feature_names_extracted_when_fitted_with_dataframe(tmp_path: Path) -> None:
    """feature_names_in_ is set by sklearn when the model is fitted with a DataFrame."""
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_random_forest_on_dataframe(), tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.feature_names_from_model is not None
    assert isinstance(loaded_model.feature_names_from_model, tuple)
    assert loaded_model.feature_names_from_model == ("feature_alpha", "feature_beta")


def test_feature_names_none_when_fitted_with_numpy_array(tmp_path: Path) -> None:
    """No feature_names_in_ is set when the model is fitted with a numpy array."""
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_random_forest_on_array(), tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.feature_names_from_model is None


def test_num_features_extracted_after_fit(tmp_path: Path) -> None:
    """n_features_in_ is set by sklearn >= 0.24 after any fit call."""
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_random_forest_on_array(), tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    # Training data has 2 features.
    assert loaded_model.num_features_from_model == 2


# ---------------------------------------------------------------------------
# Return value sanity checks
# ---------------------------------------------------------------------------

def test_detected_class_name_is_always_set(tmp_path: Path) -> None:
    """detected_class_name is always populated after a successful load."""
    config_path = _write_model_type_config(tmp_path)
    model_path = _save_as_pickle(_train_logistic_regression(), tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert loaded_model.detected_class_name != ""
    assert loaded_model.detected_class_name is not None


def test_model_object_is_the_fitted_estimator(tmp_path: Path) -> None:
    """model_object on LoadedModel is the deserialized sklearn estimator."""
    config_path = _write_model_type_config(tmp_path)
    original_model = _train_random_forest_on_array()
    model_path = _save_as_pickle(original_model, tmp_path)
    loader = ModelLoader(_make_test_logger(tmp_path), model_type_config_path=config_path)

    loaded_model = loader.load(model_path)

    assert isinstance(loaded_model.model_object, RandomForestClassifier)
    assert loaded_model.model_object.n_estimators == original_model.n_estimators
