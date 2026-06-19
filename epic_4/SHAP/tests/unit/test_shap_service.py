"""Unit tests for shap_explainability.explainers.shap_service."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from shap_explainability.errors import SHAPExecutionError
from shap_explainability.explainers.explainer_factory import BuiltExplainer
from shap_explainability.explainers.shap_service import (
    SHAPResult,
    SHAPService,
    _PREDICTION_TYPE_BINARY,
    _PREDICTION_TYPE_MULTICLASS,
    _PREDICTION_TYPE_REGRESSION,
)
from tests.fixtures.fixture_factory import FixtureFactory


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _write_full_config(directory: Path) -> Path:
    """Write model_type_detection.json with all sections needed by SHAPService."""
    config_directory = directory / "config"
    config_directory.mkdir(parents=True, exist_ok=True)
    config_path = config_directory / "model_type_detection.json"
    config_data = {
        "class_name_to_model_family": {},
        "supplied_name_to_family": {},
        "model_family_to_explainer": {},
        "class_name_to_prediction_category": {
            "XGBClassifier": "classification",
            "XGBRegressor": "regression",
            "RandomForestClassifier": "classification",
            "RandomForestRegressor": "regression",
            "LGBMClassifier": "classification",
            "LGBMRegressor": "regression",
            "CatBoostClassifier": "classification",
            "CatBoostRegressor": "regression",
            "LogisticRegression": "classification",
        },
        "tree_explainer_kwargs_by_family": {
            "XGBoost": {},
            "RandomForest": {},
            "LightGBM": {},
            "CatBoost": {"check_additivity": False},
        },
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    return config_path


def _make_service(tmp_path: Path) -> SHAPService:
    logger = FixtureFactory.make_execution_logger(tmp_path)
    config_path = _write_full_config(tmp_path)
    return SHAPService(execution_logger=logger, model_type_config_path=config_path)


def _make_feature_dataframe(num_rows: int = 5, num_features: int = 3) -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame(
        np.random.rand(num_rows, num_features),
        columns=[f"feat_{i}" for i in range(num_features)],
    )


def _make_mock_model(
    n_classes: int = 2,
    classes: Any = None,
) -> MagicMock:
    """Create a mock model with n_classes_ and optional classes_ attributes."""
    mock_model = MagicMock()
    mock_model.n_classes_ = n_classes
    if classes is not None:
        mock_model.classes_ = classes
    else:
        mock_model.classes_ = list(range(n_classes))
    return mock_model


def _make_built_explainer(
    shap_values_return_value: Any,
    model_family: str = "XGBoost",
    explainer_name: str = "TreeExplainer",
) -> BuiltExplainer:
    """Create a BuiltExplainer whose .shap_values() returns a controlled value."""
    mock_explainer_object = MagicMock()
    mock_explainer_object.shap_values.return_value = shap_values_return_value
    return BuiltExplainer(
        explainer_object=mock_explainer_object,
        explainer_name=explainer_name,
        model_family=model_family,
    )


def _make_feature_names(num_features: int = 3) -> tuple[str, ...]:
    return tuple(f"feat_{i}" for i in range(num_features))


# ---------------------------------------------------------------------------
# Prediction type detection
# ---------------------------------------------------------------------------

def test_detect_prediction_type_regression(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    mock_model = _make_mock_model()
    prediction_type = service.detect_prediction_type(mock_model, "XGBRegressor")
    assert prediction_type == _PREDICTION_TYPE_REGRESSION


def test_detect_prediction_type_binary_classification(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    mock_model = _make_mock_model(n_classes=2)
    prediction_type = service.detect_prediction_type(mock_model, "XGBClassifier")
    assert prediction_type == _PREDICTION_TYPE_BINARY


def test_detect_prediction_type_multiclass_classification(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    mock_model = _make_mock_model(n_classes=4)
    prediction_type = service.detect_prediction_type(mock_model, "RandomForestClassifier")
    assert prediction_type == _PREDICTION_TYPE_MULTICLASS


@pytest.mark.parametrize("class_name", [
    "XGBClassifier", "XGBRegressor", "RandomForestClassifier", "RandomForestRegressor",
    "LGBMClassifier", "LGBMRegressor", "CatBoostClassifier", "CatBoostRegressor",
    "LogisticRegression",
])
def test_detect_prediction_type_all_supported_class_names(
    tmp_path: Path, class_name: str
) -> None:
    """All supported class names must resolve without raising."""
    service = _make_service(tmp_path)
    mock_model = _make_mock_model(n_classes=2)
    result = service.detect_prediction_type(mock_model, class_name)
    assert result in (
        _PREDICTION_TYPE_BINARY,
        _PREDICTION_TYPE_MULTICLASS,
        _PREDICTION_TYPE_REGRESSION,
    )


def test_detect_prediction_type_unknown_class_raises(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    mock_model = _make_mock_model()
    with pytest.raises(SHAPExecutionError, match="not found"):
        service.detect_prediction_type(mock_model, "SomeUnknownModel")


def test_detect_prediction_type_uses_classes_attr_when_n_classes_absent(
    tmp_path: Path,
) -> None:
    """When n_classes_ is absent, len(model.classes_) must determine the type."""
    service = _make_service(tmp_path)
    mock_model = MagicMock(spec=[])
    mock_model.classes_ = ["cat", "dog", "fish"]

    prediction_type = service.detect_prediction_type(mock_model, "RandomForestClassifier")
    assert prediction_type == _PREDICTION_TYPE_MULTICLASS


def test_detect_prediction_type_falls_back_to_binary_when_no_class_attrs(
    tmp_path: Path,
) -> None:
    """Without n_classes_ or classes_, binary classification is assumed."""
    service = _make_service(tmp_path)
    mock_model = MagicMock(spec=[])

    prediction_type = service.detect_prediction_type(mock_model, "LogisticRegression")
    assert prediction_type == _PREDICTION_TYPE_BINARY


# ---------------------------------------------------------------------------
# SHAP value normalisation -- regression
# ---------------------------------------------------------------------------

def test_normalize_regression_2d_array_unchanged(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    raw = np.random.rand(5, 3)
    result = service._normalize_shap_values(raw, _PREDICTION_TYPE_REGRESSION)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 3)
    np.testing.assert_array_equal(result, raw)


def test_normalize_regression_bad_shape_raises(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    raw_3d = np.random.rand(5, 3, 2)
    with pytest.raises(SHAPExecutionError):
        service._normalize_shap_values(raw_3d, _PREDICTION_TYPE_REGRESSION)


# ---------------------------------------------------------------------------
# SHAP value normalisation -- binary classification
# ---------------------------------------------------------------------------

def test_normalize_binary_2d_ndarray_unchanged(tmp_path: Path) -> None:
    """XGBoost/LightGBM/CatBoost binary: 2D ndarray must be returned as-is."""
    service = _make_service(tmp_path)
    raw = np.random.rand(5, 3)
    result = service._normalize_shap_values(raw, _PREDICTION_TYPE_BINARY)
    assert result.shape == (5, 3)
    np.testing.assert_array_equal(result, raw)


def test_normalize_binary_list_takes_index_1(tmp_path: Path) -> None:
    """RandomForest binary: list[class_0, class_1] must return class_1 array."""
    service = _make_service(tmp_path)
    class_0_values = np.zeros((5, 3))
    class_1_values = np.ones((5, 3))
    result = service._normalize_shap_values(
        [class_0_values, class_1_values], _PREDICTION_TYPE_BINARY
    )
    assert result.shape == (5, 3)
    np.testing.assert_array_equal(result, class_1_values)


def test_normalize_binary_3d_ndarray_takes_positive_class(tmp_path: Path) -> None:
    """SHAP >= 0.40 returns (n_samples, n_features, n_classes) for binary tree models.
    Normalization must slice [:, :, 1] (positive class) to produce a canonical 2D array."""
    service = _make_service(tmp_path)
    raw_3d = np.random.rand(5, 3, 2)
    result = service._normalize_shap_values(raw_3d, _PREDICTION_TYPE_BINARY)
    assert result.shape == (5, 3)
    np.testing.assert_array_equal(result, raw_3d[:, :, 1])


# ---------------------------------------------------------------------------
# SHAP value normalisation -- multiclass
# ---------------------------------------------------------------------------

def test_normalize_multiclass_list_returned_unchanged(tmp_path: Path) -> None:
    """RandomForest/LightGBM multiclass: list of K arrays must be returned as-is."""
    service = _make_service(tmp_path)
    class_arrays = [np.random.rand(5, 3) for _ in range(3)]
    result = service._normalize_shap_values(class_arrays, _PREDICTION_TYPE_MULTICLASS)
    assert isinstance(result, list)
    assert len(result) == 3
    for class_index in range(3):
        np.testing.assert_array_equal(result[class_index], class_arrays[class_index])


def test_normalize_multiclass_3d_ndarray_sliced_to_list(tmp_path: Path) -> None:
    """XGBoost/CatBoost multiclass: 3D ndarray must be sliced into list of K arrays."""
    service = _make_service(tmp_path)
    raw_3d = np.random.rand(5, 3, 4)  # 4 classes
    result = service._normalize_shap_values(raw_3d, _PREDICTION_TYPE_MULTICLASS)
    assert isinstance(result, list)
    assert len(result) == 4
    for class_index in range(4):
        assert result[class_index].shape == (5, 3)
        np.testing.assert_array_equal(result[class_index], raw_3d[:, :, class_index])


def test_normalize_multiclass_bad_shape_raises(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    raw_1d = np.random.rand(15)
    with pytest.raises(SHAPExecutionError):
        service._normalize_shap_values(raw_1d, _PREDICTION_TYPE_MULTICLASS)


# ---------------------------------------------------------------------------
# Global importance computation
# ---------------------------------------------------------------------------

def test_compute_global_importance_binary_columns(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    shap_values = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])
    feature_names = ("feat_0", "feat_1", "feat_2")

    result_dataframe = service._compute_global_importance(
        shap_values, feature_names, _PREDICTION_TYPE_BINARY
    )

    assert list(result_dataframe.columns) == ["feature_name", "mean_absolute_shap_value"]
    assert len(result_dataframe) == 3


def test_compute_global_importance_values_are_mean_absolute(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    shap_values = np.array([[2.0, 4.0], [-2.0, -4.0]])
    feature_names = ("feat_0", "feat_1")

    result_dataframe = service._compute_global_importance(
        shap_values, feature_names, _PREDICTION_TYPE_BINARY
    )

    # mean(|[[2,-2],[-4,4]]|, axis=0) = [2.0, 4.0]
    sorted_values = result_dataframe["mean_absolute_shap_value"].tolist()
    assert sorted_values == sorted(sorted_values, reverse=True)
    assert abs(result_dataframe[result_dataframe["feature_name"] == "feat_0"][
        "mean_absolute_shap_value"
    ].iloc[0] - 2.0) < 1e-9
    assert abs(result_dataframe[result_dataframe["feature_name"] == "feat_1"][
        "mean_absolute_shap_value"
    ].iloc[0] - 4.0) < 1e-9


def test_compute_global_importance_sorted_descending(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    shap_values = np.array([[1.0, 5.0, 2.0]])
    feature_names = ("feat_0", "feat_1", "feat_2")

    result_dataframe = service._compute_global_importance(
        shap_values, feature_names, _PREDICTION_TYPE_REGRESSION
    )

    values = result_dataframe["mean_absolute_shap_value"].tolist()
    assert values == sorted(values, reverse=True)
    assert result_dataframe.iloc[0]["feature_name"] == "feat_1"


def test_compute_global_importance_multiclass_averages_across_classes(
    tmp_path: Path,
) -> None:
    """Multiclass global importance must average per-class mean-abs across all K classes."""
    service = _make_service(tmp_path)
    class_0_values = np.array([[4.0, 0.0], [4.0, 0.0]])  # mean_abs = [4, 0]
    class_1_values = np.array([[0.0, 2.0], [0.0, 2.0]])  # mean_abs = [0, 2]
    shap_values_list = [class_0_values, class_1_values]
    feature_names = ("feat_0", "feat_1")

    result_dataframe = service._compute_global_importance(
        shap_values_list, feature_names, _PREDICTION_TYPE_MULTICLASS
    )

    # Average: feat_0 = (4+0)/2 = 2.0, feat_1 = (0+2)/2 = 1.0
    feat_0_importance = result_dataframe[result_dataframe["feature_name"] == "feat_0"][
        "mean_absolute_shap_value"
    ].iloc[0]
    feat_1_importance = result_dataframe[result_dataframe["feature_name"] == "feat_1"][
        "mean_absolute_shap_value"
    ].iloc[0]
    assert abs(feat_0_importance - 2.0) < 1e-9
    assert abs(feat_1_importance - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Mapping DataFrame construction
# ---------------------------------------------------------------------------

def test_build_mapping_dataframe_binary_columns(tmp_path: Path) -> None:
    """Binary mapping must have columns: record_id, feature_name, feature_value, shap_value."""
    service = _make_service(tmp_path)
    feature_dataframe = _make_feature_dataframe(num_rows=3, num_features=2)
    shap_values = np.random.rand(3, 2)
    feature_names = ("feat_0", "feat_1")

    result_dataframe = service._build_mapping_dataframe(
        shap_values, feature_dataframe, feature_names,
        _PREDICTION_TYPE_BINARY, class_names=None
    )

    assert list(result_dataframe.columns) == [
        "record_id", "feature_name", "feature_value", "shap_value"
    ]
    assert len(result_dataframe) == 3 * 2


def test_build_mapping_dataframe_regression_columns(tmp_path: Path) -> None:
    """Regression mapping must have same schema as binary (no class_name column)."""
    service = _make_service(tmp_path)
    feature_dataframe = _make_feature_dataframe(num_rows=4, num_features=3)
    shap_values = np.random.rand(4, 3)

    result_dataframe = service._build_mapping_dataframe(
        shap_values, feature_dataframe, ("feat_0", "feat_1", "feat_2"),
        _PREDICTION_TYPE_REGRESSION, class_names=None
    )

    assert "class_name" not in result_dataframe.columns
    assert len(result_dataframe) == 4 * 3


def test_build_mapping_dataframe_multiclass_columns(tmp_path: Path) -> None:
    """Multiclass mapping must have columns: record_id, class_name, feature_name, feature_value, shap_value."""
    service = _make_service(tmp_path)
    feature_dataframe = _make_feature_dataframe(num_rows=2, num_features=2)
    shap_values_list = [np.random.rand(2, 2), np.random.rand(2, 2), np.random.rand(2, 2)]
    feature_names = ("feat_0", "feat_1")
    class_names = ("class_0", "class_1", "class_2")

    result_dataframe = service._build_mapping_dataframe(
        shap_values_list, feature_dataframe, feature_names,
        _PREDICTION_TYPE_MULTICLASS, class_names=class_names
    )

    assert list(result_dataframe.columns) == [
        "record_id", "class_name", "feature_name", "feature_value", "shap_value"
    ]
    # 2 samples x 2 features x 3 classes = 12 rows
    assert len(result_dataframe) == 2 * 2 * 3


def test_build_mapping_dataframe_multiclass_row_count(tmp_path: Path) -> None:
    """Multiclass row count must equal n_samples * n_features * n_classes."""
    service = _make_service(tmp_path)
    num_samples, num_features, num_classes = 10, 5, 4
    feature_dataframe = _make_feature_dataframe(num_rows=num_samples, num_features=num_features)
    feature_names = tuple(f"feat_{i}" for i in range(num_features))
    shap_values_list = [
        np.random.rand(num_samples, num_features) for _ in range(num_classes)
    ]
    class_names = tuple(f"class_{k}" for k in range(num_classes))

    result_dataframe = service._build_mapping_dataframe(
        shap_values_list, feature_dataframe, feature_names,
        _PREDICTION_TYPE_MULTICLASS, class_names=class_names
    )

    assert len(result_dataframe) == num_samples * num_features * num_classes


def test_build_mapping_dataframe_record_ids_are_sequential(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    feature_dataframe = _make_feature_dataframe(num_rows=3, num_features=2)
    shap_values = np.random.rand(3, 2)

    result_dataframe = service._build_mapping_dataframe(
        shap_values, feature_dataframe, ("feat_0", "feat_1"),
        _PREDICTION_TYPE_BINARY, class_names=None
    )

    assert set(result_dataframe["record_id"].unique()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Class name extraction
# ---------------------------------------------------------------------------

def test_get_class_names_returns_none_for_binary(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    mock_model = _make_mock_model(n_classes=2, classes=[0, 1])
    result = service._get_class_names(mock_model, _PREDICTION_TYPE_BINARY)
    assert result is None


def test_get_class_names_returns_none_for_regression(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    mock_model = _make_mock_model()
    result = service._get_class_names(mock_model, _PREDICTION_TYPE_REGRESSION)
    assert result is None


def test_get_class_names_integer_labels_formatted_as_class_n(tmp_path: Path) -> None:
    """Integer class labels must be formatted as 'class_0', 'class_1', ..."""
    service = _make_service(tmp_path)
    mock_model = _make_mock_model(n_classes=3, classes=[0, 1, 2])
    result = service._get_class_names(mock_model, _PREDICTION_TYPE_MULTICLASS)
    assert result == ("class_0", "class_1", "class_2")


def test_get_class_names_string_labels_used_directly(tmp_path: Path) -> None:
    """String class labels from model.classes_ must be used as-is."""
    service = _make_service(tmp_path)
    mock_model = MagicMock()
    mock_model.n_classes_ = 3
    mock_model.classes_ = ["cat", "dog", "fish"]
    result = service._get_class_names(mock_model, _PREDICTION_TYPE_MULTICLASS)
    assert result == ("cat", "dog", "fish")


def test_get_class_names_returns_none_when_classes_attr_absent(tmp_path: Path) -> None:
    """When model.classes_ is absent for multiclass, return None (no class names)."""
    service = _make_service(tmp_path)
    mock_model = MagicMock(spec=["n_classes_"])
    mock_model.n_classes_ = 3
    result = service._get_class_names(mock_model, _PREDICTION_TYPE_MULTICLASS)
    assert result is None


# ---------------------------------------------------------------------------
# SHAPResult dataclass
# ---------------------------------------------------------------------------

def test_shap_result_is_frozen() -> None:
    """SHAPResult must be immutable (frozen dataclass)."""
    result = SHAPResult(
        prediction_type="binary_classification",
        shap_values_array=np.zeros((2, 2)),
        feature_names=("f0", "f1"),
        class_names=None,
        global_importance_dataframe=pd.DataFrame(),
        mapping_dataframe=pd.DataFrame(),
    )
    with pytest.raises(Exception):
        result.prediction_type = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Full compute() integration
# ---------------------------------------------------------------------------

def test_compute_returns_shap_result_for_binary(tmp_path: Path) -> None:
    """compute() must return a SHAPResult with correct prediction_type for binary."""
    service = _make_service(tmp_path)
    num_samples, num_features = 5, 3
    feature_dataframe = _make_feature_dataframe(num_rows=num_samples, num_features=num_features)
    feature_names = _make_feature_names(num_features)
    shap_values_2d = np.random.rand(num_samples, num_features)

    built_explainer = _make_built_explainer(shap_values_2d, model_family="XGBoost")
    mock_model = _make_mock_model(n_classes=2)
    session_context = FixtureFactory.make_session_context()

    result = service.compute(
        built_explainer=built_explainer,
        feature_dataframe=feature_dataframe,
        feature_names=feature_names,
        model_object=mock_model,
        detected_class_name="XGBClassifier",
        session_context=session_context,
    )

    assert isinstance(result, SHAPResult)
    assert result.prediction_type == _PREDICTION_TYPE_BINARY
    assert result.class_names is None
    assert result.feature_names == feature_names
    assert result.shap_values_array.shape == (num_samples, num_features)


def test_compute_returns_shap_result_for_regression(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    num_samples, num_features = 4, 2
    feature_dataframe = _make_feature_dataframe(num_rows=num_samples, num_features=num_features)
    feature_names = _make_feature_names(num_features)
    shap_values_2d = np.random.rand(num_samples, num_features)

    built_explainer = _make_built_explainer(shap_values_2d, model_family="XGBoost")
    mock_model = MagicMock(spec=[])
    session_context = FixtureFactory.make_session_context()

    result = service.compute(
        built_explainer=built_explainer,
        feature_dataframe=feature_dataframe,
        feature_names=feature_names,
        model_object=mock_model,
        detected_class_name="XGBRegressor",
        session_context=session_context,
    )

    assert result.prediction_type == _PREDICTION_TYPE_REGRESSION
    assert result.class_names is None


def test_compute_returns_shap_result_for_multiclass(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    num_samples, num_features, num_classes = 5, 3, 4
    feature_dataframe = _make_feature_dataframe(num_rows=num_samples, num_features=num_features)
    feature_names = _make_feature_names(num_features)
    shap_values_list = [
        np.random.rand(num_samples, num_features) for _ in range(num_classes)
    ]

    mock_model = _make_mock_model(n_classes=num_classes, classes=[0, 1, 2, 3])
    built_explainer = _make_built_explainer(shap_values_list, model_family="RandomForest")
    session_context = FixtureFactory.make_session_context()

    result = service.compute(
        built_explainer=built_explainer,
        feature_dataframe=feature_dataframe,
        feature_names=feature_names,
        model_object=mock_model,
        detected_class_name="RandomForestClassifier",
        session_context=session_context,
    )

    assert result.prediction_type == _PREDICTION_TYPE_MULTICLASS
    assert result.class_names == ("class_0", "class_1", "class_2", "class_3")
    assert isinstance(result.shap_values_array, list)
    assert len(result.shap_values_array) == num_classes


def test_compute_writes_to_session_context(tmp_path: Path) -> None:
    """compute() must populate shap_values, global_feature_importance, feature_shap_mapping."""
    service = _make_service(tmp_path)
    feature_dataframe = _make_feature_dataframe(num_rows=4, num_features=2)
    feature_names = _make_feature_names(2)
    shap_values_2d = np.random.rand(4, 2)
    built_explainer = _make_built_explainer(shap_values_2d, model_family="XGBoost")
    mock_model = _make_mock_model(n_classes=2)
    session_context = FixtureFactory.make_session_context()

    service.compute(
        built_explainer=built_explainer,
        feature_dataframe=feature_dataframe,
        feature_names=feature_names,
        model_object=mock_model,
        detected_class_name="XGBClassifier",
        session_context=session_context,
    )

    assert session_context.shap_values is not None
    assert session_context.global_feature_importance is not None
    assert session_context.feature_shap_mapping is not None


def test_compute_shap_values_failure_raises_shap_execution_error(tmp_path: Path) -> None:
    """If the explainer raises, SHAPExecutionError must propagate from compute()."""
    service = _make_service(tmp_path)
    feature_dataframe = _make_feature_dataframe()
    feature_names = _make_feature_names()

    mock_explainer_object = MagicMock()
    mock_explainer_object.shap_values.side_effect = RuntimeError("explainer failure")
    built_explainer = BuiltExplainer(
        explainer_object=mock_explainer_object,
        explainer_name="TreeExplainer",
        model_family="XGBoost",
    )

    with pytest.raises(SHAPExecutionError, match="shap_values"):
        service.compute(
            built_explainer=built_explainer,
            feature_dataframe=feature_dataframe,
            feature_names=feature_names,
            model_object=_make_mock_model(n_classes=2),
            detected_class_name="XGBClassifier",
            session_context=FixtureFactory.make_session_context(),
        )


def test_compute_catboost_passes_check_additivity_false(tmp_path: Path) -> None:
    """CatBoost family must pass check_additivity=False to shap_values()."""
    service = _make_service(tmp_path)
    feature_dataframe = _make_feature_dataframe(num_rows=3, num_features=2)
    feature_names = _make_feature_names(2)
    shap_values_2d = np.random.rand(3, 2)

    mock_explainer_object = MagicMock()
    mock_explainer_object.shap_values.return_value = shap_values_2d
    catboost_explainer = BuiltExplainer(
        explainer_object=mock_explainer_object,
        explainer_name="TreeExplainer",
        model_family="CatBoost",
    )

    service.compute(
        built_explainer=catboost_explainer,
        feature_dataframe=feature_dataframe,
        feature_names=feature_names,
        model_object=_make_mock_model(n_classes=2),
        detected_class_name="CatBoostClassifier",
        session_context=FixtureFactory.make_session_context(),
    )

    mock_explainer_object.shap_values.assert_called_once_with(
        feature_dataframe, check_additivity=False
    )


def test_compute_xgboost_passes_no_extra_kwargs(tmp_path: Path) -> None:
    """XGBoost family must call shap_values() with no extra keyword arguments."""
    service = _make_service(tmp_path)
    feature_dataframe = _make_feature_dataframe(num_rows=3, num_features=2)
    feature_names = _make_feature_names(2)
    shap_values_2d = np.random.rand(3, 2)

    mock_explainer_object = MagicMock()
    mock_explainer_object.shap_values.return_value = shap_values_2d
    xgboost_explainer = BuiltExplainer(
        explainer_object=mock_explainer_object,
        explainer_name="TreeExplainer",
        model_family="XGBoost",
    )

    service.compute(
        built_explainer=xgboost_explainer,
        feature_dataframe=feature_dataframe,
        feature_names=feature_names,
        model_object=_make_mock_model(n_classes=2),
        detected_class_name="XGBClassifier",
        session_context=FixtureFactory.make_session_context(),
    )

    mock_explainer_object.shap_values.assert_called_once_with(feature_dataframe)


def test_compute_global_importance_dataframe_has_correct_columns(tmp_path: Path) -> None:
    service = _make_service(tmp_path)
    feature_dataframe = _make_feature_dataframe(num_rows=5, num_features=3)
    feature_names = _make_feature_names(3)
    shap_values_2d = np.random.rand(5, 3)
    built_explainer = _make_built_explainer(shap_values_2d)
    mock_model = _make_mock_model(n_classes=2)

    result = service.compute(
        built_explainer=built_explainer,
        feature_dataframe=feature_dataframe,
        feature_names=feature_names,
        model_object=mock_model,
        detected_class_name="XGBClassifier",
        session_context=FixtureFactory.make_session_context(),
    )

    assert list(result.global_importance_dataframe.columns) == [
        "feature_name", "mean_absolute_shap_value"
    ]
    assert len(result.global_importance_dataframe) == 3
