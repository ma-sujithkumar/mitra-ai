"""Compute SHAP values, aggregate global importance, and build the mapping table.

Implements spec.md Sections 15 (steps 7-9), 17.1, 17.2, and architecture.md
steps 8-9: SHAPService receives a BuiltExplainer from ExplainerFactory, runs
the SHAP computation, normalises the raw output to a canonical shape, detects
the prediction type, computes mean-absolute global importance, and builds the
long-form per-record/per-feature mapping DataFrame.

Prediction type detection is config-driven via model_type_detection.json
(class_name_to_prediction_category section) so no if-else ladders are used
for model class name dispatch (CLAUDE.md rule 4).
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from shap_explainability.errors import SHAPExecutionError
from shap_explainability.explainers.explainer_factory import BuiltExplainer
from shap_explainability.models.shap_result import SHAPResult
from shap_explainability.session_context import SessionContext
from shap_explainability.utils.logger import ExecutionLogger

_MODEL_TYPE_CONFIG_FILENAME: str = "model_type_detection.json"
_CLASS_TO_CATEGORY_KEY: str = "class_name_to_prediction_category"
_TREE_KWARGS_KEY: str = "tree_explainer_kwargs_by_family"

_PREDICTION_TYPE_BINARY: str = "binary_classification"
_PREDICTION_TYPE_MULTICLASS: str = "multiclass_classification"
_PREDICTION_TYPE_REGRESSION: str = "regression"
_CATEGORY_CLASSIFICATION: str = "classification"
_CATEGORY_REGRESSION: str = "regression"

_COL_RECORD_ID: str = "record_id"
_COL_CLASS_NAME: str = "class_name"
_COL_FEATURE_NAME: str = "feature_name"
_COL_FEATURE_VALUE: str = "feature_value"
_COL_SHAP_VALUE: str = "shap_value"
_COL_MEAN_ABS_SHAP: str = "mean_absolute_shap_value"


class SHAPService:
    """Runs SHAP computation and produces structured output DataFrames.

    Responsibilities (spec.md Sec 15 steps 7-9, Sec 17):
      - Detect prediction type from the model class name and introspection
        (binary_classification, multiclass_classification, regression).
      - Run the SHAP explainer via .shap_values() with per-family kwargs.
      - Normalise raw SHAP output to a canonical shape for all libraries.
      - Compute per-feature global importance as mean(|shap_values|, axis=0).
      - Build the long-form feature-SHAP mapping DataFrame per Sec 17.2.
      - Write shap_values, global_feature_importance, feature_shap_mapping to
        SessionContext for downstream exporters and visualisers.
    """

    def __init__(
        self,
        execution_logger: ExecutionLogger,
        model_type_config_path: Optional[Path] = None,
    ) -> None:
        """Initializes the SHAPService.

        Args:
            execution_logger: Session-scoped logger for recording Sec 19 events.
            model_type_config_path: Path to model_type_detection.json. Defaults
                to epic_4_shap/config/model_type_detection.json.

        Raises:
            SHAPExecutionError: If the config file cannot be read or parsed, or
                if the required sections are missing.
        """
        self._execution_logger: ExecutionLogger = execution_logger

        resolved_config_path: Path = (
            model_type_config_path
            if model_type_config_path is not None
            else self._default_model_type_config_path()
        )
        full_config: dict = self._load_full_config(resolved_config_path)
        self._class_to_category: dict[str, str] = full_config.get(
            _CLASS_TO_CATEGORY_KEY, {}
        )
        self._tree_kwargs_by_family: dict[str, dict] = full_config.get(
            _TREE_KWARGS_KEY, {}
        )

    def compute(
        self,
        built_explainer: BuiltExplainer,
        feature_dataframe: pd.DataFrame,
        feature_names: tuple[str, ...],
        model_object: Any,
        detected_class_name: str,
        session_context: SessionContext,
    ) -> SHAPResult:
        """Run full SHAP computation and return structured result.

        Args:
            built_explainer: Constructed SHAP explainer from ExplainerFactory.
            feature_dataframe: Cleaned, target-excluded feature DataFrame from
                SchemaValidationResult. Shape: (n_samples, n_features).
            feature_names: Ordered feature name tuple from SessionContext
                (written by SchemaValidator, Sec 13).
            model_object: Fitted model object needed for prediction type
                detection via n_classes_ / classes_ introspection.
            detected_class_name: Python class name from SessionContext
                (e.g. "XGBClassifier"); used for config-driven category lookup.
            session_context: Mutable pipeline state. shap_values,
                global_feature_importance, and feature_shap_mapping are written
                here for downstream phases.

        Returns:
            SHAPResult with prediction_type, normalised shap_values_array,
            feature_names, class_names, global_importance_dataframe, and
            mapping_dataframe.

        Raises:
            SHAPExecutionError: If prediction type detection, shap_values()
                call, or shape normalisation fails.
        """
        self._execution_logger.log_shap_generation(
            f"Starting SHAP computation: model={detected_class_name}, "
            f"explainer={built_explainer.explainer_name}, "
            f"samples={len(feature_dataframe)}, features={len(feature_names)}"
        )

        prediction_type: str = self.detect_prediction_type(model_object, detected_class_name)
        self._execution_logger.log_shap_generation(
            f"Detected prediction type: {prediction_type}"
        )

        shap_kwargs: dict = self._tree_kwargs_by_family.get(
            built_explainer.model_family, {}
        )
        raw_shap_output: Any = self._run_explainer(
            built_explainer.explainer_object, feature_dataframe, shap_kwargs
        )
        self._execution_logger.log_shap_generation(
            f"Raw SHAP values computed. Type: {type(raw_shap_output).__name__}"
        )

        shap_values_array: Any = self._normalize_shap_values(
            raw_shap_output, prediction_type
        )
        self._execution_logger.log_shap_generation(
            "SHAP values normalised to canonical form."
        )

        class_names: Optional[tuple[str, ...]] = self._get_class_names(
            model_object, prediction_type
        )

        global_importance_dataframe: pd.DataFrame = self._compute_global_importance(
            shap_values_array, feature_names, prediction_type
        )

        mapping_dataframe: pd.DataFrame = self._build_mapping_dataframe(
            shap_values_array, feature_dataframe, feature_names,
            prediction_type, class_names
        )

        # Write SHAP outputs to SessionContext for downstream exporters (Phase 6)
        # and metadata exporter (Phase 7).
        session_context.shap_values = shap_values_array
        session_context.global_feature_importance = global_importance_dataframe
        session_context.feature_shap_mapping = mapping_dataframe

        self._execution_logger.log_shap_generation(
            f"SHAP computation complete: {len(feature_names)} features, "
            f"{len(feature_dataframe)} samples, prediction_type={prediction_type}"
        )

        return SHAPResult(
            prediction_type=prediction_type,
            shap_values_array=shap_values_array,
            feature_names=feature_names,
            class_names=class_names,
            global_importance_dataframe=global_importance_dataframe,
            mapping_dataframe=mapping_dataframe,
        )

    def detect_prediction_type(
        self, model_object: Any, detected_class_name: str
    ) -> str:
        """Detect prediction type from class name config lookup and model introspection.

        Config-driven: class_name_to_prediction_category maps each supported
        class name to "classification" or "regression". For classification,
        n_classes_ / classes_ introspection resolves binary vs multiclass.

        Args:
            model_object: Fitted model object for n_classes_ / classes_ check.
            detected_class_name: Python class name from SessionContext.

        Returns:
            One of "binary_classification", "multiclass_classification",
            or "regression".

        Raises:
            SHAPExecutionError: If detected_class_name is not in the config map.
        """
        prediction_category: Optional[str] = self._class_to_category.get(
            detected_class_name
        )
        if prediction_category is None:
            raise SHAPExecutionError(
                f"Class '{detected_class_name}' not found in "
                f"'{_CLASS_TO_CATEGORY_KEY}' config section. "
                "Add an entry for this class before running the pipeline."
            )

        if prediction_category == _CATEGORY_REGRESSION:
            return _PREDICTION_TYPE_REGRESSION

        # Resolve binary vs multiclass using model introspection.
        num_classes: int = self._resolve_num_classes(model_object)
        if num_classes == 2:
            return _PREDICTION_TYPE_BINARY
        return _PREDICTION_TYPE_MULTICLASS

    def _resolve_num_classes(self, model_object: Any) -> int:
        """Introspect the model to determine number of output classes.

        Priority order:
          1. model.n_classes_ (sklearn standard attribute)
          2. len(model.classes_) (available when fitted on labelled data)
          3. Fall back to 2 (binary assumption) and log a warning.

        Args:
            model_object: Fitted classifier object.

        Returns:
            Number of classes as an integer (>= 2).
        """
        n_classes_attr = getattr(model_object, "n_classes_", None)
        if n_classes_attr is not None:
            return int(n_classes_attr)

        classes_attr = getattr(model_object, "classes_", None)
        if classes_attr is not None:
            return len(classes_attr)

        self._execution_logger.log_shap_generation(
            "Could not determine n_classes from model attributes. "
            "Defaulting to binary classification assumption.",
            logging.WARNING,
        )
        return 2

    def _run_explainer(
        self,
        explainer_object: Any,
        feature_dataframe: pd.DataFrame,
        shap_kwargs: dict,
    ) -> Any:
        """Call .shap_values() on the explainer with per-family keyword arguments.

        Args:
            explainer_object: Constructed SHAP explainer (TreeExplainer or
                LinearExplainer).
            feature_dataframe: Feature DataFrame for SHAP computation.
            shap_kwargs: Per-family keyword arguments from
                tree_explainer_kwargs_by_family config (e.g.
                {"check_additivity": False} for CatBoost).

        Returns:
            Raw SHAP output from .shap_values() call (shape varies by library
            and prediction type -- normalised by _normalize_shap_values).

        Raises:
            SHAPExecutionError: If .shap_values() raises.
        """
        try:
            return explainer_object.shap_values(feature_dataframe, **shap_kwargs)
        except Exception as shap_error:
            raise SHAPExecutionError(
                f"shap_values() computation failed: {shap_error}"
            ) from shap_error

    def _normalize_shap_values(
        self, raw_shap_output: Any, prediction_type: str
    ) -> Any:
        """Normalise raw SHAP output to a canonical internal shape.

        Different SHAP library versions and model families return SHAP values
        in different shapes. This method is the single point that handles all
        shape variants and produces a consistent form for downstream consumers.

        Canonical shapes:
          Binary/Regression: ndarray of shape (n_samples, n_features)
          Multiclass:        list of K ndarrays each of shape (n_samples, n_features)

        Binary normalisation rules:
          - list input (RandomForest): take index [1] (positive class values)
          - 2D ndarray: use as-is (XGBoost, LightGBM, CatBoost, LinearExplainer)

        Multiclass normalisation rules:
          - list input (RandomForest, LightGBM, older XGBoost): use as-is
          - 3D ndarray (newer XGBoost, CatBoost): slice into list of K arrays

        Regression: always a 2D ndarray -- use as-is.

        Args:
            raw_shap_output: Direct return value from .shap_values().
            prediction_type: One of the _PREDICTION_TYPE_* constants.

        Returns:
            Normalised SHAP values in canonical form.

        Raises:
            SHAPExecutionError: If the shape cannot be normalised to the
                expected canonical form.
        """
        if prediction_type == _PREDICTION_TYPE_REGRESSION:
            return self._normalize_regression_shap(raw_shap_output)
        if prediction_type == _PREDICTION_TYPE_BINARY:
            return self._normalize_binary_shap(raw_shap_output)
        if prediction_type == _PREDICTION_TYPE_MULTICLASS:
            return self._normalize_multiclass_shap(raw_shap_output)

        raise SHAPExecutionError(
            f"Unknown prediction_type '{prediction_type}' in _normalize_shap_values."
        )

    @staticmethod
    def _normalize_regression_shap(raw_shap_output: Any) -> np.ndarray:
        """Normalise regression SHAP output to 2D ndarray (n_samples, n_features)."""
        raw_array = np.asarray(raw_shap_output)
        if raw_array.ndim != 2:
            raise SHAPExecutionError(
                f"Regression SHAP values expected 2D array, got shape {raw_array.shape}."
            )
        return raw_array

    @staticmethod
    def _normalize_binary_shap(raw_shap_output: Any) -> np.ndarray:
        """Normalise binary classification SHAP output to 2D ndarray.

        RandomForest returns a list [class_0_values, class_1_values].
        XGBoost, LightGBM, CatBoost, LinearExplainer return a 2D ndarray
        directly (positive class contributions).
        """
        if isinstance(raw_shap_output, list):
            # Take index [1] = contributions toward the positive class.
            positive_class_values = np.asarray(raw_shap_output[1])
            if positive_class_values.ndim != 2:
                raise SHAPExecutionError(
                    f"Binary SHAP list[1] expected 2D array, got shape "
                    f"{positive_class_values.shape}."
                )
            return positive_class_values

        raw_array = np.asarray(raw_shap_output)
        if raw_array.ndim != 2:
            raise SHAPExecutionError(
                f"Binary SHAP values expected 2D array or list, got shape "
                f"{raw_array.shape}."
            )
        return raw_array

    @staticmethod
    def _normalize_multiclass_shap(
        raw_shap_output: Any,
    ) -> list[np.ndarray]:
        """Normalise multiclass SHAP output to list of K 2D ndarrays.

        RandomForest, LightGBM, older XGBoost return a list of K arrays.
        Newer XGBoost and CatBoost return a 3D ndarray (n_samples, n_features, K).
        """
        if isinstance(raw_shap_output, list):
            normalised_list: list[np.ndarray] = []
            for class_index, class_values in enumerate(raw_shap_output):
                class_array = np.asarray(class_values)
                if class_array.ndim != 2:
                    raise SHAPExecutionError(
                        f"Multiclass SHAP list[{class_index}] expected 2D array, "
                        f"got shape {class_array.shape}."
                    )
                normalised_list.append(class_array)
            return normalised_list

        raw_array = np.asarray(raw_shap_output)
        if raw_array.ndim == 3:
            # Shape: (n_samples, n_features, n_classes) -- slice into list.
            num_classes: int = raw_array.shape[2]
            return [raw_array[:, :, class_index] for class_index in range(num_classes)]

        raise SHAPExecutionError(
            f"Multiclass SHAP values expected list or 3D ndarray, got shape "
            f"{raw_array.shape}."
        )

    def _compute_global_importance(
        self,
        shap_values_array: Any,
        feature_names: tuple[str, ...],
        prediction_type: str,
    ) -> pd.DataFrame:
        """Compute per-feature mean absolute SHAP importance (Sec 17.1).

        Binary/Regression: mean(|shap_values|, axis=0) -> (n_features,) vector.
        Multiclass: mean(|values|, axis=0) per class, then mean across all K
            classes -> (n_features,) vector representing model-wide importance.

        Args:
            shap_values_array: Normalised SHAP values (canonical form).
            feature_names: Ordered feature name tuple (length n_features).
            prediction_type: One of the _PREDICTION_TYPE_* constants.

        Returns:
            DataFrame with columns feature_name and mean_absolute_shap_value,
            sorted descending by mean_absolute_shap_value.
        """
        if prediction_type == _PREDICTION_TYPE_MULTICLASS:
            per_class_importance: list[np.ndarray] = [
                np.mean(np.abs(class_values), axis=0)
                for class_values in shap_values_array
            ]
            importance_values: np.ndarray = np.mean(
                np.stack(per_class_importance), axis=0
            )
        else:
            importance_values = np.mean(np.abs(shap_values_array), axis=0)

        global_importance_dataframe: pd.DataFrame = pd.DataFrame(
            {
                _COL_FEATURE_NAME: list(feature_names),
                _COL_MEAN_ABS_SHAP: importance_values.tolist(),
            }
        )
        return global_importance_dataframe.sort_values(
            _COL_MEAN_ABS_SHAP, ascending=False
        ).reset_index(drop=True)

    def _build_mapping_dataframe(
        self,
        shap_values_array: Any,
        feature_dataframe: pd.DataFrame,
        feature_names: tuple[str, ...],
        prediction_type: str,
        class_names: Optional[tuple[str, ...]],
    ) -> pd.DataFrame:
        """Build the long-form per-record/per-feature SHAP mapping table (Sec 17.2).

        Binary/Regression schema: record_id, feature_name, feature_value, shap_value
        Multiclass schema: record_id, class_name, feature_name, feature_value, shap_value

        Args:
            shap_values_array: Normalised SHAP values (canonical form).
            feature_dataframe: Cleaned feature DataFrame for feature_value column.
            feature_names: Ordered feature name tuple.
            prediction_type: One of the _PREDICTION_TYPE_* constants.
            class_names: Class labels for multiclass; None for binary/regression.

        Returns:
            Long-form mapping DataFrame per Sec 17.2 column contract.
        """
        if prediction_type == _PREDICTION_TYPE_MULTICLASS:
            return self._build_multiclass_mapping(
                shap_values_array, feature_dataframe, feature_names, class_names
            )
        return self._build_flat_mapping(
            shap_values_array, feature_dataframe, feature_names
        )

    @staticmethod
    def _build_flat_mapping(
        shap_values_array: np.ndarray,
        feature_dataframe: pd.DataFrame,
        feature_names: tuple[str, ...],
    ) -> pd.DataFrame:
        """Build binary/regression mapping: record_id, feature_name, feature_value, shap_value."""
        num_samples: int = len(feature_dataframe)
        num_features: int = len(feature_names)

        record_ids: np.ndarray = np.repeat(np.arange(num_samples), num_features)
        feature_names_repeated: np.ndarray = np.tile(
            np.array(feature_names), num_samples
        )
        feature_values: np.ndarray = feature_dataframe.values.flatten()
        shap_values_flat: np.ndarray = np.asarray(shap_values_array).flatten()

        return pd.DataFrame(
            {
                _COL_RECORD_ID: record_ids,
                _COL_FEATURE_NAME: feature_names_repeated,
                _COL_FEATURE_VALUE: feature_values,
                _COL_SHAP_VALUE: shap_values_flat,
            }
        )

    @staticmethod
    def _build_multiclass_mapping(
        shap_values_array: list[np.ndarray],
        feature_dataframe: pd.DataFrame,
        feature_names: tuple[str, ...],
        class_names: Optional[tuple[str, ...]],
    ) -> pd.DataFrame:
        """Build multiclass mapping: record_id, class_name, feature_name, feature_value, shap_value."""
        num_samples: int = len(feature_dataframe)
        num_features: int = len(feature_names)
        feature_values_matrix: np.ndarray = feature_dataframe.values

        class_frame_list: list[pd.DataFrame] = []
        for class_index, class_shap_values in enumerate(shap_values_array):
            resolved_class_name: str = (
                class_names[class_index]
                if class_names is not None
                else f"class_{class_index}"
            )
            record_ids: np.ndarray = np.repeat(np.arange(num_samples), num_features)
            class_labels: np.ndarray = np.full(
                num_samples * num_features, resolved_class_name
            )
            feature_names_repeated: np.ndarray = np.tile(
                np.array(feature_names), num_samples
            )
            feature_values: np.ndarray = feature_values_matrix.flatten()
            shap_values_flat: np.ndarray = np.asarray(class_shap_values).flatten()

            class_frame_list.append(
                pd.DataFrame(
                    {
                        _COL_RECORD_ID: record_ids,
                        _COL_CLASS_NAME: class_labels,
                        _COL_FEATURE_NAME: feature_names_repeated,
                        _COL_FEATURE_VALUE: feature_values,
                        _COL_SHAP_VALUE: shap_values_flat,
                    }
                )
            )

        return pd.concat(class_frame_list, ignore_index=True)

    def _get_class_names(
        self, model_object: Any, prediction_type: str
    ) -> Optional[tuple[str, ...]]:
        """Extract class label strings from the model for multiclass prediction.

        For binary and regression, returns None (no class dimension in output).
        For multiclass, extracts model.classes_ and formats as strings.
        Integer class labels are converted to "class_0", "class_1", ... format.
        String labels from model.classes_ are used directly.

        Args:
            model_object: Fitted model object with optional classes_ attribute.
            prediction_type: One of the _PREDICTION_TYPE_* constants.

        Returns:
            Tuple of class name strings for multiclass, or None.
        """
        if prediction_type != _PREDICTION_TYPE_MULTICLASS:
            return None

        classes_attr = getattr(model_object, "classes_", None)
        if classes_attr is None:
            self._execution_logger.log_shap_generation(
                "model.classes_ not available; class names will default to "
                "positional class_{i} strings in mapping DataFrame.",
                logging.WARNING,
            )
            return None

        resolved_class_names: list[str] = []
        for class_index, class_label in enumerate(classes_attr):
            if isinstance(class_label, str):
                resolved_class_names.append(class_label)
            else:
                # Integers, numpy int types, etc. -> "class_{i}" convention (Spec Sec 17.2).
                resolved_class_names.append(f"class_{class_index}")
        return tuple(resolved_class_names)

    @staticmethod
    def _load_full_config(config_path: Path) -> dict:
        """Load the full model_type_detection.json config as a dictionary.

        Args:
            config_path: Path to model_type_detection.json.

        Returns:
            Parsed JSON dictionary.

        Raises:
            SHAPExecutionError: If the config file is missing or not valid JSON.
        """
        if not config_path.is_file():
            raise SHAPExecutionError(
                f"Model type detection config not found: {config_path}. "
                "Cannot load prediction category or tree explainer kwargs mappings."
            )
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)

    @staticmethod
    def _default_model_type_config_path() -> Path:
        """Resolves the default config path relative to this module's location.

        shap_service.py is at: epic_4_shap/src/shap_explainability/explainers/
        config/ is at:         epic_4_shap/config/
        """
        return (
            Path(__file__).resolve().parent.parent.parent.parent
            / "config"
            / _MODEL_TYPE_CONFIG_FILENAME
        )
