"""Load a trained model artifact from disk and detect its concrete type.

Implements spec.md Section 7 steps 1-3 and Sec A11 (pickle/joblib support).
ModelLoader is deliberately ignorant of model_name validation (Sec 8 Rules 1-4);
that responsibility belongs exclusively to ModelValidator.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib

from backend.agents.evaluation.shap.errors import ModelLoadError
from backend.agents.evaluation.shap.utils.logger import ExecutionLogger

_FORMAT_PICKLE: str = "pickle"
_FORMAT_JOBLIB: str = "joblib"
_JOBLIB_FILE_EXTENSION: str = ".joblib"
_MODEL_TYPE_CONFIG_FILENAME: str = "model_type_detection.json"
_CLASS_NAME_TO_FAMILY_KEY: str = "class_name_to_model_family"


@dataclass(frozen=True)
class LoadedModel:
    """Container for a successfully loaded and introspected model artifact.

    Attributes:
        model_object: Deserialized estimator (e.g. an XGBClassifier instance).
        detected_class_name: Python class name of the loaded object as returned by
            type().__name__ (e.g. "XGBClassifier"). Always set after a successful
            load, regardless of whether the model is supported.
        model_family: Supported model family resolved from the detection config
            (e.g. "XGBoost"). None when the class name is not in the supported map;
            ModelValidator applies Sec 8 Rule 4 for that case.
        serialization_format: Which deserialization path succeeded ("pickle" or
            "joblib").
        feature_names_from_model: Feature names extracted from the model's internal
            metadata, if the model stores them (e.g. when fitted on a pandas
            DataFrame). None when unavailable. The engineered dataset is the
            authoritative source (Sec 10); these names are used for cross-validation
            only.
        num_features_from_model: Feature count from model metadata, if available.
    """

    model_object: Any
    detected_class_name: str
    model_family: Optional[str]
    serialization_format: str
    feature_names_from_model: Optional[tuple[str, ...]]
    num_features_from_model: Optional[int]


class ModelLoader:
    """Loads a trained model artifact and detects its concrete model type.

    Responsibilities (spec.md Sec 7, architecture.md Section 3 and 4):
      - Validate that the artifact file exists (Sec 9, validation layer 2).
      - Deserialize via pickle or joblib using an extension-driven primary/fallback
        strategy (Sec A11).
      - Detect the concrete Python class name of the deserialized object.
      - Map the class name to a supported model family via the JSON config-driven
        lookup table (not if-else conditionals, per CLAUDE.md rule 4).
      - Defensively extract any feature metadata the model exposes for downstream
        cross-validation by SchemaValidator.
    """

    def __init__(
        self,
        execution_logger: ExecutionLogger,
        model_type_config_path: Optional[Path] = None,
    ) -> None:
        """Initializes the ModelLoader.

        Args:
            execution_logger: Session-scoped logger for recording Sec 19 events.
            model_type_config_path: Path to the model type detection JSON config.
                Defaults to epic_4_shap/config/model_type_detection.json resolved
                relative to this module's location.

        Raises:
            ModelLoadError: If the model type config file cannot be found or parsed.
        """
        self._execution_logger: ExecutionLogger = execution_logger
        resolved_config_path: Path = (
            model_type_config_path
            if model_type_config_path is not None
            else self._default_model_type_config_path()
        )
        self._class_name_to_family: dict[str, str] = self._load_type_detection_config(
            resolved_config_path
        )

    def load(self, pickle_file_path: str | Path) -> LoadedModel:
        """Load the model artifact at the given path and detect its type.

        Args:
            pickle_file_path: Path to the model artifact (.pkl or .joblib).

        Returns:
            LoadedModel with the deserialized object and all extracted metadata.

        Raises:
            ModelLoadError: If the file does not exist or deserialization fails
                with both pickle and joblib.
        """
        file_path = Path(pickle_file_path).resolve()

        self._execution_logger.log_model_validation(
            f"Validating model artifact path: {file_path}"
        )
        self._validate_file_exists(file_path)

        self._execution_logger.log_model_loading(
            f"Loading model artifact: {file_path}"
        )
        model_object, serialization_format = self._deserialize_model(file_path)

        detected_class_name = type(model_object).__name__
        model_family = self._class_name_to_family.get(detected_class_name)

        self._execution_logger.log_model_type_detection(
            f"Detected model class: {detected_class_name} | "
            f"family: {model_family if model_family is not None else 'unsupported'} | "
            f"format: {serialization_format}"
        )

        feature_names_from_model = self._extract_feature_names(model_object)
        num_features_from_model = self._extract_num_features(
            model_object, feature_names_from_model
        )

        return LoadedModel(
            model_object=model_object,
            detected_class_name=detected_class_name,
            model_family=model_family,
            serialization_format=serialization_format,
            feature_names_from_model=feature_names_from_model,
            num_features_from_model=num_features_from_model,
        )

    def _validate_file_exists(self, file_path: Path) -> None:
        if not file_path.is_file():
            raise ModelLoadError(
                f"Model artifact file does not exist: {file_path}"
            )

    def _deserialize_model(self, file_path: Path) -> tuple[Any, str]:
        """Attempt deserialization using pickle or joblib with extension-based ordering.

        Strategy:
          - .joblib extension -> try joblib first, then pickle as fallback
          - .pkl / .pickle / any other extension -> try pickle first, then joblib
        """
        extension = file_path.suffix.lower()
        if extension == _JOBLIB_FILE_EXTENSION:
            primary_format, fallback_format = _FORMAT_JOBLIB, _FORMAT_PICKLE
        else:
            primary_format, fallback_format = _FORMAT_PICKLE, _FORMAT_JOBLIB

        primary_exception: Optional[Exception] = None

        try:
            loaded_object = self._load_with_format(file_path, primary_format)
            return loaded_object, primary_format
        except Exception as first_exc:
            primary_exception = first_exc

        try:
            loaded_object = self._load_with_format(file_path, fallback_format)
            return loaded_object, fallback_format
        except Exception as second_exc:
            raise ModelLoadError(
                f"Model artifact '{file_path}' could not be deserialized. "
                f"{primary_format} failed: {primary_exception}. "
                f"{fallback_format} also failed: {second_exc}."
            ) from second_exc

    @staticmethod
    def _load_with_format(file_path: Path, format_name: str) -> Any:
        if format_name == _FORMAT_JOBLIB:
            return joblib.load(file_path)
        with open(file_path, "rb") as file_handle:
            return pickle.load(file_handle)

    @staticmethod
    def _extract_feature_names(model_object: Any) -> Optional[tuple[str, ...]]:
        """Defensively extract feature names from model metadata, if available.

        Checks in priority order:
          1. feature_names_in_ — standard sklearn attribute set when fitting on a
             pandas DataFrame (sklearn >= 1.0, XGBoost >= 1.6, LightGBM sklearn API).
          2. feature_names_ — CatBoost stores feature names here after training.

        LightGBM's feature_name() method is intentionally skipped: when the model
        was fitted on a numpy array, it returns auto-generated names ("Column_0",
        etc.) that do not match dataset column names and would cause false-positive
        validation errors in SchemaValidator.

        Returns None when feature names are unavailable. The engineered dataset is
        always the authoritative source (spec.md Sec 10).
        """
        # sklearn, XGBoost, LightGBM (fitted with pandas DataFrame)
        sklearn_feature_names = getattr(model_object, "feature_names_in_", None)
        if sklearn_feature_names is not None and len(sklearn_feature_names) > 0:
            return tuple(str(name) for name in sklearn_feature_names)

        # CatBoost uses a separate attribute
        catboost_feature_names = getattr(model_object, "feature_names_", None)
        if catboost_feature_names is not None and len(catboost_feature_names) > 0:
            return tuple(str(name) for name in catboost_feature_names)

        return None

    @staticmethod
    def _extract_num_features(
        model_object: Any,
        feature_names_from_model: Optional[tuple[str, ...]],
    ) -> Optional[int]:
        """Defensively extract feature count from model metadata, if available.

        Checks in priority order:
          1. n_features_in_ — standard sklearn attribute (sklearn, XGBoost, LightGBM).
          2. Derives count from feature_names_from_model length when that was resolved.
          3. num_feature() — LightGBM method that returns a reliable count even when
             the model was fitted without named features (fallback only).
        """
        n_features_attribute = getattr(model_object, "n_features_in_", None)
        if n_features_attribute is not None:
            return int(n_features_attribute)

        if feature_names_from_model is not None:
            return len(feature_names_from_model)

        # LightGBM fallback for older API or when n_features_in_ is absent
        num_feature_method = getattr(model_object, "num_feature", None)
        if callable(num_feature_method):
            try:
                return int(num_feature_method())
            except Exception:
                pass

        return None

    @staticmethod
    def _load_type_detection_config(config_path: Path) -> dict[str, str]:
        """Load and parse the model type detection JSON config.

        Raises:
            ModelLoadError: If the config file is missing or not valid JSON.
        """
        if not config_path.is_file():
            raise ModelLoadError(
                f"Model type detection config not found: {config_path}"
            )
        with open(config_path, "r", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
        return raw_config.get(_CLASS_NAME_TO_FAMILY_KEY, {})

    @staticmethod
    def _default_model_type_config_path() -> Path:
        """Returns the default config path relative to this module's location.

        model_loader.py is at: epic_4_shap/src/backend.agents.evaluation.shap/loaders/
        config/ is at:         epic_4_shap/config/
        """
        return (
            Path(__file__).resolve().parent.parent
            / "config"
            / _MODEL_TYPE_CONFIG_FILENAME
        )
