"""Select and construct the SHAP explainer for a confirmed model family.

Implements spec.md Section 14 and architecture.md steps 7-8: ExplainerFactory
maps a validated model family to a shap.TreeExplainer or shap.LinearExplainer
instance. No SHAP value computation occurs here -- that is SHAPService's
responsibility.

Explainer selection is config-driven via model_type_detection.json
(model_family_to_explainer section) to avoid if-else ladders (CLAUDE.md rule 4).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import shap

from shap_explainability.errors import SHAPExecutionError
from shap_explainability.session_context import SessionContext
from shap_explainability.utils.logger import ExecutionLogger

_MODEL_TYPE_CONFIG_FILENAME: str = "model_type_detection.json"
_FAMILY_TO_EXPLAINER_KEY: str = "model_family_to_explainer"
_EXPLAINER_TREE: str = "TreeExplainer"
_EXPLAINER_LINEAR: str = "LinearExplainer"
_DEFAULT_LINEAR_BACKGROUND_SAMPLES: int = 200


@dataclass(frozen=True)
class BuiltExplainer:
    """Result container for a successfully constructed SHAP explainer.

    Attributes:
        explainer_object: Constructed shap.TreeExplainer or shap.LinearExplainer
            instance, ready for SHAP value computation by SHAPService.
        explainer_name: Human-readable explainer type string written to
            SessionContext.explainer_name and metadata.json (Sec 18).
        model_family: The model family that was used for explainer selection,
            carried for downstream traceability in SHAPService.
    """

    explainer_object: Any
    explainer_name: str
    model_family: str


class ExplainerFactory:
    """Constructs a SHAP explainer appropriate for the confirmed model family.

    Responsibilities (spec.md Sec 14, architecture.md step 7):
      - Load the model_family_to_explainer mapping from model_type_detection.json.
      - Dispatch to _build_tree_explainer or _build_linear_explainer based on
        the config mapping for the given model_family.
      - Return a BuiltExplainer frozen dataclass for SHAPService to use.
      - Write explainer_name to SessionContext for metadata.

    No SHAP value computation occurs here.
    """

    def __init__(
        self,
        execution_logger: ExecutionLogger,
        model_type_config_path: Optional[Path] = None,
        linear_background_samples: int = _DEFAULT_LINEAR_BACKGROUND_SAMPLES,
    ) -> None:
        """Initializes the ExplainerFactory.

        Args:
            execution_logger: Session-scoped logger for recording Sec 19 events.
            model_type_config_path: Path to model_type_detection.json. Defaults
                to epic_4_shap/config/model_type_detection.json.
            linear_background_samples: Maximum number of background samples for
                LinearExplainer masker construction (CFG model section). If the
                feature DataFrame has more rows, a random subsample is used.

        Raises:
            SHAPExecutionError: If the config file cannot be read or parsed.
        """
        self._execution_logger: ExecutionLogger = execution_logger
        self._linear_background_samples: int = linear_background_samples

        resolved_config_path: Path = (
            model_type_config_path
            if model_type_config_path is not None
            else self._default_model_type_config_path()
        )
        self._family_to_explainer: dict[str, str] = self._load_family_to_explainer_config(
            resolved_config_path
        )

    def create(
        self,
        model_family: str,
        model_object: Any,
        feature_dataframe: pd.DataFrame,
        session_context: SessionContext,
    ) -> BuiltExplainer:
        """Construct a SHAP explainer for the given model family.

        Args:
            model_family: Confirmed model family string from ModelValidator
                (e.g. "XGBoost", "LogisticRegression").
            model_object: Deserialized, fitted model object from LoadedModel.
            feature_dataframe: Cleaned, target-excluded feature DataFrame from
                SchemaValidationResult. Required for LinearExplainer background
                data construction; passed but unused by TreeExplainer.
            session_context: Mutable pipeline state. explainer_name is written
                here after successful construction.

        Returns:
            BuiltExplainer containing the constructed explainer, its name, and
            the model family used for selection.

        Raises:
            SHAPExecutionError: If model_family is not in the config mapping, or
                if explainer construction raises an exception.
        """
        explainer_type_name: Optional[str] = self._family_to_explainer.get(model_family)
        if explainer_type_name is None:
            failure_message: str = (
                f"No explainer mapping found for model family '{model_family}' in "
                f"model_type_detection.json [{_FAMILY_TO_EXPLAINER_KEY}]. "
                "Add an entry for this family before running the pipeline."
            )
            self._execution_logger.log_explainer_selection(failure_message)
            session_context.mark_failed(failure_message)
            raise SHAPExecutionError(failure_message)

        self._execution_logger.log_explainer_selection(
            f"Selected explainer: {explainer_type_name} for model family: {model_family}"
        )

        explainer_object: Any = self._dispatch_explainer_build(
            explainer_type_name, model_object, feature_dataframe
        )

        session_context.explainer_name = explainer_type_name
        self._execution_logger.log_explainer_selection(
            f"Explainer constructed successfully: {explainer_type_name} | family: {model_family}"
        )

        return BuiltExplainer(
            explainer_object=explainer_object,
            explainer_name=explainer_type_name,
            model_family=model_family,
        )

    def _dispatch_explainer_build(
        self,
        explainer_type_name: str,
        model_object: Any,
        feature_dataframe: pd.DataFrame,
    ) -> Any:
        """Dispatch to the correct build method based on the explainer type string.

        Args:
            explainer_type_name: Explainer type as configured in JSON
                (e.g. "TreeExplainer" or "LinearExplainer").
            model_object: Fitted model object passed to the explainer constructor.
            feature_dataframe: Feature DataFrame needed for LinearExplainer only.

        Returns:
            Constructed SHAP explainer object.

        Raises:
            SHAPExecutionError: If explainer_type_name is unrecognised or if
                the underlying SHAP constructor raises.
        """
        explainer_dispatch: dict[str, Any] = {
            _EXPLAINER_TREE: lambda: self._build_tree_explainer(model_object),
            _EXPLAINER_LINEAR: lambda: self._build_linear_explainer(
                model_object, feature_dataframe
            ),
        }

        build_function = explainer_dispatch.get(explainer_type_name)
        if build_function is None:
            raise SHAPExecutionError(
                f"Unrecognised explainer type '{explainer_type_name}' in config. "
                f"Supported types: {list(explainer_dispatch.keys())}."
            )

        try:
            return build_function()
        except SHAPExecutionError:
            raise
        except Exception as construction_error:
            raise SHAPExecutionError(
                f"Failed to construct {explainer_type_name} for model: "
                f"{type(model_object).__name__}. "
                f"Underlying error: {construction_error}"
            ) from construction_error

    def _build_tree_explainer(self, model_object: Any) -> Any:
        """Construct shap.TreeExplainer for tree-based model families.

        Supports XGBoost, RandomForest, LightGBM, and CatBoost. No background
        data is required for TreeExplainer construction (only for .shap_values()
        where CatBoost may need check_additivity=False -- handled by SHAPService).

        Args:
            model_object: Fitted tree-based model object.

        Returns:
            shap.TreeExplainer instance.
        """
        self._execution_logger.log_explainer_selection(
            f"Building TreeExplainer for model: {type(model_object).__name__}"
        )
        return shap.TreeExplainer(model_object)

    def _build_linear_explainer(
        self, model_object: Any, feature_dataframe: pd.DataFrame
    ) -> Any:
        """Construct shap.LinearExplainer with a background masker.

        Resolves architecture.md [OPEN A-5]: uses the cleaned inference DataFrame
        as the background population via shap.maskers.Independent. If the DataFrame
        has more rows than linear_background_samples, a random subsample is used to
        control memory usage (see config.ini LINEAR_EXPLAINER_BACKGROUND_SAMPLES).

        Args:
            model_object: Fitted linear model object (e.g. LogisticRegression).
            feature_dataframe: Cleaned, target-excluded feature DataFrame used as
                background data for the LinearExplainer masker.

        Returns:
            shap.LinearExplainer instance.
        """
        num_available_samples: int = len(feature_dataframe)
        if num_available_samples > self._linear_background_samples:
            background_dataframe: pd.DataFrame = feature_dataframe.sample(
                n=self._linear_background_samples, random_state=42
            )
            self._execution_logger.log_explainer_selection(
                f"LinearExplainer: sampled {self._linear_background_samples} background "
                f"rows from {num_available_samples} available (config cap applied)."
            )
        else:
            background_dataframe = feature_dataframe
            self._execution_logger.log_explainer_selection(
                f"LinearExplainer: using full dataset as background "
                f"({num_available_samples} rows)."
            )

        masker = shap.maskers.Independent(background_dataframe)
        return shap.LinearExplainer(model_object, masker)

    @staticmethod
    def _load_family_to_explainer_config(config_path: Path) -> dict[str, str]:
        """Load the model_family_to_explainer mapping from JSON config.

        Args:
            config_path: Path to model_type_detection.json.

        Returns:
            Dictionary mapping model family strings to explainer type strings.

        Raises:
            SHAPExecutionError: If the config file is missing or not valid JSON.
        """
        if not config_path.is_file():
            raise SHAPExecutionError(
                f"Model type detection config not found: {config_path}. "
                "Cannot load explainer family mapping."
            )
        with open(config_path, "r", encoding="utf-8") as config_file:
            raw_config: dict = json.load(config_file)

        family_to_explainer: dict[str, str] = raw_config.get(_FAMILY_TO_EXPLAINER_KEY, {})
        if not family_to_explainer:
            raise SHAPExecutionError(
                f"'{_FAMILY_TO_EXPLAINER_KEY}' section is missing or empty in "
                f"{config_path}. Add model family to explainer type mappings."
            )
        return family_to_explainer

    @staticmethod
    def _default_model_type_config_path() -> Path:
        """Resolves the default config path relative to this module's location.

        explainer_factory.py is at: epic_4_shap/src/shap_explainability/explainers/
        config/ is at:              epic_4_shap/config/
        """
        return (
            Path(__file__).resolve().parent.parent.parent.parent
            / "config"
            / _MODEL_TYPE_CONFIG_FILENAME
        )
