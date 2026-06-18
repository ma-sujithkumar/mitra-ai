"""Validate dataset schema and feature compatibility for SHAP processing.

Implements spec.md Sections 9-13:
  Sec 11 - Target column identification and exclusion before SHAP processing.
  Sec 12 - Schema compatibility validation between dataset and loaded model.
  Sec 13 - Dynamic feature name enforcement (no hardcoding).

DatasetLoader handles basic structural validation (file exists, non-empty, has columns).
SchemaValidator handles everything after: target column removal, feature resolution,
and model compatibility checking.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from shap_explainability.errors import SchemaValidationError
from shap_explainability.loaders.dataset_loader import LoadedDataset
from shap_explainability.loaders.model_loader import LoadedModel
from shap_explainability.session_context import SessionContext
from shap_explainability.utils.logger import ExecutionLogger


@dataclass(frozen=True)
class SchemaValidationResult:
    """Cleaned, feature-only view of the dataset after schema validation (Sec 11-13).

    Attributes:
        feature_dataframe: DataFrame with the target column removed (Sec 11).
            Column ordering is preserved exactly as in the original dataset (Sec 10, 13).
        feature_names: Ordered tuple of feature column names from feature_dataframe.
            This is the authoritative feature name list for all downstream SHAP
            processing (Sec 10, 13). Names are never hardcoded.
        target_column_name: Name of the column excluded from SHAP processing, or
            None if no target column matched any configured candidate.
        num_samples: Number of data rows in feature_dataframe.
        num_features: Number of feature columns in feature_dataframe.
    """

    feature_dataframe: pd.DataFrame
    feature_names: tuple[str, ...]
    target_column_name: Optional[str]
    num_samples: int
    num_features: int


class SchemaValidator:
    """Identifies the target column and validates dataset/model feature compatibility.

    Responsibilities (spec.md Sec 9-13, architecture.md Section 3 and 4):
      - Identify the target column from a configurable candidate list (Sec 11, CFG-03).
        First matching candidate wins; matching is exact and case-sensitive.
      - Exclude the identified target column before SHAP processing (Sec 11).
      - Validate at least one feature remains after target column exclusion (Sec 9).
      - Cross-validate feature schema against model metadata when available (Sec 12):
          - Both names and count present: validate names (count is implicit).
          - Count only: validate count.
          - Names only: validate names.
          - Neither present: log a warning and skip schema validation.
      - Enforce the Sec 13 invariant that feature names always come from the dataset,
        never from hardcoded literals or model-only metadata.
    """

    def __init__(
        self,
        execution_logger: ExecutionLogger,
        target_column_candidates: tuple[str, ...],
    ) -> None:
        """Initializes the SchemaValidator.

        Args:
            execution_logger: Session-scoped logger for recording Sec 19 events.
            target_column_candidates: Ordered candidate column names for target
                column identification (CFG-03). First match wins.
        """
        self._execution_logger: ExecutionLogger = execution_logger
        self._target_column_candidates: tuple[str, ...] = target_column_candidates

    def validate(
        self,
        loaded_dataset: LoadedDataset,
        loaded_model: LoadedModel,
        session_context: SessionContext,
    ) -> SchemaValidationResult:
        """Identify the target column, exclude it, and validate feature compatibility.

        Updates session_context with target_column_name, feature_names, num_samples,
        and num_features. Adds a non-terminating warning when no target column is
        found or when model metadata is absent.

        Args:
            loaded_dataset: Structurally validated dataset from DatasetLoader.
            loaded_model: Loaded model artifact from ModelLoader; used for optional
                feature metadata cross-validation (Sec 12).
            session_context: Mutable pipeline state written by this validator.

        Returns:
            SchemaValidationResult containing the cleaned feature-only DataFrame
            and resolved feature metadata.

        Raises:
            SchemaValidationError: If zero features remain after target exclusion
                (Sec 9), or if feature names/count are incompatible with model
                metadata (Sec 12).
        """
        all_column_names: tuple[str, ...] = loaded_dataset.column_names

        target_column_name: Optional[str] = self._identify_target_column(all_column_names)
        if target_column_name is not None:
            self._execution_logger.log_schema_validation(
                f"Target column identified: '{target_column_name}'. "
                "Excluding from SHAP processing."
            )
        else:
            no_target_warning: str = (
                f"No target column found matching candidates "
                f"{self._target_column_candidates}. "
                "Treating all columns as features."
            )
            self._execution_logger.log_schema_validation(no_target_warning, logging.WARNING)
            session_context.add_warning(no_target_warning)

        feature_dataframe: pd.DataFrame = self._exclude_target_column(
            loaded_dataset.dataframe, target_column_name
        )
        feature_names: tuple[str, ...] = tuple(
            str(column) for column in feature_dataframe.columns
        )

        # Sec 9: feature count must be > 0 after target exclusion.
        if len(feature_names) == 0:
            failure_message = (
                "No feature columns remain after excluding the target column. "
                "The dataset must contain at least one feature column for SHAP processing."
            )
            self._execution_logger.log_schema_validation(failure_message, logging.ERROR)
            session_context.mark_failed(failure_message)
            raise SchemaValidationError(failure_message)

        self._execution_logger.log_schema_validation(
            f"Feature columns resolved: {len(feature_names)} features, "
            f"{len(feature_dataframe)} samples."
        )

        # Sec 12: cross-validate feature schema against model metadata when available.
        try:
            self._validate_feature_compatibility(
                feature_names, loaded_model, session_context
            )
        except SchemaValidationError as schema_error:
            session_context.mark_failed(str(schema_error))
            raise

        session_context.target_column_name = target_column_name
        session_context.feature_names = list(feature_names)
        session_context.num_samples = len(feature_dataframe)
        session_context.num_features = len(feature_names)

        self._execution_logger.log_schema_validation(
            "Schema validation completed successfully."
        )

        return SchemaValidationResult(
            feature_dataframe=feature_dataframe,
            feature_names=feature_names,
            target_column_name=target_column_name,
            num_samples=len(feature_dataframe),
            num_features=len(feature_names),
        )

    def _identify_target_column(
        self, column_names: tuple[str, ...]
    ) -> Optional[str]:
        """Returns the first candidate column name found in the dataset columns.

        Matching is exact and case-sensitive; candidate order determines priority.

        Args:
            column_names: All column names from the loaded dataset.

        Returns:
            The matched candidate name, or None if no candidate is present.
        """
        column_name_set: set[str] = set(column_names)
        for candidate_name in self._target_column_candidates:
            if candidate_name in column_name_set:
                return candidate_name
        return None

    @staticmethod
    def _exclude_target_column(
        dataframe: pd.DataFrame, target_column_name: Optional[str]
    ) -> pd.DataFrame:
        """Returns the dataframe with the target column dropped, or unchanged if None."""
        if target_column_name is None:
            return dataframe
        return dataframe.drop(columns=[target_column_name])

    def _validate_feature_compatibility(
        self,
        feature_names: tuple[str, ...],
        loaded_model: LoadedModel,
        session_context: SessionContext,
    ) -> None:
        """Cross-validates feature names and count against model metadata when available.

        Four combinations handled per architecture.md Section 4:
          - Both names and count available: validate names (count implicit in name list).
          - Count only: validate count.
          - Names only: validate names.
          - Neither available: log warning and skip (dataset schema is authoritative).

        Args:
            feature_names: Ordered feature names resolved from the dataset.
            loaded_model: Loaded model with optional feature metadata.
            session_context: Used to record the no-metadata warning.

        Raises:
            SchemaValidationError: If feature count or names are incompatible.
        """
        model_feature_names: Optional[tuple[str, ...]] = loaded_model.feature_names_from_model
        model_feature_count: Optional[int] = loaded_model.num_features_from_model

        if model_feature_names is None and model_feature_count is None:
            no_metadata_warning: str = (
                f"Model '{loaded_model.detected_class_name}' provides no feature metadata. "
                "Skipping schema compatibility check; proceeding with dataset feature schema."
            )
            self._execution_logger.log_schema_validation(
                no_metadata_warning, logging.WARNING
            )
            session_context.add_warning(no_metadata_warning)
            return

        if model_feature_names is not None:
            self._validate_feature_names_match(
                feature_names, model_feature_names, loaded_model.detected_class_name
            )
        elif model_feature_count is not None:
            # Count-only path: names are unavailable so only count can be checked.
            self._validate_feature_count_matches(
                len(feature_names), model_feature_count, loaded_model.detected_class_name
            )

    def _validate_feature_names_match(
        self,
        dataset_feature_names: tuple[str, ...],
        model_feature_names: tuple[str, ...],
        detected_class_name: str,
    ) -> None:
        """Validates that dataset feature names match the model's expected feature names.

        Count is validated implicitly: if lengths differ the name comparison would
        also fail, but we produce a clearer count-mismatch message first.

        Raises:
            SchemaValidationError: If feature counts differ or any name does not match.
        """
        if len(dataset_feature_names) != len(model_feature_names):
            failure_message = (
                f"Feature count mismatch: dataset has {len(dataset_feature_names)} features "
                f"but model '{detected_class_name}' was trained on "
                f"{len(model_feature_names)} features."
            )
            self._execution_logger.log_schema_validation(failure_message, logging.ERROR)
            raise SchemaValidationError(failure_message)

        mismatched_pairs: list[tuple[str, str]] = [
            (dataset_name, model_name)
            for dataset_name, model_name in zip(dataset_feature_names, model_feature_names)
            if dataset_name != model_name
        ]
        if mismatched_pairs:
            first_dataset_name, first_model_name = mismatched_pairs[0]
            failure_message = (
                f"Feature name mismatch between dataset and model '{detected_class_name}'. "
                f"First mismatch: dataset column '{first_dataset_name}' vs "
                f"model feature '{first_model_name}'. "
                f"Total mismatched columns: {len(mismatched_pairs)}."
            )
            self._execution_logger.log_schema_validation(failure_message, logging.ERROR)
            raise SchemaValidationError(failure_message)

        self._execution_logger.log_schema_validation(
            f"Feature names validated: {len(dataset_feature_names)} features match "
            f"model '{detected_class_name}' metadata."
        )

    def _validate_feature_count_matches(
        self,
        dataset_feature_count: int,
        model_feature_count: int,
        detected_class_name: str,
    ) -> None:
        """Validates that dataset feature count matches the model's expected feature count.

        Raises:
            SchemaValidationError: If feature counts do not match.
        """
        if dataset_feature_count != model_feature_count:
            failure_message = (
                f"Feature count mismatch: dataset has {dataset_feature_count} features "
                f"but model '{detected_class_name}' was trained on "
                f"{model_feature_count} features."
            )
            self._execution_logger.log_schema_validation(failure_message, logging.ERROR)
            raise SchemaValidationError(failure_message)

        self._execution_logger.log_schema_validation(
            f"Feature count validated: {dataset_feature_count} features match "
            f"model '{detected_class_name}' metadata."
        )
