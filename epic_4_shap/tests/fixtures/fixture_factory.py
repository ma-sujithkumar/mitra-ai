"""Standalone test fixture factory for the SHAP Explainability Module.

Implements spec.md Section 22: enables testing without Epic 2 or Epic 3 integration
by synthesizing datasets, trained models, session contexts, and loggers. All methods
are static so the class can be used without instantiation.
"""

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shap_explainability.loaders.dataset_loader import LoadedDataset
from shap_explainability.loaders.model_loader import LoadedModel
from shap_explainability.session_context import SessionContext
from shap_explainability.utils.logger import ExecutionLogger


class FixtureFactory:
    """Factory for constructing lightweight, self-contained test fixtures.

    All factory methods are static and return fully constructed domain objects
    ready for use as inputs to unit tests of any Phase 1-8 pipeline component.
    No file I/O is required unless a tmp_path is passed for logger creation.
    """

    @staticmethod
    def make_execution_logger(
        tmp_path: Path, session_id: Optional[str] = None
    ) -> ExecutionLogger:
        """Creates an ExecutionLogger writing to a temp log file.

        Args:
            tmp_path: Any writable directory (typically pytest's tmp_path fixture).
            session_id: Unique session identifier. Defaults to a UUID-based string
                so each call produces an isolated logger instance.

        Returns:
            A fully initialized ExecutionLogger at INFO level.
        """
        resolved_session_id: str = session_id or f"test-{uuid.uuid4().hex}"
        return ExecutionLogger(
            session_id=resolved_session_id,
            log_file_path=tmp_path / "logs" / "execution.log",
            log_level="INFO",
        )

    @staticmethod
    def make_session_context(
        session_id: Optional[str] = None,
        supplied_model_name: str = "xgboost",
    ) -> SessionContext:
        """Creates a minimal SessionContext suitable for testing validators.

        Args:
            session_id: Unique session identifier. Defaults to a UUID-based string.
            supplied_model_name: The model_name field from the integration payload
                (spec.md Sec 4.2).

        Returns:
            A SessionContext in RUNNING status with all required fields populated.
        """
        resolved_session_id: str = session_id or f"test-{uuid.uuid4().hex}"
        return SessionContext(
            session_id=resolved_session_id,
            supplied_model_name=supplied_model_name,
            pickle_file_path="/test/model.pkl",
            engineered_dataset_path="/test/dataset.csv",
        )

    @staticmethod
    def make_loaded_model(
        detected_class_name: str = "XGBClassifier",
        model_family: Optional[str] = "XGBoost",
        feature_names_from_model: Optional[tuple[str, ...]] = None,
        num_features_from_model: Optional[int] = None,
        serialization_format: str = "pickle",
    ) -> LoadedModel:
        """Creates a LoadedModel fixture without loading a real artifact from disk.

        Args:
            detected_class_name: Python class name as returned by type().__name__.
            model_family: Resolved model family from the detection config. Pass None
                to simulate an unsupported or undetectable model (Rules 3/4).
            feature_names_from_model: Optional tuple of feature names from model
                metadata. Mirrors what ModelLoader would extract from feature_names_in_.
            num_features_from_model: Optional feature count from model metadata.
                Mirrors what ModelLoader would extract from n_features_in_.
            serialization_format: The deserialization format used ("pickle"/"joblib").

        Returns:
            A frozen LoadedModel ready for use in ModelValidator and SchemaValidator tests.
        """
        return LoadedModel(
            model_object=object(),
            detected_class_name=detected_class_name,
            model_family=model_family,
            serialization_format=serialization_format,
            feature_names_from_model=feature_names_from_model,
            num_features_from_model=num_features_from_model,
        )

    @staticmethod
    def make_loaded_dataset(
        feature_names: Optional[list[str]] = None,
        num_rows: int = 10,
        include_target_column: bool = True,
        target_column_name: str = "target",
    ) -> LoadedDataset:
        """Creates a LoadedDataset fixture with synthetic numeric feature values.

        Args:
            feature_names: Feature column names. Defaults to ["feature_a", "feature_b"].
            num_rows: Number of data rows to generate.
            include_target_column: Whether to include a target column in the dataset.
            target_column_name: Name of the target column to include when
                include_target_column is True.

        Returns:
            A frozen LoadedDataset ready for use in SchemaValidator tests.
        """
        resolved_feature_names: list[str] = (
            feature_names if feature_names is not None else ["feature_a", "feature_b"]
        )

        column_data: dict = {
            feature_name: np.random.rand(num_rows).tolist()
            for feature_name in resolved_feature_names
        }

        if include_target_column:
            column_data[target_column_name] = [row_index % 2 for row_index in range(num_rows)]

        dataframe: pd.DataFrame = pd.DataFrame(column_data)
        column_names: tuple[str, ...] = tuple(str(col) for col in dataframe.columns)

        return LoadedDataset(
            dataframe=dataframe,
            column_names=column_names,
            num_rows=len(dataframe),
            num_columns=len(column_names),
        )
