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
from shap_explainability.models.shap_result import SHAPResult
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

    @staticmethod
    def make_feature_dataframe(
        num_samples: int = 10,
        num_features: int = 3,
    ) -> pd.DataFrame:
        """Creates a synthetic feature DataFrame for use in plot generator tests.

        Args:
            num_samples: Number of rows (samples).
            num_features: Number of feature columns.

        Returns:
            A DataFrame with columns named feature_0, feature_1, ... feature_N-1.
        """
        np.random.seed(42)
        feature_column_names = [f"feature_{index}" for index in range(num_features)]
        return pd.DataFrame(
            np.random.randn(num_samples, num_features),
            columns=feature_column_names,
        )

    @staticmethod
    def make_shap_result_binary(
        num_samples: int = 10,
        num_features: int = 3,
    ) -> SHAPResult:
        """Creates a binary classification SHAPResult with synthetic SHAP values.

        Args:
            num_samples: Number of samples (rows) in the SHAP value array.
            num_features: Number of features (columns) in the SHAP value array.

        Returns:
            A fully constructed SHAPResult for prediction_type='binary_classification'.
        """
        np.random.seed(42)
        feature_names = tuple(f"feature_{index}" for index in range(num_features))
        shap_values = np.random.randn(num_samples, num_features)

        mean_abs_per_feature = np.abs(shap_values).mean(axis=0)
        sorted_feature_indices = np.argsort(mean_abs_per_feature)[::-1]

        global_importance_dataframe = pd.DataFrame({
            "feature_name": [feature_names[idx] for idx in sorted_feature_indices],
            "mean_absolute_shap_value": mean_abs_per_feature[sorted_feature_indices],
        })

        record_ids = np.repeat(np.arange(num_samples), num_features)
        feature_names_repeated = np.tile(list(feature_names), num_samples)
        feature_values_flat = np.random.randn(num_samples * num_features)
        shap_values_flat = shap_values.flatten()

        mapping_dataframe = pd.DataFrame({
            "record_id": record_ids,
            "feature_name": feature_names_repeated,
            "feature_value": feature_values_flat,
            "shap_value": shap_values_flat,
        })

        return SHAPResult(
            prediction_type="binary_classification",
            shap_values_array=shap_values,
            feature_names=feature_names,
            class_names=None,
            global_importance_dataframe=global_importance_dataframe,
            mapping_dataframe=mapping_dataframe,
        )

    @staticmethod
    def make_shap_result_regression(
        num_samples: int = 10,
        num_features: int = 3,
    ) -> SHAPResult:
        """Creates a regression SHAPResult with synthetic SHAP values.

        Args:
            num_samples: Number of samples (rows) in the SHAP value array.
            num_features: Number of features (columns) in the SHAP value array.

        Returns:
            A fully constructed SHAPResult for prediction_type='regression'.
        """
        np.random.seed(7)
        feature_names = tuple(f"feature_{index}" for index in range(num_features))
        shap_values = np.random.randn(num_samples, num_features)

        mean_abs_per_feature = np.abs(shap_values).mean(axis=0)
        sorted_feature_indices = np.argsort(mean_abs_per_feature)[::-1]

        global_importance_dataframe = pd.DataFrame({
            "feature_name": [feature_names[idx] for idx in sorted_feature_indices],
            "mean_absolute_shap_value": mean_abs_per_feature[sorted_feature_indices],
        })

        record_ids = np.repeat(np.arange(num_samples), num_features)
        feature_names_repeated = np.tile(list(feature_names), num_samples)
        feature_values_flat = np.random.randn(num_samples * num_features)
        shap_values_flat = shap_values.flatten()

        mapping_dataframe = pd.DataFrame({
            "record_id": record_ids,
            "feature_name": feature_names_repeated,
            "feature_value": feature_values_flat,
            "shap_value": shap_values_flat,
        })

        return SHAPResult(
            prediction_type="regression",
            shap_values_array=shap_values,
            feature_names=feature_names,
            class_names=None,
            global_importance_dataframe=global_importance_dataframe,
            mapping_dataframe=mapping_dataframe,
        )

    @staticmethod
    def make_shap_result_multiclass(
        num_samples: int = 10,
        num_features: int = 3,
        num_classes: int = 3,
    ) -> SHAPResult:
        """Creates a multiclass SHAPResult with synthetic per-class SHAP value arrays.

        Args:
            num_samples: Number of samples (rows) in each per-class SHAP array.
            num_features: Number of features (columns) in each per-class SHAP array.
            num_classes: Number of classes K; produces a list of K arrays.

        Returns:
            A fully constructed SHAPResult for prediction_type='multiclass_classification'.
            shap_values_array is a list of K ndarrays each of shape (num_samples, num_features).
        """
        np.random.seed(99)
        feature_names = tuple(f"feature_{index}" for index in range(num_features))
        class_names = tuple(f"class_{class_index}" for class_index in range(num_classes))

        # List of K arrays: one per class
        per_class_shap_arrays = [
            np.random.randn(num_samples, num_features) for _ in range(num_classes)
        ]

        # Mean absolute SHAP averaged across all classes for global importance
        stacked_abs = np.mean(
            [np.abs(class_array) for class_array in per_class_shap_arrays], axis=0
        )
        mean_abs_per_feature = stacked_abs.mean(axis=0)
        sorted_feature_indices = np.argsort(mean_abs_per_feature)[::-1]

        global_importance_dataframe = pd.DataFrame({
            "feature_name": [feature_names[idx] for idx in sorted_feature_indices],
            "mean_absolute_shap_value": mean_abs_per_feature[sorted_feature_indices],
        })

        rows = []
        for class_index, class_array in enumerate(per_class_shap_arrays):
            for sample_index in range(num_samples):
                for feature_index in range(num_features):
                    rows.append({
                        "record_id": sample_index,
                        "class_name": class_names[class_index],
                        "feature_name": feature_names[feature_index],
                        "feature_value": np.random.randn(),
                        "shap_value": class_array[sample_index, feature_index],
                    })
        mapping_dataframe = pd.DataFrame(rows)

        return SHAPResult(
            prediction_type="multiclass_classification",
            shap_values_array=per_class_shap_arrays,
            feature_names=feature_names,
            class_names=class_names,
            global_importance_dataframe=global_importance_dataframe,
            mapping_dataframe=mapping_dataframe,
        )
