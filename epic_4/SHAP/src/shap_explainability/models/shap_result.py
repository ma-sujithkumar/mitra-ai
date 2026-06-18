"""Result container for a completed SHAP computation.

Defines the canonical SHAPResult frozen dataclass that SHAPService returns and
all downstream consumers (exporters, visualizers, pipeline) depend on.
Spec.md Section 17 specifies the output schemas this result feeds into.
"""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class SHAPResult:
    """Result container for a completed SHAP computation (spec.md Sec 17).

    Attributes:
        prediction_type: One of "binary_classification",
            "multiclass_classification", or "regression". Determined by
            SHAPService from the model class name and introspection. Written
            to metadata.json by MetadataExporter.
        shap_values_array: Normalised SHAP values in canonical internal form:
            - Binary/Regression: ndarray of shape (n_samples, n_features)
              representing contributions toward the positive class (binary)
              or the regression target.
            - Multiclass: list of K ndarrays each of shape (n_samples, n_features),
              one array per class in class-index order.
        feature_names: Ordered tuple of feature names echoed from the input.
            Length n_features, same ordering as the cleaned feature DataFrame.
        class_names: Class label strings for multiclass only; None for binary
            and regression. Values are from model.classes_ formatted as strings,
            or "class_0", "class_1", ... when model.classes_ contains integers.
        global_importance_dataframe: Per-feature mean absolute SHAP value table
            (Sec 17.1). Columns: feature_name, mean_absolute_shap_value.
            Rows sorted descending by mean_absolute_shap_value. Ready for
            GlobalImportanceExporter.
        mapping_dataframe: Long-form per-record/per-feature SHAP table (Sec 17.2).
            Binary/Regression columns: record_id, feature_name, feature_value,
            shap_value.
            Multiclass columns: record_id, class_name, feature_name,
            feature_value, shap_value.
            Ready for FeatureSHAPMappingExporter.
    """

    prediction_type: str
    shap_values_array: Any
    feature_names: tuple[str, ...]
    class_names: Optional[tuple[str, ...]]
    global_importance_dataframe: pd.DataFrame
    mapping_dataframe: pd.DataFrame
