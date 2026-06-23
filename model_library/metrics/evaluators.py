# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


TASK_TYPE_CLASSIFICATION = "classification"
TASK_TYPE_REGRESSION = "regression"
VALID_TASK_TYPES = [TASK_TYPE_CLASSIFICATION, TASK_TYPE_REGRESSION]


@dataclass
class MetricsResult:
    """Typed container for model evaluation results.

    Classification fields are None when task_type is regression, and vice versa.
    """

    task_type: str
    model_name: str

    # Classification metrics
    accuracy: Optional[float]
    f1_macro: Optional[float]
    f1_weighted: Optional[float]
    precision_macro: Optional[float]
    recall_macro: Optional[float]

    # Regression metrics
    mse: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    r2: Optional[float]

    def __str__(self) -> str:
        lines = [f"[{self.model_name}] task_type={self.task_type}"]
        if self.task_type == TASK_TYPE_CLASSIFICATION:
            lines += [
                f"  accuracy       = {self.accuracy:.4f}",
                f"  f1_macro       = {self.f1_macro:.4f}",
                f"  f1_weighted    = {self.f1_weighted:.4f}",
                f"  precision_macro= {self.precision_macro:.4f}",
                f"  recall_macro   = {self.recall_macro:.4f}",
            ]
        else:
            lines += [
                f"  mse  = {self.mse:.4f}",
                f"  rmse = {self.rmse:.4f}",
                f"  mae  = {self.mae:.4f}",
                f"  r2   = {self.r2:.4f}",
            ]
        return "\n".join(lines)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    model_name: str = "unknown",
) -> MetricsResult:
    """Compute standard metrics for classification or regression.

    Args:
        y_true: Ground truth labels or values.
        y_pred: Predicted labels or values from model.predict().
        task_type: Either 'classification' or 'regression'.
        model_name: Used for display in MetricsResult.__str__().

    Returns:
        A MetricsResult dataclass with task-appropriate fields populated.

    Raises:
        ValueError: If task_type is not one of the valid values.
    """
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Invalid task_type '{task_type}'. Must be one of: {VALID_TASK_TYPES}"
        )

    if task_type == TASK_TYPE_CLASSIFICATION:
        return MetricsResult(
            task_type=task_type,
            model_name=model_name,
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            f1_weighted=float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            precision_macro=float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            recall_macro=float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            mse=None,
            rmse=None,
            mae=None,
            r2=None,
        )

    # Regression
    mse_value = float(mean_squared_error(y_true, y_pred))
    return MetricsResult(
        task_type=task_type,
        model_name=model_name,
        accuracy=None,
        f1_macro=None,
        f1_weighted=None,
        precision_macro=None,
        recall_macro=None,
        mse=mse_value,
        rmse=float(np.sqrt(mse_value)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )
