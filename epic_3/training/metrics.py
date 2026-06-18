"""Normalize MLKit's typed metric objects into JSON-safe dictionaries."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


_CLASSIFICATION_FIELDS = (
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
)
_REGRESSION_FIELDS = ("mse", "rmse", "mae", "r2")


def build_metrics_payload(
    *,
    task_type: str,
    train_metrics: Any,
    validation_metrics: Any,
) -> dict[str, Any]:
    """Create the standard train/validation metrics payload."""

    train = _metric_object_to_dict(train_metrics, task_type)
    validation = _metric_object_to_dict(validation_metrics, task_type)
    primary_field = "accuracy" if task_type == "classification" else "r2"
    return {
        "task_type": task_type,
        "primary_metric": primary_field,
        "train_score": train[primary_field],
        "validation_score": validation[primary_field],
        "train": train,
        "validation": validation,
    }


def _metric_object_to_dict(metric_object: Any, task_type: str) -> dict[str, float]:
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"unsupported task_type: {task_type}")

    if is_dataclass(metric_object):
        raw = asdict(metric_object)
    elif hasattr(metric_object, "model_dump"):
        raw = metric_object.model_dump()
    elif isinstance(metric_object, dict):
        raw = dict(metric_object)
    else:
        raw = vars(metric_object)

    fields = (
        _CLASSIFICATION_FIELDS
        if task_type == "classification"
        else _REGRESSION_FIELDS
    )
    result: dict[str, float] = {}
    for field in fields:
        value = raw.get(field)
        if value is None:
            raise ValueError(f"metric '{field}' is missing for task_type={task_type}")
        result[field] = float(value)
    return result
