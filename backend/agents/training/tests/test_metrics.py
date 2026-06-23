from dataclasses import dataclass

from backend.agents.training.metrics import build_metrics_payload


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float


def test_builds_json_safe_classification_payload() -> None:
    train = ClassificationMetrics(1.0, 1.0, 1.0, 1.0, 1.0)
    validation = ClassificationMetrics(0.9, 0.88, 0.9, 0.89, 0.88)

    payload = build_metrics_payload(
        task_type="classification",
        train_metrics=train,
        validation_metrics=validation,
    )

    assert payload["primary_metric"] == "accuracy"
    assert payload["train_score"] == 1.0
    assert payload["validation_score"] == 0.9
