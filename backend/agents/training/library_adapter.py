"""Execute an exact registry model through the existing MLKit library."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

from backend.agents.model_selection.catalog import ModelLibraryCatalogAgent
from backend.agents.training_orchestrator.contracts import TrainingJob

from .data_loader import LoadedTrainingData
from .errors import ModelLibraryExecutionError
from .metrics import build_metrics_payload

_IMPORT_LOCK = Lock()


@dataclass(frozen=True)
class TrainedModelArtifacts:
    metrics: dict[str, Any]
    model_path: Path


class MLKitTrainingAdapter:
    """Thin adapter around ``model_library/ml_kit.py``.

    Model existence and task type are revalidated here so the training worker
    never silently substitutes a model, even when it receives a hand-written
    ``TrainingJob``.
    """

    def __init__(self, model_library_root: str | Path) -> None:
        self.root = Path(model_library_root).expanduser().resolve()
        if not (self.root / "ml_kit.py").is_file():
            raise ModelLibraryExecutionError(
                f"model library root does not contain ml_kit.py: {self.root}"
            )
        self.catalog = ModelLibraryCatalogAgent(self.root).run()

    def train_and_evaluate(
        self,
        *,
        job: TrainingJob,
        data: LoadedTrainingData,
        model_path: str | Path,
    ) -> TrainedModelArtifacts:
        descriptor = self.catalog.get(job.model_name)
        if descriptor is None:
            raise ModelLibraryExecutionError(
                f"model '{job.model_name}' is not present in MODEL_REGISTRY"
            )
        if descriptor.task_type != job.task_type:
            raise ModelLibraryExecutionError(
                f"model '{job.model_name}' is registered as {descriptor.task_type}, "
                f"not {job.task_type}"
            )

        runtime = self._load_runtime()
        common = runtime.CommonData(
            X_train=data.X_train,
            y_train=data.y_train,
            X_test=data.X_test,
            y_test=data.y_test,
        )
        bundle = runtime.DataBundle(
            common=common,
            hyperparameters=dict(job.parameters),
        )
        destination = Path(model_path).expanduser().resolve()

        try:
            kit = runtime.MLKit(
                model_name=job.model_name,
                data=bundle,
                training_mode=runtime.TRAINING_MODE_FULL_TRAIN,
            )
            kit.train()
            train_predictions = np.asarray(kit.model.predict(data.X_train))
            validation_predictions = np.asarray(kit.test())
            kit.save(str(destination))
        except Exception as exc:  # MLKit exposes several third-party exception types.
            raise ModelLibraryExecutionError(
                f"MLKit execution failed for '{job.model_name}': {exc}"
            ) from exc

        if not destination.is_file():
            raise ModelLibraryExecutionError(
                f"MLKit did not create the model artifact: {destination}"
            )

        # Clustering: compute silhouette score from X (no y_true ground truth).
        if job.task_type == "unsupervised":
            train_score = _silhouette_safe(data.X_train, train_predictions)
            val_score = _silhouette_safe(data.X_test, validation_predictions)
            return TrainedModelArtifacts(
                metrics={
                    "task_type": "unsupervised",
                    "primary_metric": "silhouette_score",
                    "train_score": train_score,
                    "validation_score": val_score,
                    "train": {"silhouette_score": train_score},
                    "validation": {"silhouette_score": val_score},
                },
                model_path=destination,
            )

        train_metrics = runtime.compute_metrics(
            data.y_train,
            train_predictions,
            job.task_type,
            job.model_name,
        )
        validation_metrics = runtime.compute_metrics(
            data.y_test,
            validation_predictions,
            job.task_type,
            job.model_name,
        )
        return TrainedModelArtifacts(
            metrics=build_metrics_payload(
                task_type=job.task_type,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            ),
            model_path=destination,
        )

    def _load_runtime(self) -> Any:
        """Import the legacy library whose modules use root-relative imports."""

        with _IMPORT_LOCK:
            root_str = str(self.root)
            inserted = root_str not in sys.path
            if inserted:
                sys.path.insert(0, root_str)
            try:
                ml_kit = importlib.import_module("ml_kit")
                data_bundle = importlib.import_module("core.data_bundle")
                evaluators = importlib.import_module("metrics.evaluators")
            except Exception as exc:
                raise ModelLibraryExecutionError(
                    f"failed to import MLKit from {self.root}: {exc}"
                ) from exc
            finally:
                if inserted:
                    try:
                        sys.path.remove(root_str)
                    except ValueError:
                        pass

        class Runtime:
            MLKit = ml_kit.MLKit
            TRAINING_MODE_FULL_TRAIN = ml_kit.TRAINING_MODE_FULL_TRAIN
            CommonData = data_bundle.CommonData
            DataBundle = data_bundle.DataBundle
            compute_metrics = staticmethod(evaluators.compute_metrics)

        return Runtime


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score safely, returning 0.0 on degenerate inputs."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return 0.0
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return 0.0
