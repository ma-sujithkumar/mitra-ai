"""Route model-selection output into deterministic training jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from backend.agents.model_selection.catalog import ModelLibraryCatalogAgent
from backend.agents.model_selection.schemas import ModelDescriptor

from .contracts import (
    OrchestratorMetadata,
    SelectedModelConfig,
    TrainerType,
    TrainingJob,
)
from .errors import InvalidModelConfigError, ModelRoutingError


class ModelRouter:
    """Validate selected models against MLKit and create ``TrainingJob`` items.

    The model library remains the source of truth for model existence, task type,
    and default parameters.  The router never imports or instantiates estimators.
    """

    def __init__(self, model_library_root: str | Path) -> None:
        self._catalog = ModelLibraryCatalogAgent(model_library_root).run()

    def route_all(
        self,
        *,
        selected_models: Iterable[SelectedModelConfig],
        metadata: OrchestratorMetadata,
        train_path: str | Path,
        test_path: str | Path,
        session_dir: str | Path,
        model_id_start: int = 1,
    ) -> list[TrainingJob]:
        models = sorted(selected_models, key=lambda item: item.priority)
        if not models:
            raise InvalidModelConfigError("model_config.json must contain at least one model")

        names = [item.model_name for item in models]
        if len(names) != len(set(names)):
            raise InvalidModelConfigError("model_config.json contains duplicate model names")

        priorities = [item.priority for item in models]
        if len(priorities) != len(set(priorities)):
            raise InvalidModelConfigError("model_config.json contains duplicate priorities")

        if metadata.problem_type == "unsupervised":
            raise ModelRoutingError(
                "The current model library exposes no unsupervised estimator; "
                "training jobs cannot be created"
            )
        if not metadata.output_cols:
            raise ModelRoutingError(
                f"{metadata.problem_type} metadata must declare at least one output column"
            )
        if metadata.data_format == "image" and metadata.problem_type != "classification":
            raise ModelRoutingError("Image routing currently supports classification only")

        train = Path(train_path).resolve()
        test = Path(test_path).resolve()
        root = Path(session_dir).resolve()

        jobs: list[TrainingJob] = []
        # model_id_start lets callback training use IDs that don't conflict with already-trained models.
        for index, selected in enumerate(models, start=model_id_start):
            descriptor = self._resolve_descriptor(selected, metadata)
            trainer_type = self._trainer_type(descriptor, metadata)
            model_id = f"model_{index:03d}"
            jobs.append(
                TrainingJob(
                    model_id=model_id,
                    model_name=descriptor.model_name,
                    task_type=descriptor.task_type,
                    data_format=metadata.data_format,
                    trainer_type=trainer_type,
                    # Defaults are copied from the model library, not trusted from
                    # the generated model_config.json.
                    parameters=dict(descriptor.default_hyperparameters),
                    train_path=str(train),
                    test_path=str(test),
                    output_dir=str(root / model_id),
                    priority=selected.priority,
                    rationale=selected.rationale,
                    source="model_library/ml_kit.py::MODEL_REGISTRY",
                )
            )
        return jobs

    def _resolve_descriptor(
        self,
        selected: SelectedModelConfig,
        metadata: OrchestratorMetadata,
    ) -> ModelDescriptor:
        descriptor = self._catalog.get(selected.model_name)
        if descriptor is None:
            raise ModelRoutingError(
                f"Selected model '{selected.model_name}' is not present in "
                "model_library/ml_kit.py::MODEL_REGISTRY"
            )
        if selected.task_type != descriptor.task_type:
            raise ModelRoutingError(
                f"Selected model '{selected.model_name}' declares task_type="
                f"'{selected.task_type}' but the model library declares "
                f"'{descriptor.task_type}'"
            )
        if descriptor.task_type != metadata.problem_type:
            raise ModelRoutingError(
                f"Model '{selected.model_name}' is a {descriptor.task_type} model "
                f"and cannot be routed for a {metadata.problem_type} dataset"
            )
        if (
            selected.default_hyperparameters
            and selected.default_hyperparameters != descriptor.default_hyperparameters
        ):
            raise ModelRoutingError(
                f"Default parameters for '{selected.model_name}' do not match the "
                "model library; regenerate model_config.json"
            )
        return descriptor

    @staticmethod
    def _trainer_type(
        descriptor: ModelDescriptor,
        metadata: OrchestratorMetadata,
    ) -> TrainerType:
        if metadata.data_format == "image":
            # MODEL_REGISTRY currently contains one purpose-built image wrapper.
            # This is a routing capability check, not an alternative model list.
            if "CNN" not in descriptor.wrapper_class:
                raise ModelRoutingError(
                    f"Model '{descriptor.model_name}' is not image-compatible"
                )
            return "image_classification"
        if descriptor.task_type == "classification":
            return "tabular_classification"
        if descriptor.task_type == "unsupervised":
            return "tabular_clustering"
        return "tabular_regression"
