"""Model-library discovery agent.

This module deliberately uses AST parsing instead of importing ``ml_kit.py``.
Importing MLKit would eagerly import sklearn, PyTorch, and XGBoost just to list
model names, which is slow and fragile inside an orchestrator process.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import yaml

from .errors import ModelLibraryCatalogError
from .schemas import ModelDescriptor, TaskType


class ModelLibraryCatalogAgent:
    """Discover the exact trainable models exposed by MLKit.MODEL_REGISTRY."""

    def __init__(self, model_library_root: str | Path) -> None:
        self.root = Path(model_library_root).resolve()
        self.registry_file = self.root / "ml_kit.py"
        self.config_file = self.root / "config" / "config.yaml"
        self._catalog: dict[str, ModelDescriptor] | None = None

    def run(self) -> dict[str, ModelDescriptor]:
        """Return a name -> descriptor mapping sourced only from model_library."""
        if self._catalog is not None:
            return dict(self._catalog)

        if not self.registry_file.is_file():
            raise ModelLibraryCatalogError(
                f"MLKit registry file not found: {self.registry_file}"
            )
        if not self.config_file.is_file():
            raise ModelLibraryCatalogError(
                f"MLKit config file not found: {self.config_file}"
            )

        try:
            source = self.registry_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(self.registry_file))
        except (OSError, SyntaxError) as exc:
            raise ModelLibraryCatalogError(
                f"Unable to parse MLKit registry: {exc}"
            ) from exc

        import_modules = self._import_map(tree)
        registry = self._registry_entries(tree)
        defaults = self._load_defaults()

        catalog: dict[str, ModelDescriptor] = {}
        for model_name, wrapper_class in registry:
            module = import_modules.get(wrapper_class, "")
            task_type = self._task_from_module(model_name, module)
            if model_name not in defaults:
                raise ModelLibraryCatalogError(
                    f"Model '{model_name}' is registered in ml_kit.py but missing "
                    "from model_library/config/config.yaml"
                )
            catalog[model_name] = ModelDescriptor(
                model_name=model_name,
                task_type=task_type,
                wrapper_class=wrapper_class,
                import_module=module,
                default_hyperparameters=defaults[model_name],
            )

        extra_config = sorted(set(defaults) - set(catalog))
        if extra_config:
            raise ModelLibraryCatalogError(
                "config.yaml contains models not exposed by MODEL_REGISTRY: "
                + ", ".join(extra_config)
            )
        if not catalog:
            raise ModelLibraryCatalogError("MODEL_REGISTRY is empty")

        self._catalog = catalog
        return dict(catalog)

    def names(self, task_type: TaskType | None = None) -> list[str]:
        catalog = self.run()
        names = [
            name
            for name, descriptor in catalog.items()
            if task_type is None or descriptor.task_type == task_type
        ]
        return names

    def descriptors(self, task_type: TaskType | None = None) -> list[ModelDescriptor]:
        catalog = self.run()
        return [
            descriptor
            for descriptor in catalog.values()
            if task_type is None or descriptor.task_type == task_type
        ]

    @staticmethod
    def _import_map(tree: ast.AST) -> dict[str, str]:
        imports: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            for alias in node.names:
                imports[alias.asname or alias.name] = node.module
        return imports

    @staticmethod
    def _registry_entries(tree: ast.AST) -> list[tuple[str, str]]:
        value: ast.AST | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if any(
                    isinstance(target, ast.Name) and target.id == "MODEL_REGISTRY"
                    for target in node.targets
                ):
                    value = node.value
                    break
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "MODEL_REGISTRY":
                    value = node.value
                    break

        if not isinstance(value, ast.Dict):
            raise ModelLibraryCatalogError(
                "MODEL_REGISTRY must be a literal dictionary in ml_kit.py"
            )

        entries: list[tuple[str, str]] = []
        for key_node, value_node in zip(value.keys, value.values, strict=True):
            if key_node is None:
                continue
            try:
                model_name = ast.literal_eval(key_node)
            except (ValueError, TypeError) as exc:
                raise ModelLibraryCatalogError(
                    "MODEL_REGISTRY keys must be string literals"
                ) from exc
            if not isinstance(model_name, str) or not isinstance(value_node, ast.Name):
                raise ModelLibraryCatalogError(
                    "MODEL_REGISTRY values must be directly imported wrapper classes"
                )
            entries.append((model_name, value_node.id))
        return entries

    def _load_defaults(self) -> dict[str, dict]:
        try:
            payload = yaml.safe_load(self.config_file.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ModelLibraryCatalogError(
                f"Unable to load model-library config: {exc}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
            raise ModelLibraryCatalogError(
                "model_library/config/config.yaml must contain a top-level 'models' map"
            )
        models = payload["models"]
        for name, config in models.items():
            if not isinstance(name, str) or not isinstance(config, dict):
                raise ModelLibraryCatalogError(
                    "Every model config entry must be a mapping"
                )
        return models

    @staticmethod
    def _task_from_module(model_name: str, module: str) -> TaskType:
        if ".classifiers." in module:
            return "classification"
        if ".regressors." in module:
            return "regression"
        raise ModelLibraryCatalogError(
            f"Cannot infer task type for '{model_name}' from wrapper import '{module}'"
        )
