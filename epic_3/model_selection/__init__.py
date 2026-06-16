"""MITRA model-selection component.

The public entry point is :func:`select_models`.  The implementation never
imports or instantiates the heavy model library while selecting candidates;
it reads ``MODEL_REGISTRY`` from ``model_library/ml_kit.py`` using Python's AST.
"""

from .selector import select_models
from .agents import ModelSelectionOrchestratorAgent
from .catalog import ModelLibraryCatalogAgent

__all__ = [
    "select_models",
    "ModelSelectionOrchestratorAgent",
    "ModelLibraryCatalogAgent",
]
