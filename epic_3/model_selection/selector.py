"""Convenience API for invoking the model-selection orchestrator."""

from __future__ import annotations

from pathlib import Path

from .agents import LLMClient, ModelSelectionOrchestratorAgent
from .schemas import ModelCandidate


def select_models(
    *,
    metadata_path: str | Path,
    feature_selection_path: str | Path,
    mini_data_path: str | Path | None,
    model_library_root: str | Path,
    output_path: str | Path,
    max_models: int = 5,
    llm_client: LLMClient | None = None,
    report_path: str | Path | None = None,
) -> list[ModelCandidate]:
    """Select models strictly from MLKit.MODEL_REGISTRY and write JSON output."""
    return ModelSelectionOrchestratorAgent(
        model_library_root=model_library_root,
        llm_client=llm_client,
    ).run(
        metadata_path=metadata_path,
        feature_selection_path=feature_selection_path,
        mini_data_path=mini_data_path,
        output_path=output_path,
        max_models=max_models,
        report_path=report_path,
    )
