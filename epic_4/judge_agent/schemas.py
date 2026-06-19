import sys
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Reuse task-type constants from model_library to avoid redefinition.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "model_library"))
from metrics.evaluators import TASK_TYPE_CLASSIFICATION, TASK_TYPE_REGRESSION, VALID_TASK_TYPES


class ComplexityDescriptor(BaseModel):
    """Explicit complexity descriptor supplied per model."""

    n_params: int = Field(..., description="Number of model parameters.")
    depth: int = Field(..., description="Tree depth or network layer count.")
    family_rank: int = Field(
        ...,
        description=(
            "Integer rank within the model family from simplest (1) to most complex (N). "
            "E.g. LinearRegression=1, Ridge=2, GradientBoosting=8."
        ),
    )


class OverfittingInfo(BaseModel):
    """Adapted subset of the overfitting tool's output, keyed fields only."""

    is_overfitted: bool
    gap: float = Field(
        ...,
        description="Direction-aware primary metric gap (train - test for higher-is-better).",
    )
    train_vs_cv_gap: Optional[float] = Field(
        None,
        description="Train score minus CV mean. Null if K-fold was skipped.",
    )


class CandidateModel(BaseModel):
    """A single ML model candidate to be judged."""

    model_name: str
    task_type: str = Field(..., description="'classification' or 'regression'.")
    # Performance metrics dict: keys match MetricsResult field names.
    metrics: Dict[str, Optional[float]]
    overfitting: OverfittingInfo
    complexity: ComplexityDescriptor
    # Context-only inputs: passed to the LLM for rationale, not scored.
    shap_summary: Optional[Dict[str, Any]] = None
    hyperparam_sensitivity: Optional[Dict[str, Any]] = None


class JudgeInput(BaseModel):
    """Full input to the judge agent."""

    dataset_id: Optional[str] = None
    candidates: List[CandidateModel]
    # Dataset-level context-only inputs.
    minidata: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class RankedModel(BaseModel):
    """Per-model output in the ranking."""

    model_name: str
    rank: int
    score: float
    verdict: str = Field(..., description="'select' or 'reject'.")
    reasons: List[str]
    llm_flags: List[str] = Field(default_factory=list)


class DecisionTrace(BaseModel):
    """Audit trail separating rule outcomes from LLM commentary."""

    rule_outcomes: Dict[str, Any]
    llm_commentary: Optional[str] = None


class JudgeDecision(BaseModel):
    """Final output written to judge_decision.json and returned to the orchestrator."""

    dataset_id: Optional[str]
    selected_model: Optional[str]
    ranked_models: List[RankedModel]
    decision_trace: DecisionTrace
