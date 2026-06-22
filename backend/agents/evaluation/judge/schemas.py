import sys
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Reuse task-type constants from model_library to avoid redefinition.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "model_library"))
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
    train_metrics: Optional[Dict[str, Optional[float]]] = None
    test_metrics: Optional[Dict[str, Optional[float]]] = None
    cv_results: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None



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
    # Domain-reasoning agent output (column meanings, problem summary, leakage
    # flags). Generated once per session, attached unchanged to every turn's
    # JudgeInput. None when the domain-reasoning agent did not run or failed.
    domain_reasoning: Optional[Dict[str, Any]] = None


class Finding(BaseModel):
    """One structured Judge finding for a single governance dimension.

    The `status` drives the leaderboard marker (pass => check, fail => cross,
    info => neutral) so the frontend never has to embed unicode glyphs in code.
    """

    dimension: str = Field(..., description="Stable dimension key, e.g. 'predictive_quality'.")
    label: str = Field(..., description="Human-readable dimension label.")
    status: str = Field(..., description="'pass', 'fail', or 'info'.")
    message: str = Field(..., description="One-line human-readable finding.")


class RankedModel(BaseModel):
    """Per-model output in the ranking."""

    model_name: str
    rank: int
    score: float
    verdict: str = Field(
        ...,
        description=(
            "'select' (passed hard gate, in top-N% selection), 'rank_only' "
            "(passed hard gate, ranked but outside the top-N% cutoff), or "
            "'reject' (failed the deterministic hard gate)."
        ),
    )
    reasons: List[str]
    llm_flags: List[str] = Field(default_factory=list)
    # Governance-dashboard additions (deterministic, rule-derived):
    decision: str = Field(
        default="PENDING",
        description="'APPROVED', 'RANKED', or 'REJECTED', derived from verdict.",
    )
    findings: List[Finding] = Field(
        default_factory=list,
        description="Per-dimension structured findings for the model decision card.",
    )
    ranking_explanation: Optional[str] = Field(
        default=None,
        description="Why this model ranked where it did (rule-engine-authored).",
    )
    llm_ranking_reasoning: Optional[str] = Field(
        default=None,
        description=(
            "LLM-authored reasoning for this model's rank, including SHAP/"
            "domain-reasoning correlation. Distinct from ranking_explanation, "
            "which is rule-engine-authored. None when the LLM ranking step "
            "did not run or failed."
        ),
    )


class DecisionTrace(BaseModel):
    """Audit trail separating rule outcomes from LLM commentary."""

    rule_outcomes: Dict[str, Any]
    llm_commentary: Optional[str] = None
    transcript: Optional[str] = None
    llm_ranking_status: Optional[str] = Field(
        default=None,
        description=(
            "'applied' (LLM ranking succeeded and reordered survivors), "
            "'failed' (LLM ranking was attempted but exhausted retries), or "
            "'skipped' (use_llm was False). Never silently missing -- a null "
            "value here only ever means the LLM step was never attempted."
        ),
    )
    llm_ranking_error: Optional[str] = Field(
        default=None,
        description="Error message when llm_ranking_status == 'failed'.",
    )


class JudgeDecision(BaseModel):
    """Final output written to judge_decision.json and returned to the orchestrator."""

    dataset_id: Optional[str]
    selected_model: Optional[str] = Field(
        default=None,
        description="Deprecated: use selected_models[0] when present. Kept for backward compatibility.",
    )
    selected_models: List[str] = Field(
        default_factory=list,
        description="Deterministic top-N% (min-3-floor) selection among eligible ranked models.",
    )
    ranked_models: List[RankedModel]
    decision_trace: DecisionTrace
    comparison_explanation: Optional[str] = Field(
        default=None,
        description="Why the top model was preferred over the runner-up.",
    )
