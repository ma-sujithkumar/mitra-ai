"""
Tests for findings_engine.py: per-dimension Judge findings, ranking explanation,
and head-to-head comparison explanation.

All tests are deterministic and run with no network or LLM access.
"""

from typing import Dict, List, Optional

import pytest

from backend.agents.evaluation.judge.findings_engine import (
    STATUS_FAIL,
    STATUS_INFO,
    STATUS_PASS,
    FindingsEngine,
)
from backend.agents.evaluation.judge.rule_engine import RuleEngine
from backend.agents.evaluation.judge.schemas import (
    CandidateModel,
    ComplexityDescriptor,
    OverfittingInfo,
)


def _make_candidate(
    model_name: str,
    task_type: str = "classification",
    metrics: Optional[Dict[str, float]] = None,
    gap: float = 0.05,
    is_overfitted: bool = False,
    train_metrics: Optional[Dict[str, float]] = None,
    cv_results: Optional[Dict[str, float]] = None,
    shap_summary: Optional[Dict[str, object]] = None,
) -> CandidateModel:
    default_metrics = {
        "accuracy": 0.85,
        "f1_macro": 0.80,
        "f1_weighted": 0.83,
        "precision_macro": 0.81,
        "recall_macro": 0.79,
    }
    return CandidateModel(
        model_name=model_name,
        task_type=task_type,
        metrics=metrics if metrics is not None else default_metrics,
        overfitting=OverfittingInfo(
            is_overfitted=is_overfitted,
            gap=gap,
            train_vs_cv_gap=gap,
            train_metrics=train_metrics,
            cv_results=cv_results,
        ),
        complexity=ComplexityDescriptor(n_params=1000, depth=5, family_rank=5),
        shap_summary=shap_summary,
    )


def _status_by_dimension(findings) -> Dict[str, str]:
    return {finding.dimension: finding.status for finding in findings}


class TestPerDimensionFindings:

    def test_all_six_dimensions_present(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        findings = engine.build_findings(_make_candidate("M"))
        dimensions = {finding.dimension for finding in findings}
        assert dimensions == {
            "predictive_quality",
            "class_balance",
            "generalization",
            "feature_utilization",
            "data_leakage",
            "robustness",
        }

    def test_strong_model_passes_predictive_quality(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        findings = engine.build_findings(_make_candidate("Strong"))
        statuses = _status_by_dimension(findings)
        assert statuses["predictive_quality"] == STATUS_PASS

    def test_below_floor_fails_predictive_quality(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        weak = _make_candidate("Weak", metrics={"accuracy": 0.30, "recall_macro": 0.30})
        statuses = _status_by_dimension(engine.build_findings(weak))
        assert statuses["predictive_quality"] == STATUS_FAIL

    def test_weak_minority_recall_fails_class_balance(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        skewed = _make_candidate(
            "Skewed",
            metrics={"accuracy": 0.80, "f1_macro": 0.40, "f1_weighted": 0.78, "recall_macro": 0.35},
        )
        statuses = _status_by_dimension(engine.build_findings(skewed))
        assert statuses["class_balance"] == STATUS_FAIL

    def test_overfitting_fails_generalization(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        overfit = _make_candidate("Overfit", gap=0.40, is_overfitted=True)
        statuses = _status_by_dimension(engine.build_findings(overfit))
        assert statuses["generalization"] == STATUS_FAIL

    def test_dominant_feature_fails_feature_utilization(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        dominant_shap = {
            "top_features": ["f1", "f2", "f3"],
            "mean_abs_shap": {"f1": 0.90, "f2": 0.05, "f3": 0.05},
            "feature_concentration": 1.0,
        }
        candidate = _make_candidate("Dominant", shap_summary=dominant_shap)
        statuses = _status_by_dimension(engine.build_findings(candidate))
        assert statuses["feature_utilization"] == STATUS_FAIL

    def test_diverse_features_pass_feature_utilization(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        diverse_shap = {
            "top_features": ["f1", "f2", "f3", "f4"],
            "mean_abs_shap": {"f1": 0.30, "f2": 0.25, "f3": 0.25, "f4": 0.20},
            "feature_concentration": 0.6,
        }
        candidate = _make_candidate("Diverse", shap_summary=diverse_shap)
        statuses = _status_by_dimension(engine.build_findings(candidate))
        assert statuses["feature_utilization"] == STATUS_PASS

    def test_missing_shap_is_info(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        statuses = _status_by_dimension(engine.build_findings(_make_candidate("NoShap")))
        assert statuses["feature_utilization"] == STATUS_INFO

    def test_leakage_detected_when_train_perfect_and_large_gap(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        leaky = _make_candidate(
            "Leaky",
            gap=0.30,
            train_metrics={"accuracy": 1.0},
        )
        statuses = _status_by_dimension(engine.build_findings(leaky))
        assert statuses["data_leakage"] == STATUS_FAIL

    def test_no_leakage_when_train_metrics_reasonable(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        clean = _make_candidate("Clean", gap=0.05, train_metrics={"accuracy": 0.88})
        statuses = _status_by_dimension(engine.build_findings(clean))
        assert statuses["data_leakage"] == STATUS_PASS

    def test_unstable_cv_fails_robustness(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        unstable = _make_candidate("Unstable", cv_results={"mean": 0.80, "std": 0.20})
        statuses = _status_by_dimension(engine.build_findings(unstable))
        assert statuses["robustness"] == STATUS_FAIL

    def test_stable_cv_passes_robustness(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        stable = _make_candidate("Stable", cv_results={"mean": 0.86, "std": 0.01})
        statuses = _status_by_dimension(engine.build_findings(stable))
        assert statuses["robustness"] == STATUS_PASS

    def test_regression_class_balance_is_info(self, judge_config: dict) -> None:
        engine = FindingsEngine(judge_config)
        regressor = _make_candidate("Reg", task_type="regression", metrics={"r2": 0.75})
        statuses = _status_by_dimension(engine.build_findings(regressor))
        assert statuses["class_balance"] == STATUS_INFO


class TestExplanations:

    def test_ranking_explanation_for_winner_via_rule_engine(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        strong = _make_candidate("Strong", gap=0.03, cv_results={"mean": 0.86, "std": 0.01})
        decision = engine.rank([strong], {}, [strong])
        winner = decision.ranked_models[0]
        assert winner.decision == "APPROVED"
        assert winner.ranking_explanation is not None
        assert "Why Ranked #1" in winner.ranking_explanation
        assert len(winner.findings) == 6

    def test_rejected_model_has_findings_and_decision(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        weak = _make_candidate("Weak", metrics={"accuracy": 0.30, "recall_macro": 0.30})
        survivors, gate_outcomes = engine.apply_hard_gates([weak])
        decision = engine.rank(survivors, gate_outcomes, [weak])
        rejected = decision.ranked_models[0]
        assert rejected.decision == "REJECTED"
        assert len(rejected.findings) == 6

    def test_comparison_explanation_for_top_two(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        best = _make_candidate("Best", metrics={"accuracy": 0.90, "recall_macro": 0.88}, gap=0.02)
        second = _make_candidate("Second", metrics={"accuracy": 0.82, "recall_macro": 0.70}, gap=0.10)
        survivors, gate_outcomes = engine.apply_hard_gates([best, second])
        decision = engine.rank(survivors, gate_outcomes, [best, second])
        assert decision.comparison_explanation is not None
        assert "Beat" in decision.comparison_explanation

    def test_no_comparison_when_single_model(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        only = _make_candidate("Only")
        decision = engine.rank([only], {}, [only])
        assert decision.comparison_explanation is None
