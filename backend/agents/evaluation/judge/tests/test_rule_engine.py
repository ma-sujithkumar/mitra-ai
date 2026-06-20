"""
Tests for rule_engine.py: hard gates, scoring, tie-break, and ranking.

All tests run without network or LLM access (rule-only path).
"""

import os
import sys
from typing import List

import pytest

from backend.agents.evaluation.judge.schemas import CandidateModel, ComplexityDescriptor, JudgeDecision, JudgeInput, OverfittingInfo
from backend.agents.evaluation.judge.rule_engine import RuleEngine


def _make_candidate(
    model_name: str,
    task_type: str,
    primary_metric_value: float,
    gap: float = 0.05,
    family_rank: int = 5,
    n_params: int = 1000,
    depth: int = 5,
) -> CandidateModel:
    primary_key = "accuracy" if task_type == "classification" else "r2"
    return CandidateModel(
        model_name=model_name,
        task_type=task_type,
        metrics={primary_key: primary_metric_value},
        overfitting=OverfittingInfo(is_overfitted=gap > 0.10, gap=gap, train_vs_cv_gap=gap),
        complexity=ComplexityDescriptor(n_params=n_params, depth=depth, family_rank=family_rank),
    )


class TestHardGates:

    def test_classification_below_floor_is_rejected(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        below_floor = _make_candidate("WeakModel", "classification", primary_metric_value=0.45)
        survivors, gate_outcomes = engine.apply_hard_gates([below_floor])
        assert len(survivors) == 0
        assert "WeakModel" in gate_outcomes
        assert "below floor" in gate_outcomes["WeakModel"]

    def test_regression_below_floor_is_rejected(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        below_floor = _make_candidate("WeakRegressor", "regression", primary_metric_value=0.20)
        survivors, gate_outcomes = engine.apply_hard_gates([below_floor])
        assert len(survivors) == 0
        assert "WeakRegressor" in gate_outcomes

    def test_above_floor_passes(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        good_model = _make_candidate("GoodModel", "classification", primary_metric_value=0.85)
        survivors, gate_outcomes = engine.apply_hard_gates([good_model])
        assert len(survivors) == 1
        assert "GoodModel" not in gate_outcomes

    def test_mixed_batch_gates_only_bad(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        candidates = [
            _make_candidate("GoodA", "classification", primary_metric_value=0.88),
            _make_candidate("BadB", "classification", primary_metric_value=0.40),
            _make_candidate("GoodC", "classification", primary_metric_value=0.75),
        ]
        survivors, gate_outcomes = engine.apply_hard_gates(candidates)
        survivor_names = [candidate.model_name for candidate in survivors]
        assert "GoodA" in survivor_names
        assert "GoodC" in survivor_names
        assert "BadB" not in survivor_names
        assert "BadB" in gate_outcomes


class TestTieBreak:

    def test_tie_break_prefers_simpler_model(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        # Two models within 1% performance; model_simple has lower family_rank.
        model_complex = _make_candidate(
            "ModelComplex", "classification",
            primary_metric_value=0.900, gap=0.05,
            family_rank=9, n_params=500000, depth=20,
        )
        model_simple = _make_candidate(
            "ModelSimple", "classification",
            primary_metric_value=0.905, gap=0.05,
            family_rank=2, n_params=200, depth=2,
        )
        candidates = [model_complex, model_simple]
        survivors, gate_outcomes = engine.apply_hard_gates(candidates)
        decision = engine.rank(
            survivors=survivors,
            gate_outcomes=gate_outcomes,
            all_candidates=candidates,
        )
        ranked_names = [rm.model_name for rm in decision.ranked_models if rm.verdict == "select"]
        assert ranked_names[0] == "ModelSimple", (
            "Simpler model should rank first when performance differs by <= tie_break_pct"
        )

    def test_no_tie_break_when_diff_exceeds_threshold(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        model_high = _make_candidate(
            "HighPerf", "classification",
            primary_metric_value=0.95, gap=0.05, family_rank=9,
        )
        model_low = _make_candidate(
            "LowPerf", "classification",
            primary_metric_value=0.80, gap=0.02, family_rank=1,
        )
        candidates = [model_high, model_low]
        survivors, gate_outcomes = engine.apply_hard_gates(candidates)
        decision = engine.rank(
            survivors=survivors,
            gate_outcomes=gate_outcomes,
            all_candidates=candidates,
        )
        selected_names = [rm.model_name for rm in decision.ranked_models if rm.verdict == "select"]
        assert selected_names[0] == "HighPerf", (
            "Higher-performing model should rank first when diff exceeds tie_break_pct"
        )


class TestRankingWithClassificationFixture:

    def test_classification_decision_schema_valid(
        self, classification_judge_input_dict: dict, judge_config: dict
    ) -> None:
        from backend.agents.evaluation.judge.schemas import JudgeInput, JudgeDecision
        judge_input = JudgeInput.model_validate(classification_judge_input_dict)
        engine = RuleEngine(judge_config)
        survivors, gate_outcomes = engine.apply_hard_gates(judge_input.candidates)
        decision = engine.rank(
            survivors=survivors,
            gate_outcomes=gate_outcomes,
            all_candidates=judge_input.candidates,
        )
        assert isinstance(decision, JudgeDecision)
        assert len(decision.ranked_models) == len(judge_input.candidates)
        rejected = [rm for rm in decision.ranked_models if rm.verdict == "reject"]
        # KNeighborsClassifier has accuracy=0.45, below floor=0.60; must be rejected.
        rejected_names = [rm.model_name for rm in rejected]
        assert "KNeighborsClassifier" in rejected_names

    def test_regression_below_floor_model_rejected(
        self, regression_judge_input_dict: dict, judge_config: dict
    ) -> None:
        from backend.agents.evaluation.judge.schemas import JudgeInput
        judge_input = JudgeInput.model_validate(regression_judge_input_dict)
        engine = RuleEngine(judge_config)
        survivors, gate_outcomes = engine.apply_hard_gates(judge_input.candidates)
        decision = engine.rank(
            survivors=survivors,
            gate_outcomes=gate_outcomes,
            all_candidates=judge_input.candidates,
        )
        rejected = [rm.model_name for rm in decision.ranked_models if rm.verdict == "reject"]
        # DummyRegressor has r2=0.10, below r2_floor=0.40.
        assert "DummyRegressor" in rejected

    def test_all_rejected_returns_null_selected_model(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        all_bad = [
            _make_candidate("BadA", "classification", primary_metric_value=0.10),
            _make_candidate("BadB", "classification", primary_metric_value=0.20),
        ]
        survivors, gate_outcomes = engine.apply_hard_gates(all_bad)
        decision = engine.rank(
            survivors=survivors,
            gate_outcomes=gate_outcomes,
            all_candidates=all_bad,
        )
        assert decision.selected_model is None
        for rm in decision.ranked_models:
            assert rm.verdict == "reject"


class TestJudgeAgentRuleOnly:
    """End-to-end rule-only test: no LLM, no network required."""

    def test_classification_end_to_end_rule_only(
        self, classification_judge_input_dict: dict, judge_config: dict
    ) -> None:
        from backend.agents.evaluation.judge.judge_agent import JudgeAgent
        from backend.agents.evaluation.judge.schemas import JudgeInput, JudgeDecision
        judge_input = JudgeInput.model_validate(classification_judge_input_dict)
        agent = JudgeAgent(config=judge_config)
        decision = agent.judge(judge_input=judge_input, use_llm=False)
        assert isinstance(decision, JudgeDecision)
        assert decision.selected_model is not None
        assert decision.selected_model != "KNeighborsClassifier"
        assert all(rm.verdict in ("select", "reject") for rm in decision.ranked_models)
        assert decision.decision_trace.llm_commentary is None

    def test_regression_end_to_end_rule_only(
        self, regression_judge_input_dict: dict, judge_config: dict
    ) -> None:
        from backend.agents.evaluation.judge.judge_agent import JudgeAgent
        from backend.agents.evaluation.judge.schemas import JudgeInput, JudgeDecision
        judge_input = JudgeInput.model_validate(regression_judge_input_dict)
        agent = JudgeAgent(config=judge_config)
        decision = agent.judge(judge_input=judge_input, use_llm=False)
        assert isinstance(decision, JudgeDecision)
        rejected = [rm.model_name for rm in decision.ranked_models if rm.verdict == "reject"]
        assert "DummyRegressor" in rejected
        selected_ranks = [rm for rm in decision.ranked_models if rm.verdict == "select"]
        assert len(selected_ranks) > 0


class TestJudgeAgentLiveLlm:
    """Live LLM test: skipped unless CLAUDE_CLI_PATH and ANTHROPIC_MODEL_NAME are set."""

    @pytest.mark.skipif(
        not (os.environ.get("CLAUDE_CLI_PATH") and os.environ.get("ANTHROPIC_MODEL_NAME")),
        reason="CLAUDE_CLI_PATH and ANTHROPIC_MODEL_NAME must be set for live LLM test.",
    )
    def test_llm_enrichment_does_not_change_verdict(
        self, classification_judge_input_dict: dict, judge_config: dict
    ) -> None:
        from backend.agents.evaluation.judge.judge_agent import JudgeAgent
        from backend.agents.evaluation.judge.schemas import JudgeInput
        judge_input = JudgeInput.model_validate(classification_judge_input_dict)
        agent = JudgeAgent(config=judge_config)

        rule_only_decision = agent.judge(judge_input=judge_input, use_llm=False)
        llm_decision = agent.judge(judge_input=judge_input, use_llm=True)

        # Verdicts and selected_model must be unchanged by LLM.
        assert rule_only_decision.selected_model == llm_decision.selected_model
        for rule_rm, llm_rm in zip(rule_only_decision.ranked_models, llm_decision.ranked_models):
            assert rule_rm.verdict == llm_rm.verdict
            assert rule_rm.rank == llm_rm.rank

        # LLM commentary should be populated.
        assert llm_decision.decision_trace.llm_commentary is not None
        assert len(llm_decision.decision_trace.llm_commentary) > 0
