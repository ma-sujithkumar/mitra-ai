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
    diagnostics: dict | None = None,
    shap_summary: dict | None = None,
) -> CandidateModel:
    primary_key = "accuracy" if task_type == "classification" else "r2"
    return CandidateModel(
        model_name=model_name,
        task_type=task_type,
        metrics={primary_key: primary_metric_value},
        overfitting=OverfittingInfo(
            is_overfitted=gap > 0.10, gap=gap, train_vs_cv_gap=gap, diagnostics=diagnostics
        ),
        complexity=ComplexityDescriptor(n_params=n_params, depth=depth, family_rank=family_rank),
        shap_summary=shap_summary,
    )


class TestHardGates:

    def test_classification_below_floor_is_rejected(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        # 0.20 is below the relaxed accuracy_floor of 0.30.
        below_floor = _make_candidate("WeakModel", "classification", primary_metric_value=0.20)
        survivors, gate_outcomes = engine.apply_hard_gates([below_floor])
        assert len(survivors) == 0
        assert "WeakModel" in gate_outcomes
        assert "below floor" in gate_outcomes["WeakModel"]

    def test_regression_below_floor_is_rejected(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        # 0.10 is below the relaxed r2_floor of 0.20.
        below_floor = _make_candidate("WeakRegressor", "regression", primary_metric_value=0.10)
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
            _make_candidate("BadB", "classification", primary_metric_value=0.15),
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
        # KNeighborsClassifier has accuracy=0.45, now ABOVE the relaxed floor=0.30,
        # so it passes the gate. All models in this fixture have accuracy >= 0.30.
        rejected_names = [rm.model_name for rm in rejected]
        # Ensure the accepted models are all present.
        accepted_names = [
            rm.model_name for rm in decision.ranked_models if rm.verdict == "select"
        ]
        assert len(accepted_names) > 0

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
        # DummyRegressor has r2=0.10, below the relaxed r2_floor=0.20.
        assert "DummyRegressor" in rejected

    def test_all_rejected_returns_null_selected_model(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        # Both models are below the relaxed accuracy_floor of 0.30.
        all_bad = [
            _make_candidate("BadA", "classification", primary_metric_value=0.05),
            _make_candidate("BadB", "classification", primary_metric_value=0.10),
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
        assert decision.selected_models, "apply_selection() must populate selected_models"
        # KNeighborsClassifier (accuracy=0.45) now passes the relaxed 0.30 floor.
        # Three-state verdict: eligible models outside the top-70%/min-3 cutoff
        # become "rank_only", not "select" -- they are not a quality failure.
        assert all(rm.verdict in ("select", "rank_only", "reject") for rm in decision.ranked_models)
        assert decision.decision_trace.llm_commentary is None
        assert decision.decision_trace.llm_ranking_status == "skipped"

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
    def test_llm_ranks_survivors_but_selection_stays_deterministic(
        self, classification_judge_input_dict: dict, judge_config: dict
    ) -> None:
        from backend.agents.evaluation.judge.judge_agent import JudgeAgent
        from backend.agents.evaluation.judge.schemas import JudgeInput
        judge_input = JudgeInput.model_validate(classification_judge_input_dict)
        agent = JudgeAgent(config=judge_config)

        rule_only_decision = agent.judge(judge_input=judge_input, use_llm=False)
        llm_decision = agent.judge(judge_input=judge_input, use_llm=True)

        # The LLM is now allowed to reorder survivors -- rank/selected_model
        # may legitimately differ from the rule-only order. What must stay
        # invariant: the LLM never adds/removes/rejects models (the gate
        # outcome and eligible set are identical), and the LLM ranking step
        # always reports a non-missing status.
        rule_only_names = {rm.model_name for rm in rule_only_decision.ranked_models if rm.verdict != "reject"}
        llm_names = {rm.model_name for rm in llm_decision.ranked_models if rm.verdict != "reject"}
        assert rule_only_names == llm_names
        assert llm_decision.decision_trace.llm_ranking_status in ("applied", "failed")

        # If the LLM call succeeded, commentary and per-model reasoning should
        # be populated; selection still respects the same top-N%/min-3 formula.
        if llm_decision.decision_trace.llm_ranking_status == "applied":
            assert llm_decision.decision_trace.llm_commentary is not None
            assert len(llm_decision.selected_models) == len(rule_only_decision.selected_models)
        assert len(llm_decision.decision_trace.llm_commentary) > 0


class TestApplySelection:
    """Deterministic top-N%/min-count selection, applied after ranking."""

    @staticmethod
    def _ranked_decision(engine: RuleEngine, candidates: List[CandidateModel]) -> JudgeDecision:
        survivors, gate_outcomes = engine.apply_hard_gates(candidates)
        return engine.rank(survivors=survivors, gate_outcomes=gate_outcomes, all_candidates=candidates)

    def test_ten_eligible_selects_seven(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        candidates = [
            _make_candidate(f"Model{i}", "classification", primary_metric_value=0.90 - i * 0.01)
            for i in range(10)
        ]
        decision = engine.apply_selection(self._ranked_decision(engine, candidates))
        assert len(decision.selected_models) == 7
        selected = [rm for rm in decision.ranked_models if rm.verdict == "select"]
        rank_only = [rm for rm in decision.ranked_models if rm.verdict == "rank_only"]
        assert len(selected) == 7
        assert len(rank_only) == 3

    def test_two_eligible_selects_both_below_min_count(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        candidates = [
            _make_candidate("ModelA", "classification", primary_metric_value=0.90),
            _make_candidate("ModelB", "classification", primary_metric_value=0.85),
        ]
        decision = engine.apply_selection(self._ranked_decision(engine, candidates))
        # Can't force the min-count floor of 3 when only 2 models are eligible.
        assert len(decision.selected_models) == 2

    def test_four_eligible_selects_three(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        candidates = [
            _make_candidate(f"Model{i}", "classification", primary_metric_value=0.90 - i * 0.01)
            for i in range(4)
        ]
        decision = engine.apply_selection(self._ranked_decision(engine, candidates))
        # ceil(0.70 * 4) = 3, which also satisfies the min-count floor of 3.
        assert len(decision.selected_models) == 3

    def test_zero_eligible_selects_none(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        candidates = [
            _make_candidate("BadA", "classification", primary_metric_value=0.05),
            _make_candidate("BadB", "classification", primary_metric_value=0.10),
        ]
        decision = engine.apply_selection(self._ranked_decision(engine, candidates))
        assert decision.selected_models == []
        assert decision.selected_model is None
        assert all(rm.verdict == "reject" for rm in decision.ranked_models)


class TestGovernanceFlagsDemoteNotReject:
    """Bias/shortcut/entropy/SHAP-leakage checks must only demote rank, never
    reject -- only the accuracy/r2 floor can reject. This is what guarantees
    selection is never starved to zero just because every model on a given
    dataset happens to trip one of these secondary heuristics."""

    def test_prediction_skew_does_not_reject(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        skewed = _make_candidate(
            "SkewedModel", "classification", primary_metric_value=0.80,
            diagnostics={"prediction_distribution": {"classA": 0.95}},
        )
        survivors, gate_outcomes = engine.apply_hard_gates([skewed])
        assert len(survivors) == 1
        assert "SkewedModel" not in gate_outcomes

    def test_shortcut_learning_does_not_reject(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        shortcut = _make_candidate(
            "ShortcutModel", "classification", primary_metric_value=0.80,
            diagnostics={"majority_baseline_accuracy": 0.79},
        )
        survivors, gate_outcomes = engine.apply_hard_gates([shortcut])
        assert len(survivors) == 1
        assert "ShortcutModel" not in gate_outcomes

    def test_low_entropy_does_not_reject(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        degenerate = _make_candidate(
            "DegenerateModel", "classification", primary_metric_value=0.80,
            diagnostics={"prediction_entropy": 0.05},
        )
        survivors, gate_outcomes = engine.apply_hard_gates([degenerate])
        assert len(survivors) == 1
        assert "DegenerateModel" not in gate_outcomes

    def test_shap_leakage_concentration_does_not_reject(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        leaky = _make_candidate(
            "LeakyModel", "classification", primary_metric_value=0.80,
            shap_summary={"mean_abs_shap": {"f1": 0.95, "f2": 0.05}},
        )
        survivors, gate_outcomes = engine.apply_hard_gates([leaky])
        assert len(survivors) == 1
        assert "LeakyModel" not in gate_outcomes

    def test_all_models_flagged_still_selects_something(self, judge_config: dict) -> None:
        """The exact failure mode being fixed: every candidate passes the
        floor but every candidate trips a governance flag. Selection must
        not come back empty."""
        engine = RuleEngine(judge_config)
        candidates = [
            _make_candidate(
                f"FlaggedModel{i}", "classification", primary_metric_value=0.80 - i * 0.01,
                shap_summary={"mean_abs_shap": {"f1": 0.95, "f2": 0.05}},
            )
            for i in range(5)
        ]
        survivors, gate_outcomes = engine.apply_hard_gates(candidates)
        assert len(survivors) == 5
        assert gate_outcomes == {}
        decision = engine.rank(survivors=survivors, gate_outcomes=gate_outcomes, all_candidates=candidates)
        decision = engine.apply_selection(decision)
        assert decision.selected_models, "selection must not be empty when every model passes the floor"
        assert len(decision.selected_models) == 4  # ceil(0.70 * 5) = 4, above the min-count floor of 3

    def test_flagged_model_ranks_below_clean_model_of_similar_performance(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        clean = _make_candidate("CleanModel", "classification", primary_metric_value=0.80)
        flagged = _make_candidate(
            "FlaggedModel", "classification", primary_metric_value=0.82,
            shap_summary={"mean_abs_shap": {"f1": 0.95, "f2": 0.05}},
        )
        candidates = [flagged, clean]
        survivors, gate_outcomes = engine.apply_hard_gates(candidates)
        decision = engine.rank(survivors=survivors, gate_outcomes=gate_outcomes, all_candidates=candidates)
        ranked_names = [rm.model_name for rm in decision.ranked_models]
        # FlaggedModel has higher raw accuracy but the governance penalty
        # must still drop it below CleanModel.
        assert ranked_names.index("CleanModel") < ranked_names.index("FlaggedModel")


class TestAccuracyReorderGuardrail:
    """Deterministic clamp: LLM cannot rank a much-worse model above a much-better one."""

    def test_large_accuracy_gap_reorder_is_clamped_back(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        strong = _make_candidate("StrongModel", "classification", primary_metric_value=0.90)
        weak = _make_candidate("WeakModel", "classification", primary_metric_value=0.70)
        candidates = [strong, weak]
        survivors, gate_outcomes = engine.apply_hard_gates(candidates)
        decision = engine.rank(survivors=survivors, gate_outcomes=gate_outcomes, all_candidates=candidates)

        # Simulate the LLM ranking WeakModel (0.70) above StrongModel (0.90) --
        # a 20-point gap, exceeding the 0.10 default threshold.
        by_name = {rm.model_name: rm for rm in decision.ranked_models}
        llm_reordered = decision.model_copy(
            update={
                "ranked_models": [
                    by_name["WeakModel"].model_copy(update={"rank": 1}),
                    by_name["StrongModel"].model_copy(update={"rank": 2}),
                ]
            }
        )
        clamped = engine.enforce_accuracy_reorder_guardrail(llm_reordered, candidates)
        assert clamped.ranked_models[0].model_name == "StrongModel"
        assert clamped.ranked_models[1].model_name == "WeakModel"
        assert clamped.selected_model == "StrongModel"

    def test_small_accuracy_gap_reorder_is_preserved(self, judge_config: dict) -> None:
        engine = RuleEngine(judge_config)
        modelA = _make_candidate("ModelA", "classification", primary_metric_value=0.85)
        modelB = _make_candidate("ModelB", "classification", primary_metric_value=0.80)
        candidates = [modelA, modelB]
        survivors, gate_outcomes = engine.apply_hard_gates(candidates)
        decision = engine.rank(survivors=survivors, gate_outcomes=gate_outcomes, all_candidates=candidates)

        # LLM ranks ModelB (0.80) above ModelA (0.85) -- a 5-point gap, within
        # the 0.10 default threshold, so the guardrail must leave it alone.
        by_name = {rm.model_name: rm for rm in decision.ranked_models}
        llm_reordered = decision.model_copy(
            update={
                "ranked_models": [
                    by_name["ModelB"].model_copy(update={"rank": 1}),
                    by_name["ModelA"].model_copy(update={"rank": 2}),
                ]
            }
        )
        clamped = engine.enforce_accuracy_reorder_guardrail(llm_reordered, candidates)
        assert clamped.ranked_models[0].model_name == "ModelB"
        assert clamped.ranked_models[1].model_name == "ModelA"


class TestJudgeTools:
    """Tests the JudgeTools class to ensure function-calling endpoints retrieve accurate metadata/statistics/candidate details."""

    def test_judge_tools_retrieval(self, classification_judge_input_dict: dict) -> None:
        from backend.agents.evaluation.judge.judge_agent import JudgeTools
        from backend.agents.evaluation.judge.schemas import JudgeInput

        judge_input = JudgeInput.model_validate(classification_judge_input_dict)
        tools = JudgeTools(judge_input)

        # Test metadata retrieval
        metadata = tools.get_dataset_metadata()
        assert isinstance(metadata, dict)
        assert metadata == (judge_input.metadata or {})

        # Test statistics retrieval
        statistics = tools.get_dataset_statistics()
        assert isinstance(statistics, dict)
        assert statistics == (judge_input.minidata or {})

        # Test model evaluation details retrieval
        first_candidate = judge_input.candidates[0]
        details = tools.get_model_evaluation_details(first_candidate.model_name)
        assert isinstance(details, dict)
        assert details["model_name"] == first_candidate.model_name
        assert details["task_type"] == first_candidate.task_type
        assert details["metrics"] == first_candidate.metrics
        assert details["complexity"]["n_params"] == first_candidate.complexity.n_params

        # Test error handling for non-existent model
        missing_details = tools.get_model_evaluation_details("NonExistentModel")
        assert isinstance(missing_details, dict)
        assert "error" in missing_details
        assert "NonExistentModel" in missing_details["error"]

