"""
RuleEngine: deterministic gating, scoring, and ranking of candidate ML models.

This is the authoritative source of truth for verdicts and selected_model.
The LLM layer is strictly additive and cannot override outcomes produced here.

Scoring formula:
    score = w_perf * norm_perf
          + w_overfit * (1 - norm_overfit_signal)
          + w_complex * (1 - norm_complexity)

where all three component values are in [0, 1].
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from .findings_engine import FindingsEngine
from .schemas import CandidateModel, DecisionTrace, JudgeDecision, RankedModel

logger = logging.getLogger(__name__)

# Map rule verdicts to governance-dashboard decision labels (no if-else ladder).
_VERDICT_DECISION_MAP: Dict[str, str] = {
    "select": "APPROVED",
    "rank_only": "RANKED",
    "reject": "REJECTED",
}


class RuleEngine:
    """Applies hard gates, scoring, and tie-break ranking to candidate models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._findings_engine = FindingsEngine(config)
        self._weights: Dict[str, float] = config["weights"]
        self._accuracy_floor: float = float(config["accuracy_floor"])
        self._r2_floor: float = float(config["r2_floor"])
        self._tie_break_pct: float = float(config["tie_break_pct"])
        self._gap_cap: float = float(config.get("overfitting_gap_cap", 0.5))
        self._complexity_norm: Dict[str, int] = config["complexity_normalization"]
        # Deterministic post-ranking selection (applied after the optional LLM
        # ranking step, never decided by the LLM itself).
        self._selection_top_pct: float = float(config["selection_top_pct"])
        self._selection_min_count: int = int(config["selection_min_count"])
        # Deterministic clamp on the LLM's reordering: SHAP/domain-reasoning
        # signal may only break ties among comparably-performing models, never
        # override a large predictive-quality gap.
        self._max_accuracy_drop: float = float(config["llm_ranking_max_accuracy_drop"])
        # Score deduction per governance flag (bias/shortcut/entropy/SHAP
        # leakage) -- demotes rank without rejecting the candidate.
        self._governance_flag_penalty: float = float(config["governance_flag_penalty"])
        # Map: task_type => floor value; avoids if-else ladder.
        self._floor_map: Dict[str, float] = {
            "classification": self._accuracy_floor,
            "regression": self._r2_floor,
        }
        # Map: task_type => primary metric name for scoring; from config.
        self._primary_metric_map: Dict[str, str] = config["primary_metric_map"]

    def _primary_perf(self, candidate: CandidateModel) -> Optional[float]:
        """Return the primary performance metric value for the candidate's task type."""
        primary_key = self._primary_metric_map.get(candidate.task_type)
        if primary_key is None:
            logger.warning(
                "=> Unknown task_type '%s' for model '%s'",
                candidate.task_type,
                candidate.model_name,
            )
            return None
        return candidate.metrics.get(primary_key)

    def apply_hard_gates(
        self, candidates: List[CandidateModel]
    ) -> Tuple[List[CandidateModel], Dict[str, str]]:
        """Reject any model whose primary metric is below the floor.

        This is the ONLY hard-reject gate. Bias/shortcut-learning/degenerate-
        behavior/SHAP-leakage concerns (see _governance_flags) are deliberately
        NOT rejection gates -- they only demote a model's rank via a score
        penalty in _compute_score. Rejecting on those secondary heuristics
        could legitimately zero out every candidate on a given dataset (e.g.
        an imbalanced dataset where most models trip the skew check) even
        though every candidate clears the real predictive-quality floor,
        starving the deterministic top-N% selection step of anything to
        select. The floor is the only signal serious enough to warrant never
        selecting a model outright.

        Returns:
            survivors: list of candidates that passed the floor.
            gate_outcomes: dict of model_name => rejection reason (for rejected models).
        """
        survivors: List[CandidateModel] = []
        gate_outcomes: Dict[str, str] = {}

        for candidate in candidates:
            perf = self._primary_perf(candidate)
            floor = self._floor_map.get(candidate.task_type)
            if floor is None:
                reason = f"Unknown task_type '{candidate.task_type}'; rejected by safety gate."
                gate_outcomes[candidate.model_name] = reason
                logger.debug("=> GATE REJECT %s: %s", candidate.model_name, reason)
                continue
            if perf is None:
                reason = (
                    f"Primary metric '{self._primary_metric_map.get(candidate.task_type)}' "
                    "is missing from metrics; rejected."
                )
                gate_outcomes[candidate.model_name] = reason
                logger.debug("=> GATE REJECT %s: %s", candidate.model_name, reason)
                continue
            if perf < floor:
                reason = (
                    f"Primary metric {perf:.4f} is below floor {floor:.4f} "
                    f"for task_type '{candidate.task_type}'."
                )
                gate_outcomes[candidate.model_name] = reason
                logger.debug("=> GATE REJECT %s: %s", candidate.model_name, reason)
                continue

            survivors.append(candidate)
            logger.debug(
                "=> GATE PASS %s: perf=%.4f >= floor=%.4f",
                candidate.model_name,
                perf,
                floor,
            )
        return survivors, gate_outcomes

    def _governance_flags(self, candidate: CandidateModel) -> List[str]:
        """Detect bias/shortcut-learning/degenerate-behavior/SHAP-leakage concerns.

        These NEVER reject a candidate (see apply_hard_gates docstring) --
        they only feed a score penalty in _compute_score, demoting a flagged
        model's rank (and therefore its chance of landing in the top-N%
        selection) without removing it from the eligible pool outright.
        """
        flags: List[str] = []
        perf = self._primary_perf(candidate) or 0.0

        if candidate.task_type == "classification" and candidate.overfitting.diagnostics:
            diag = candidate.overfitting.diagnostics

            pred_dist = diag.get("prediction_distribution") or {}
            for cls, fraction in pred_dist.items():
                if fraction >= 0.85:
                    flags.append(
                        f"Suspicious model bias: class '{cls}' dominates predictions ({fraction * 100:.1f}% of all outputs)."
                    )
                    break

            majority_baseline = diag.get("majority_baseline_accuracy")
            if majority_baseline is not None:
                improvement = perf - majority_baseline
                if improvement < 0.02:
                    flags.append(
                        f"Shortcut learning detected: model accuracy ({perf:.4f}) is similar to majority-class baseline ({majority_baseline:.4f}) with +{improvement * 100:.1f}% improvement (minimum +2.0% required)."
                    )

            entropy = diag.get("prediction_entropy")
            if entropy is not None and entropy < 0.15:
                flags.append(
                    f"Degenerate model behavior: prediction entropy is extremely low ({entropy:.4f}), indicating near-identical outputs and zero prediction diversity."
                )

        if candidate.shap_summary and "mean_abs_shap" in candidate.shap_summary:
            shap_dict = candidate.shap_summary["mean_abs_shap"]
            total_shap = sum(shap_dict.values())
            if total_shap > 0:
                top_feature_val = max(shap_dict.values())
                top_feature_name = [k for k, v in shap_dict.items() if v == top_feature_val][0]
                ratio = top_feature_val / total_shap
                if ratio >= 0.80:
                    flags.append(
                        f"Label leakage suspected: feature '{top_feature_name}' dominates importance ({ratio * 100:.1f}% of top-5 SHAP sum)."
                    )

        return flags

    def _normalize_complexity(
        self, candidates: List[CandidateModel]
    ) -> Dict[str, float]:
        """Min-max normalize complexity across surviving candidates.

        Complexity score is a weighted average of normalized n_params, depth,
        and family_rank, all capped at their configured maxima.
        """
        n_params_max = max(self._complexity_norm.get("n_params_max", 1), 1)
        depth_max = max(self._complexity_norm.get("depth_max", 1), 1)
        family_rank_max = max(self._complexity_norm.get("family_rank_max", 1), 1)

        norm_scores: Dict[str, float] = {}
        for candidate in candidates:
            comp = candidate.complexity
            norm_n_params = min(comp.n_params / n_params_max, 1.0)
            norm_depth = min(comp.depth / depth_max, 1.0)
            norm_family_rank = min(comp.family_rank / family_rank_max, 1.0)
            # Equal weight across the three complexity dimensions.
            norm_scores[candidate.model_name] = (
                norm_n_params + norm_depth + norm_family_rank
            ) / 3.0
        return norm_scores

    def _compute_score(
        self,
        candidate: CandidateModel,
        norm_complexity: float,
        governance_flag_count: int = 0,
    ) -> float:
        """Compute the weighted composite score for a single candidate.

        governance_flag_count demotes (never rejects) a model that tripped a
        bias/shortcut-learning/degenerate-behavior/SHAP-leakage check -- each
        flag subtracts governance_flag_penalty from the composite score,
        clamped at 0, so flagged models sort toward the bottom of the
        eligible pool without being removed from selection consideration.
        """
        perf = self._primary_perf(candidate) or 0.0
        # Overfitting signal: clip the gap to [0, gap_cap] then normalize.
        overfit_signal = min(max(candidate.overfitting.gap, 0.0), self._gap_cap)
        norm_overfit = overfit_signal / self._gap_cap

        weight_perf = self._weights.get("performance", 0.6)
        weight_overfit = self._weights.get("overfitting", 0.3)
        weight_complex = self._weights.get("complexity", 0.1)

        score = (
            weight_perf * perf
            + weight_overfit * (1.0 - norm_overfit)
            + weight_complex * (1.0 - norm_complexity)
        )
        if governance_flag_count > 0:
            score = max(0.0, score - self._governance_flag_penalty * governance_flag_count)
        logger.debug(
            "=> Score %s: perf=%.4f overfit_signal=%.4f norm_complex=%.4f governance_flags=%d => %.4f",
            candidate.model_name,
            perf,
            norm_overfit,
            norm_complexity,
            governance_flag_count,
            score,
        )
        return score

    def rank(
        self,
        survivors: List[CandidateModel],
        gate_outcomes: Dict[str, str],
        all_candidates: List[CandidateModel],
    ) -> JudgeDecision:
        """Score, tie-break, and rank survivors; build the JudgeDecision.

        Args:
            survivors: candidates that passed the hard gates.
            gate_outcomes: model_name => rejection reason for gated-out models.
            all_candidates: the complete original candidate list (for building output).

        Returns:
            A JudgeDecision with ranked_models and selected_model (rules authoritative).
        """
        norm_complexity_map = self._normalize_complexity(survivors)
        # Governance flags (bias/shortcut/entropy/SHAP leakage) demote score
        # but never reject -- computed once per survivor and reused below for
        # both scoring and the per-model reasons list.
        governance_flags_map: Dict[str, List[str]] = {
            candidate.model_name: self._governance_flags(candidate) for candidate in survivors
        }

        # Compute scores for survivors.
        scored: List[Tuple[float, CandidateModel]] = [
            (
                self._compute_score(
                    candidate,
                    norm_complexity_map[candidate.model_name],
                    governance_flag_count=len(governance_flags_map[candidate.model_name]),
                ),
                candidate,
            )
            for candidate in survivors
        ]

        # Sort descending by score; apply tie-break by complexity when within tie_break_pct.
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Tie-break pass: compare each adjacent pair and re-sort on complexity when tied.
        scored = self._apply_tie_break(scored, norm_complexity_map)

        ranked_models: List[RankedModel] = []
        selected_model: Optional[str] = None
        # Lookup so findings/explanations can reach the source candidate cheaply.
        candidate_by_name: Dict[str, CandidateModel] = {
            candidate.model_name: candidate for candidate in all_candidates
        }

        for rank_index, (score, candidate) in enumerate(scored, start=1):
            perf = self._primary_perf(candidate) or 0.0
            reasons = [
                f"Passed hard gate: primary metric={perf:.4f}",
                f"Overfitting gap={candidate.overfitting.gap:.4f}",
                f"Complexity family_rank={candidate.complexity.family_rank}",
            ]
            reasons.extend(governance_flags_map.get(candidate.model_name, []))
            # Build structured per-dimension findings + a ranking explanation.
            findings = self._findings_engine.build_findings(candidate)
            ranked_model = RankedModel(
                model_name=candidate.model_name,
                rank=rank_index,
                score=round(score, 6),
                verdict="select",
                reasons=reasons,
                llm_flags=[],
                decision=_VERDICT_DECISION_MAP["select"],
                findings=findings,
            )
            ranked_model = ranked_model.model_copy(
                update={
                    "ranking_explanation": self._findings_engine.build_ranking_explanation(
                        ranked_model=ranked_model,
                        candidate=candidate,
                        findings=findings,
                    )
                }
            )
            ranked_models.append(ranked_model)
            if rank_index == 1:
                selected_model = candidate.model_name

        # Append gated-out models at the bottom.
        gate_rank_start = len(ranked_models) + 1
        rejected_names = {candidate.model_name for candidate in all_candidates} - {
            candidate.model_name for candidate in survivors
        }
        for candidate in all_candidates:
            if candidate.model_name not in rejected_names:
                continue
            reason = gate_outcomes.get(candidate.model_name, "Rejected by hard gate.")
            # Rejected models still get full findings so the card explains *why*.
            findings = self._findings_engine.build_findings(candidate)
            ranked_models.append(
                RankedModel(
                    model_name=candidate.model_name,
                    rank=gate_rank_start,
                    score=0.0,
                    verdict="reject",
                    reasons=[reason],
                    llm_flags=[],
                    decision=_VERDICT_DECISION_MAP["reject"],
                    findings=findings,
                )
            )
            gate_rank_start += 1

        # Head-to-head comparison for the top two approved models.
        comparison_explanation = self._build_top_comparison(ranked_models, candidate_by_name)

        rule_outcomes = {
            "gate_outcomes": gate_outcomes,
            "scores": {rm.model_name: rm.score for rm in ranked_models},
        }
        logger.debug(
            "=> Ranking complete: selected=%s survivors=%d rejected=%d",
            selected_model,
            len(survivors),
            len(rejected_names),
        )
        return JudgeDecision(
            dataset_id=None,
            selected_model=selected_model,
            ranked_models=ranked_models,
            decision_trace=DecisionTrace(rule_outcomes=rule_outcomes, llm_commentary=None),
            comparison_explanation=comparison_explanation,
        )

    def enforce_accuracy_reorder_guardrail(
        self,
        decision: JudgeDecision,
        candidates: List[CandidateModel],
    ) -> JudgeDecision:
        """Clamp the LLM's reordering so it can never rank a model above
        another whose primary metric (accuracy/r2) is more than
        llm_ranking_max_accuracy_drop better.

        The LLM may use SHAP/domain-reasoning correlation to break ties among
        comparably-performing survivors, but it cannot override a large
        predictive-quality gap just because a weaker model's features look
        cleaner. Runs after the optional LLM ranking step and before
        deterministic selection -- so selection always sees a guardrail-
        respecting order regardless of whether the LLM ran.

        Implemented as a guarded bubble pass: repeatedly swap adjacent
        survivors whenever the one ranked below is more than
        llm_ranking_max_accuracy_drop better by primary metric than the one
        ranked above it. Terminates because each pass either makes no swaps
        (done) or strictly reduces the number of such inversions.
        """
        primary_by_name: Dict[str, float] = {
            candidate.model_name: (self._primary_perf(candidate) or 0.0)
            for candidate in candidates
        }
        survivors = [rm for rm in decision.ranked_models if rm.verdict != "reject"]
        rejected = [rm for rm in decision.ranked_models if rm.verdict == "reject"]
        if len(survivors) < 2:
            return decision

        ordered = list(survivors)
        swapped_any = False
        changed = True
        while changed:
            changed = False
            for index in range(len(ordered) - 1):
                current_metric = primary_by_name.get(ordered[index].model_name, 0.0)
                next_metric = primary_by_name.get(ordered[index + 1].model_name, 0.0)
                if next_metric - current_metric > self._max_accuracy_drop:
                    ordered[index], ordered[index + 1] = ordered[index + 1], ordered[index]
                    changed = True
                    swapped_any = True

        if not swapped_any:
            return decision

        logger.info(
            "=> [JUDGE] accuracy-reorder guardrail clamped LLM ranking (threshold=%.3f)",
            self._max_accuracy_drop,
        )
        renumbered_survivors = [
            ranked_model.model_copy(update={"rank": new_rank})
            for new_rank, ranked_model in enumerate(ordered, start=1)
        ]
        reject_rank_start = len(renumbered_survivors) + 1
        renumbered_rejected = [
            ranked_model.model_copy(update={"rank": reject_rank_start + offset})
            for offset, ranked_model in enumerate(rejected)
        ]
        return decision.model_copy(
            update={
                "ranked_models": renumbered_survivors + renumbered_rejected,
                "selected_model": (
                    renumbered_survivors[0].model_name if renumbered_survivors else decision.selected_model
                ),
            }
        )

    def apply_selection(self, decision: JudgeDecision) -> JudgeDecision:
        """Deterministically select the top-N% (min-count floor) of eligible models.

        Runs as the LAST step of JudgeAgent.judge(), after the optional LLM
        ranking step, so it always operates on whichever order is authoritative
        at that point (LLM-reordered survivors if ranking succeeded, rule-only
        order otherwise). This is plain Python -- the LLM never decides
        selection, only ranking.

        "Eligible" means verdict != "reject" (already passed the deterministic
        hard floor / bias / leakage gates in apply_hard_gates). Eligible models
        outside the top-N% cutoff become verdict="rank_only" (ranked, not
        selected) rather than "reject" -- they are not demoted to a quality
        failure, just excluded from the selected set.
        """
        eligible = [rm for rm in decision.ranked_models if rm.verdict != "reject"]
        eligible_count = len(eligible)
        selected_count = 0
        if eligible_count > 0:
            # ceil (not floor) biases toward inclusion at rounding boundaries,
            # consistent with the min-count floor's inclusive intent.
            selected_count = min(
                eligible_count,
                max(
                    self._selection_min_count,
                    math.ceil(self._selection_top_pct * eligible_count),
                ),
            )
        selected_names = {rm.model_name for rm in eligible[:selected_count]}

        updated_ranked: List[RankedModel] = []
        for ranked_model in decision.ranked_models:
            if ranked_model.verdict == "reject":
                updated_ranked.append(ranked_model)
                continue
            new_verdict = "select" if ranked_model.model_name in selected_names else "rank_only"
            updated_ranked.append(
                ranked_model.model_copy(
                    update={
                        "verdict": new_verdict,
                        "decision": _VERDICT_DECISION_MAP[new_verdict],
                    }
                )
            )

        selected_models = [
            ranked_model.model_name
            for ranked_model in updated_ranked
            if ranked_model.verdict == "select"
        ]
        logger.debug(
            "=> Selection complete: eligible=%d selected=%d/%s",
            eligible_count,
            selected_count,
            selected_models,
        )
        return decision.model_copy(
            update={
                "ranked_models": updated_ranked,
                "selected_models": selected_models,
                "selected_model": selected_models[0] if selected_models else None,
            }
        )

    def _build_top_comparison(
        self,
        ranked_models: List[RankedModel],
        candidate_by_name: Dict[str, CandidateModel],
    ) -> Optional[str]:
        """Build the 'Why A beat B' explanation for the top two approved models."""
        approved = [rm for rm in ranked_models if rm.verdict == "select"]
        if len(approved) < 2:
            return None
        winner_ranked, runner_ranked = approved[0], approved[1]
        winner_candidate = candidate_by_name.get(winner_ranked.model_name)
        runner_candidate = candidate_by_name.get(runner_ranked.model_name)
        if winner_candidate is None or runner_candidate is None:
            return None
        return self._findings_engine.build_comparison_explanation(
            winner_ranked=winner_ranked,
            winner_candidate=winner_candidate,
            runner_ranked=runner_ranked,
            runner_candidate=runner_candidate,
        )

    def _apply_tie_break(
        self,
        scored: List[Tuple[float, CandidateModel]],
        norm_complexity_map: Dict[str, float],
    ) -> List[Tuple[float, CandidateModel]]:
        """Within tie_break_pct performance, prefer the simpler model (stable sort)."""
        if len(scored) < 2:
            return scored

        # Use insertion sort to respect tie-break without disturbing non-tied pairs.
        result = list(scored)
        for outer_index in range(1, len(result)):
            current_score, current_candidate = result[outer_index]
            inner_index = outer_index
            while inner_index > 0:
                prev_score, prev_candidate = result[inner_index - 1]
                perf_diff = abs(prev_score - current_score)
                if perf_diff <= self._tie_break_pct:
                    # Within tie-break range: prefer lower norm_complexity.
                    current_complexity = norm_complexity_map.get(current_candidate.model_name, 1.0)
                    prev_complexity = norm_complexity_map.get(prev_candidate.model_name, 1.0)
                    if current_complexity < prev_complexity:
                        result[inner_index] = result[inner_index - 1]
                        result[inner_index - 1] = (current_score, current_candidate)
                        inner_index -= 1
                    else:
                        break
                else:
                    break
        return result
