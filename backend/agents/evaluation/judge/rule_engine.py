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
from typing import Any, Dict, List, Optional, Tuple

from .schemas import CandidateModel, DecisionTrace, JudgeDecision, RankedModel

logger = logging.getLogger(__name__)


class RuleEngine:
    """Applies hard gates, scoring, and tie-break ranking to candidate models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._weights: Dict[str, float] = config["weights"]
        self._accuracy_floor: float = float(config["accuracy_floor"])
        self._r2_floor: float = float(config["r2_floor"])
        self._tie_break_pct: float = float(config["tie_break_pct"])
        self._gap_cap: float = float(config.get("overfitting_gap_cap", 0.5))
        self._complexity_norm: Dict[str, int] = config["complexity_normalization"]
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
        """Reject any model whose primary metric is below the task-type floor.

        Returns:
            survivors: list of candidates that passed the gate.
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
            else:
                survivors.append(candidate)
                logger.debug(
                    "=> GATE PASS %s: perf=%.4f >= floor=%.4f",
                    candidate.model_name,
                    perf,
                    floor,
                )
        return survivors, gate_outcomes

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
    ) -> float:
        """Compute the weighted composite score for a single candidate."""
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
        logger.debug(
            "=> Score %s: perf=%.4f overfit_signal=%.4f norm_complex=%.4f => %.4f",
            candidate.model_name,
            perf,
            norm_overfit,
            norm_complexity,
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

        # Compute scores for survivors.
        scored: List[Tuple[float, CandidateModel]] = [
            (self._compute_score(candidate, norm_complexity_map[candidate.model_name]), candidate)
            for candidate in survivors
        ]

        # Sort descending by score; apply tie-break by complexity when within tie_break_pct.
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Tie-break pass: compare each adjacent pair and re-sort on complexity when tied.
        scored = self._apply_tie_break(scored, norm_complexity_map)

        ranked_models: List[RankedModel] = []
        selected_model: Optional[str] = None

        for rank_index, (score, candidate) in enumerate(scored, start=1):
            perf = self._primary_perf(candidate) or 0.0
            reasons = [
                f"Passed hard gate: primary metric={perf:.4f}",
                f"Overfitting gap={candidate.overfitting.gap:.4f}",
                f"Complexity family_rank={candidate.complexity.family_rank}",
            ]
            ranked_models.append(
                RankedModel(
                    model_name=candidate.model_name,
                    rank=rank_index,
                    score=round(score, 6),
                    verdict="select",
                    reasons=reasons,
                    llm_flags=[],
                )
            )
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
            ranked_models.append(
                RankedModel(
                    model_name=candidate.model_name,
                    rank=gate_rank_start,
                    score=0.0,
                    verdict="reject",
                    reasons=[reason],
                    llm_flags=[],
                )
            )
            gate_rank_start += 1

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
