import logging

from d2v_core.schema import RankedModelEntry, VerificationResult, VerificationSummary
from d2v_core.sweep import (
    METRIC_DIRECTION,
    CommonData,
    execute_model_trial,
    metrics_result_to_dict,
)

logger = logging.getLogger(__name__)


def run_verification_trial(
    model_name: str,
    hyperparameters: dict,
    common: CommonData,
    task_type: str,
    primary_metric: str,
    expected_metric: float,
    tolerance: float,
) -> VerificationResult:
    """ACTUALLY TRAINS model_name on the query dataset (via the single shared
    execute_model_trial core in d2v_core/sweep.py -- no duplicated training
    code between the Optuna sweep and this verification path) and compares the
    achieved primary_metric against the warm-start prior's expected_metric."""
    metrics = execute_model_trial(model_name, hyperparameters, common, task_type)
    achieved_metrics = metrics_result_to_dict(metrics)
    achieved_metric = achieved_metrics[primary_metric]
    delta_vs_expected = achieved_metric - expected_metric

    return VerificationResult(
        trained=True,
        achieved_metric=achieved_metric,
        achieved_full=achieved_metrics,
        delta_vs_expected=delta_vs_expected,
        within_tolerance=abs(delta_vs_expected) <= tolerance,
    )


class VerificationRunner:
    """Trains every ranked_models entry's suggested model + hyperparameters on
    the query dataset and attaches a VerificationResult to each entry, then
    rolls the results up into one VerificationSummary."""

    def __init__(self, common: CommonData, task_type: str, primary_metric: str, tolerance: float) -> None:
        self.common = common
        self.task_type = task_type
        self.primary_metric = primary_metric
        self.tolerance = tolerance

    def run(self, ranked_models: list[RankedModelEntry]) -> VerificationSummary:
        is_maximize = METRIC_DIRECTION[self.primary_metric] == "max"
        best_achieved_metrics: dict = None
        best_achieved_value: float = None
        absolute_deltas: list[float] = []
        n_within_tolerance = 0

        for ranked_entry in ranked_models:
            verification_result = run_verification_trial(
                model_name=ranked_entry.model_name,
                hyperparameters=ranked_entry.recommended_hyperparameters,
                common=self.common,
                task_type=self.task_type,
                primary_metric=self.primary_metric,
                expected_metric=ranked_entry.expected_metric,
                tolerance=self.tolerance,
            )
            ranked_entry.verification = verification_result
            logger.info(
                "=> verified model_name='%s': achieved=%.4f expected=%.4f delta=%.4f within_tolerance=%s.",
                ranked_entry.model_name,
                verification_result.achieved_metric,
                ranked_entry.expected_metric,
                verification_result.delta_vs_expected,
                verification_result.within_tolerance,
            )

            absolute_deltas.append(abs(verification_result.delta_vs_expected))
            if verification_result.within_tolerance:
                n_within_tolerance += 1

            is_new_best = best_achieved_value is None or (
                verification_result.achieved_metric > best_achieved_value
                if is_maximize
                else verification_result.achieved_metric < best_achieved_value
            )
            if is_new_best:
                best_achieved_value = verification_result.achieved_metric
                best_achieved_metrics = verification_result.achieved_full

        mean_abs_delta = sum(absolute_deltas) / len(absolute_deltas) if absolute_deltas else None
        return VerificationSummary(
            tolerance=self.tolerance,
            n_verified=len(ranked_models),
            n_within_tolerance=n_within_tolerance,
            best_achieved=best_achieved_metrics,
            mean_abs_delta=mean_abs_delta,
        )
