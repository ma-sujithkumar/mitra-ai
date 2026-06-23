"""
FindingsEngine: deterministic, per-dimension Judge findings for the model
governance dashboard.

For every candidate the engine emits one structured Finding per mandatory
evaluation dimension (predictive quality, class balance, generalization,
feature utilization, data leakage, robustness). It also synthesizes the
per-model ranking explanation ("Why ranked #N") and the head-to-head
comparison explanation ("Why A beat B").

Everything here is derived deterministically from the CandidateModel inputs the
rule engine already has (metrics, overfitting, SHAP), so the findings are fully
reproducible and unit-testable. The LLM layer remains strictly additive on top.

Dimension dispatch is a hash-map (dimension key => bound method) loaded from
config, avoiding any if-else ladder over dimension names. All thresholds live in
config.yaml under the `findings` section.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .schemas import CandidateModel, Finding, RankedModel

logger = logging.getLogger(__name__)

# Finding status constants (also consumed by the frontend marker map).
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_INFO = "info"


class FindingsEngine:
    """Builds structured Judge findings + ranking/comparison explanations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        findings_config: Dict[str, Any] = config.get("findings", {}) or {}
        self._dimension_labels: Dict[str, str] = findings_config.get("dimension_labels", {})
        self._classification_metrics: List[str] = findings_config.get(
            "classification_report_metrics", []
        )
        self._regression_metrics: List[str] = findings_config.get(
            "regression_report_metrics", []
        )
        self._macro_recall_floor: float = float(findings_config.get("macro_recall_floor", 0.5))
        self._f1_spread_warn: float = float(findings_config.get("f1_spread_warn", 0.15))
        self._overfit_gap_warn: float = float(findings_config.get("overfit_gap_warn", 0.1))
        self._dominant_feature_share: float = float(
            findings_config.get("dominant_feature_share", 0.6)
        )
        self._leakage_train_perfect: float = float(
            findings_config.get("leakage_train_perfect", 0.999)
        )
        self._leakage_gap_warn: float = float(findings_config.get("leakage_gap_warn", 0.2))
        self._cv_std_unstable: float = float(findings_config.get("cv_std_unstable", 0.05))

        # Reuse the rule-engine floors/primary-metric map for consistency.
        self._floor_map: Dict[str, float] = {
            "classification": float(config.get("accuracy_floor", 0.6)),
            "regression": float(config.get("r2_floor", 0.4)),
        }
        self._primary_metric_map: Dict[str, str] = config.get("primary_metric_map", {})

        # Dimension dispatch table: dimension key => builder method. Ordered by
        # the dimension_labels map so the card always renders dimensions in order.
        self._dimension_builders: Dict[str, Callable[[CandidateModel], Finding]] = {
            "predictive_quality": self._predictive_quality,
            "class_balance": self._class_balance,
            "generalization": self._generalization,
            "feature_utilization": self._feature_utilization,
            "data_leakage": self._data_leakage,
            "robustness": self._robustness,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build_findings(self, candidate: CandidateModel) -> List[Finding]:
        """Build the ordered list of per-dimension findings for one candidate."""
        findings: List[Finding] = []
        for dimension_key in self._dimension_labels:
            builder = self._dimension_builders.get(dimension_key)
            if builder is None:
                logger.debug("=> No builder registered for dimension '%s'", dimension_key)
                continue
            findings.append(builder(candidate))
        return findings

    def build_ranking_explanation(
        self,
        ranked_model: RankedModel,
        candidate: CandidateModel,
        findings: List[Finding],
    ) -> str:
        """Synthesize a 'Why Ranked #N' explanation from the model's findings."""
        header = f"Why Ranked #{ranked_model.rank}"
        primary_value = self._primary_perf(candidate)
        lines: List[str] = [
            f"Composite judge score of {ranked_model.score:.4f} "
            f"(primary metric {self._fmt(candidate.task_type, primary_value)})."
        ]
        # Surface every strength (pass) finding as a ranking justification.
        lines.extend(
            f"{finding.label}: {finding.message}"
            for finding in findings
            if finding.status == STATUS_PASS
        )
        # Surface any residual concern so the ranking is not misleadingly clean.
        lines.extend(
            f"Caveat -- {finding.label}: {finding.message}"
            for finding in findings
            if finding.status == STATUS_FAIL
        )
        return header + "\n" + "\n".join(f"- {line}" for line in lines)

    def build_comparison_explanation(
        self,
        winner_ranked: RankedModel,
        winner_candidate: CandidateModel,
        runner_ranked: RankedModel,
        runner_candidate: CandidateModel,
    ) -> str:
        """Explain why the winner was preferred over the runner-up."""
        header = f"Why {winner_ranked.model_name} Beat {runner_ranked.model_name}"
        task_type = winner_candidate.task_type
        lines: List[str] = [
            f"Higher composite judge score "
            f"({winner_ranked.score:.4f} vs {runner_ranked.score:.4f})."
        ]

        # Compare primary performance metric.
        winner_primary = self._primary_perf(winner_candidate)
        runner_primary = self._primary_perf(runner_candidate)
        primary_key = self._primary_metric_map.get(task_type, "metric")
        if winner_primary is not None and runner_primary is not None:
            direction = "higher" if winner_primary >= runner_primary else "comparable"
            lines.append(
                f"{direction.capitalize()} {primary_key} "
                f"({self._fmt(task_type, winner_primary)} vs {self._fmt(task_type, runner_primary)})."
            )

        # Compare overfitting gap (lower is better).
        winner_gap = winner_candidate.overfitting.gap
        runner_gap = runner_candidate.overfitting.gap
        if winner_gap < runner_gap:
            lines.append(
                f"Lower overfitting gap "
                f"({self._fmt_gap(task_type, winner_gap)} vs {self._fmt_gap(task_type, runner_gap)})."
            )

        # Compare minority-class recall when available (classification only).
        winner_recall = winner_candidate.metrics.get("recall_macro")
        runner_recall = runner_candidate.metrics.get("recall_macro")
        if winner_recall is not None and runner_recall is not None and winner_recall > runner_recall:
            lines.append(
                f"Better minority-class recall "
                f"({self._fmt(task_type, winner_recall)} vs {self._fmt(task_type, runner_recall)})."
            )

        # Compare feature diversity via SHAP concentration when available.
        winner_div = self._feature_diversity(winner_candidate)
        runner_div = self._feature_diversity(runner_candidate)
        if winner_div is not None and runner_div is not None and winner_div > runner_div:
            lines.append("Better feature diversity (less reliance on a single dominant feature).")

        return header + "\n" + "\n".join(f"- {line}" for line in lines)

    # ------------------------------------------------------------------ #
    # Per-dimension builders
    # ------------------------------------------------------------------ #
    def _predictive_quality(self, candidate: CandidateModel) -> Finding:
        label = self._dimension_labels.get("predictive_quality", "Predictive Quality")
        primary_value = self._primary_perf(candidate)
        floor = self._floor_map.get(candidate.task_type)
        report_keys = (
            self._classification_metrics
            if candidate.task_type == "classification"
            else self._regression_metrics
        )
        reported = [
            f"{metric_key}={self._fmt(candidate.task_type, candidate.metrics.get(metric_key))}"
            for metric_key in report_keys
            if candidate.metrics.get(metric_key) is not None
        ]
        reported_text = ", ".join(reported) if reported else "no metrics reported"

        passed = (
            primary_value is not None and floor is not None and primary_value >= floor
        )
        if passed:
            message = f"Meets predictive-quality floor ({reported_text})."
        elif primary_value is None or floor is None:
            message = f"Primary metric unavailable; cannot confirm predictive quality ({reported_text})."
        else:
            message = (
                f"Primary metric {self._fmt(candidate.task_type, primary_value)} below floor "
                f"{self._fmt(candidate.task_type, floor)} ({reported_text})."
            )
        return self._finding("predictive_quality", label, passed, message)

    def _class_balance(self, candidate: CandidateModel) -> Finding:
        label = self._dimension_labels.get("class_balance", "Class Balance")
        if candidate.task_type != "classification":
            return Finding(
                dimension="class_balance",
                label=label,
                status=STATUS_INFO,
                message="Class balance not applicable to regression tasks.",
            )

        recall_macro = candidate.metrics.get("recall_macro")
        f1_macro = candidate.metrics.get("f1_macro")
        f1_weighted = candidate.metrics.get("f1_weighted")

        if recall_macro is None and f1_macro is None:
            return Finding(
                dimension="class_balance",
                label=label,
                status=STATUS_INFO,
                message="Class-balance metrics (macro recall / F1) unavailable.",
            )

        # Weak minority-class recall is the strongest negative signal.
        if recall_macro is not None and recall_macro < self._macro_recall_floor:
            message = (
                f"Weak minority-class recall (macro recall="
                f"{self._fmt(candidate.task_type, recall_macro)})."
            )
            return self._finding("class_balance", label, False, message)

        # Large gap between weighted and macro F1 indicates class prediction skew.
        if f1_macro is not None and f1_weighted is not None:
            spread = abs(f1_weighted - f1_macro)
            if spread > self._f1_spread_warn:
                message = (
                    f"Class prediction skew: weighted-vs-macro F1 spread="
                    f"{self._fmt_gap(candidate.task_type, spread)}."
                )
                return self._finding("class_balance", label, False, message)

        message = (
            f"Balanced class-wise performance (macro recall="
            f"{self._fmt(candidate.task_type, recall_macro)})."
        )
        return self._finding("class_balance", label, True, message)

    def _generalization(self, candidate: CandidateModel) -> Finding:
        label = self._dimension_labels.get("generalization", "Generalization")
        overfitting = candidate.overfitting
        gap = overfitting.gap
        cv_gap = overfitting.train_vs_cv_gap
        cv_gap_text = (
            f", train-vs-CV gap={self._fmt_gap(candidate.task_type, cv_gap)}"
            if cv_gap is not None
            else ""
        )
        is_overfitted = bool(overfitting.is_overfitted) or gap > self._overfit_gap_warn
        if is_overfitted:
            message = (
                f"Overfitting detected: train-test gap="
                f"{self._fmt_gap(candidate.task_type, gap)}{cv_gap_text}."
            )
            return self._finding("generalization", label, False, message)
        message = (
            f"Generalizes well: train-test gap="
            f"{self._fmt_gap(candidate.task_type, gap)}{cv_gap_text}."
        )
        return self._finding("generalization", label, True, message)

    def _feature_utilization(self, candidate: CandidateModel) -> Finding:
        label = self._dimension_labels.get("feature_utilization", "Feature Utilization")
        shap_summary = candidate.shap_summary
        if not shap_summary:
            return Finding(
                dimension="feature_utilization",
                label=label,
                status=STATUS_INFO,
                message="SHAP feature-importance analysis unavailable.",
            )

        top_features: List[str] = shap_summary.get("top_features", []) or []
        mean_abs_shap: Dict[str, float] = shap_summary.get("mean_abs_shap", {}) or {}
        dominant_share = self._dominant_feature_share_value(mean_abs_shap)

        if dominant_share is not None and dominant_share >= self._dominant_feature_share:
            dominant_feature = top_features[0] if top_features else "a single feature"
            message = (
                f"Feature importance concentrated on '{dominant_feature}' "
                f"(share={dominant_share * 100:.1f}% of top features) -- possible shortcut."
            )
            return self._finding("feature_utilization", label, False, message)

        feature_count = len(top_features)
        share_text = (
            f"top-feature share={dominant_share * 100:.1f}%"
            if dominant_share is not None
            else "balanced importances"
        )
        message = f"Strong feature utilization across {feature_count} features ({share_text})."
        return self._finding("feature_utilization", label, True, message)

    def _data_leakage(self, candidate: CandidateModel) -> Finding:
        label = self._dimension_labels.get("data_leakage", "Data Leakage")
        train_metrics = candidate.overfitting.train_metrics or {}
        primary_key = self._primary_metric_map.get(candidate.task_type)
        train_primary = train_metrics.get(primary_key) if primary_key else None
        gap = candidate.overfitting.gap

        suspicious = (
            train_primary is not None
            and train_primary >= self._leakage_train_perfect
            and gap >= self._leakage_gap_warn
        )
        if suspicious:
            message = (
                f"Potential shortcut learning: near-perfect train {primary_key}="
                f"{self._fmt(candidate.task_type, train_primary)} with large gap "
                f"{self._fmt_gap(candidate.task_type, gap)}."
            )
            return self._finding("data_leakage", label, False, message)
        message = "No evidence of label leakage or shortcut learning."
        return self._finding("data_leakage", label, True, message)

    def _robustness(self, candidate: CandidateModel) -> Finding:
        label = self._dimension_labels.get("robustness", "Robustness")
        cv_results = candidate.overfitting.cv_results or {}
        cv_std = cv_results.get("std")
        cv_mean = cv_results.get("mean")
        if cv_std is None:
            return Finding(
                dimension="robustness",
                label=label,
                status=STATUS_INFO,
                message="Cross-validation stability not assessed (K-fold skipped).",
            )
        if cv_std > self._cv_std_unstable:
            message = f"Unstable across folds (CV std={cv_std:.4f})."
            return self._finding("robustness", label, False, message)
        mean_text = f", mean={cv_mean:.4f}" if cv_mean is not None else ""
        message = f"Stable across folds (CV std={cv_std:.4f}{mean_text})."
        return self._finding("robustness", label, True, message)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _primary_perf(self, candidate: CandidateModel) -> Optional[float]:
        """Return the primary performance metric value for the candidate."""
        primary_key = self._primary_metric_map.get(candidate.task_type)
        if primary_key is None:
            return None
        return candidate.metrics.get(primary_key)

    def _feature_diversity(self, candidate: CandidateModel) -> Optional[float]:
        """Higher value => more diverse feature usage (lower dominant share)."""
        if not candidate.shap_summary:
            return None
        dominant_share = self._dominant_feature_share_value(
            candidate.shap_summary.get("mean_abs_shap", {}) or {}
        )
        if dominant_share is None:
            return None
        return 1.0 - dominant_share

    @staticmethod
    def _dominant_feature_share_value(mean_abs_shap: Dict[str, float]) -> Optional[float]:
        """Share of the single largest |SHAP| value among the reported features."""
        values = [value for value in mean_abs_shap.values() if value is not None]
        total = sum(values)
        if not values or total <= 0.0:
            return None
        return max(values) / total

    @staticmethod
    def _finding(dimension: str, label: str, passed: bool, message: str) -> Finding:
        """Build a pass/fail Finding (info findings are constructed inline)."""
        return Finding(
            dimension=dimension,
            label=label,
            status=STATUS_PASS if passed else STATUS_FAIL,
            message=message,
        )

    def _fmt(self, task_type: str, value: Optional[float]) -> str:
        """Format a metric value: percent for classification, 4dp otherwise."""
        if value is None:
            return "n/a"
        if task_type == "classification":
            return f"{value * 100:.1f}%"
        return f"{value:.4f}"

    def _fmt_gap(self, task_type: str, value: Optional[float]) -> str:
        """Format a gap/spread value with the same percent/decimal convention."""
        if value is None:
            return "n/a"
        if task_type == "classification":
            return f"{value * 100:.1f}%"
        return f"{value:.4f}"
