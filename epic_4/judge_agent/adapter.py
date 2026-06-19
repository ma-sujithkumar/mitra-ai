"""
UpstreamAdapter: maps each upstream source into the judge's CandidateModel schema.

The judge never directly parses upstream output formats; all translation lives here.
This keeps the judge decoupled from upstream schema changes.

Upstream sources (one JSON file per model):
  - overfitting_analysis.json  (from epic_4/overfitting_analysis_tool)
  - inference_metrics          (per-model metrics dict)
  - shap_summary               (optional per-model SHAP dict)
  - hyperparam_sensitivity     (optional per-model tuning sensitivity dict)
  - complexity                 (explicit n_params/depth/family_rank dict)
"""

import logging
from typing import Any, Dict, List, Optional

from schemas import (
    CandidateModel,
    ComplexityDescriptor,
    JudgeInput,
    OverfittingInfo,
)

logger = logging.getLogger(__name__)


class UpstreamAdapter:
    """Translates upstream tool outputs into the JudgeInput adapter schema."""

    def adapt_overfitting(self, overfitting_json: Dict[str, Any]) -> OverfittingInfo:
        """Map overfitting_analysis.json => OverfittingInfo.

        The upstream schema (epic_4/overfitting_analysis_tool/SPEC.md Section 6)
        provides is_overfitted, gaps (dict), and k_fold_cross_validation_results.
        We extract the primary_metric gap and train_vs_cv_gap.
        """
        primary_metric = overfitting_json.get("primary_metric", "accuracy")
        gaps = overfitting_json.get("gaps", {})
        primary_gap = gaps.get(primary_metric, 0.0)

        cv_results = overfitting_json.get("k_fold_cross_validation_results") or {}
        train_vs_cv_gap = cv_results.get("train_vs_cv_gap", None)

        logger.debug(
            "=> Adapted overfitting: primary_metric=%s gap=%.4f train_vs_cv_gap=%s",
            primary_metric,
            primary_gap,
            train_vs_cv_gap,
        )
        return OverfittingInfo(
            is_overfitted=bool(overfitting_json.get("is_overfitted", False)),
            gap=float(primary_gap),
            train_vs_cv_gap=float(train_vs_cv_gap) if train_vs_cv_gap is not None else None,
        )

    def adapt_complexity(self, complexity_dict: Dict[str, Any]) -> ComplexityDescriptor:
        """Map a raw complexity dict => ComplexityDescriptor."""
        return ComplexityDescriptor(
            n_params=int(complexity_dict.get("n_params", 0)),
            depth=int(complexity_dict.get("depth", 0)),
            family_rank=int(complexity_dict.get("family_rank", 1)),
        )

    def adapt_candidate(
        self,
        model_name: str,
        task_type: str,
        metrics: Dict[str, Optional[float]],
        overfitting_json: Dict[str, Any],
        complexity_dict: Dict[str, Any],
        shap_summary: Optional[Dict[str, Any]] = None,
        hyperparam_sensitivity: Optional[Dict[str, Any]] = None,
    ) -> CandidateModel:
        """Build a single CandidateModel from all upstream sources for one model."""
        overfitting_info = self.adapt_overfitting(overfitting_json)
        complexity_descriptor = self.adapt_complexity(complexity_dict)
        logger.debug(
            "=> Adapted candidate: model=%s task=%s", model_name, task_type
        )
        return CandidateModel(
            model_name=model_name,
            task_type=task_type,
            metrics=metrics,
            overfitting=overfitting_info,
            complexity=complexity_descriptor,
            shap_summary=shap_summary,
            hyperparam_sensitivity=hyperparam_sensitivity,
        )

    def adapt_judge_input(
        self,
        candidate_raw_list: List[Dict[str, Any]],
        dataset_id: Optional[str] = None,
        minidata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeInput:
        """Build a full JudgeInput from a list of raw per-model dicts.

        Each element of candidate_raw_list must have:
            model_name, task_type, metrics,
            overfitting_json, complexity,
        And optionally: shap_summary, hyperparam_sensitivity.
        """
        candidates = []
        for raw in candidate_raw_list:
            candidate = self.adapt_candidate(
                model_name=raw["model_name"],
                task_type=raw["task_type"],
                metrics=raw["metrics"],
                overfitting_json=raw["overfitting_json"],
                complexity_dict=raw["complexity"],
                shap_summary=raw.get("shap_summary"),
                hyperparam_sensitivity=raw.get("hyperparam_sensitivity"),
            )
            candidates.append(candidate)

        logger.debug("=> Adapted %d candidates into JudgeInput", len(candidates))
        return JudgeInput(
            dataset_id=dataset_id,
            candidates=candidates,
            minidata=minidata,
            metadata=metadata,
        )
