"""Tests for adapter.py: UpstreamAdapter maps upstream JSONs into CandidateModel."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapter import UpstreamAdapter
from schemas import CandidateModel, JudgeInput, OverfittingInfo


class TestUpstreamAdapter:

    def test_adapt_overfitting_basic(self, overfitting_logistic_dict: dict) -> None:
        adapter = UpstreamAdapter()
        result = adapter.adapt_overfitting(overfitting_logistic_dict)
        assert isinstance(result, OverfittingInfo)
        assert result.is_overfitted is False
        assert abs(result.gap - 0.03) < 1e-6
        assert result.train_vs_cv_gap is not None
        assert abs(result.train_vs_cv_gap - 0.02) < 1e-6

    def test_adapt_overfitting_missing_cv(self) -> None:
        adapter = UpstreamAdapter()
        minimal = {
            "is_overfitted": True,
            "primary_metric": "accuracy",
            "gaps": {"accuracy": 0.18},
        }
        result = adapter.adapt_overfitting(minimal)
        assert result.is_overfitted is True
        assert abs(result.gap - 0.18) < 1e-6
        assert result.train_vs_cv_gap is None

    def test_adapt_judge_input_from_classification_fixture(
        self, classification_judge_input_dict: dict
    ) -> None:
        adapter = UpstreamAdapter()
        # The classification fixture is already in JudgeInput format.
        # Test adapt_candidate directly for one model.
        raw_candidate = classification_judge_input_dict["candidates"][0]
        candidate = adapter.adapt_candidate(
            model_name=raw_candidate["model_name"],
            task_type=raw_candidate["task_type"],
            metrics=raw_candidate["metrics"],
            overfitting_json={
                "is_overfitted": raw_candidate["overfitting"]["is_overfitted"],
                "primary_metric": "accuracy",
                "gaps": {"accuracy": raw_candidate["overfitting"]["gap"]},
                "k_fold_cross_validation_results": {
                    "train_vs_cv_gap": raw_candidate["overfitting"]["train_vs_cv_gap"]
                },
            },
            complexity_dict=raw_candidate["complexity"],
            shap_summary=raw_candidate.get("shap_summary"),
            hyperparam_sensitivity=raw_candidate.get("hyperparam_sensitivity"),
        )
        assert isinstance(candidate, CandidateModel)
        assert candidate.model_name == "LogisticRegression"
        assert candidate.task_type == "classification"
        assert candidate.complexity.family_rank == 1
        assert candidate.shap_summary is not None
