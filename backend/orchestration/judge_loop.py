"""Judge + model-selection feedback loop.

Flow per turn:
  1. Build JudgeInput from eval_runner artifacts + training_summary.
  2. Run JudgeAgent.judge() -> JudgeDecision.
  3. If selected_model is None (all rejected) AND turns_remaining > 0:
       - exclude rejected model names from the model pool
       - re-invoke select_models() to get new candidates
       - caller is expected to re-train + re-evaluate (see run_pipeline.py)
  4. Otherwise return the JudgeDecision.

JudgeLoop.run() executes all turns inline; it does NOT re-train.
Re-training with feedback is done by run_pipeline.py which calls this.
For multi-turn the caller should use run_with_feedback() which accepts a
training_callback.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from backend.agents.evaluation.judge.adapter import UpstreamAdapter
from backend.agents.evaluation.judge.judge_agent import JudgeAgent
from backend.agents.evaluation.judge.schemas import JudgeDecision, JudgeInput

logger = logging.getLogger(__name__)


class EvalArtifacts:
    """Value object holding paths from EvalRunner.run()."""

    def __init__(
        self,
        shap_dirs: Dict[str, Optional[str]],
        overfitting_dirs: Dict[str, Optional[str]],
        hpt_results_path: Optional[str],
    ) -> None:
        self.shap_dirs = shap_dirs
        self.overfitting_dirs = overfitting_dirs
        self.hpt_results_path = hpt_results_path


class JudgeInputBuilder:
    """Builds JudgeInput from eval artifacts + training summary for one turn."""

    def __init__(self, task_type: str) -> None:
        self.task_type = task_type
        self.adapter = UpstreamAdapter()

    def build(
        self,
        eval_artifacts: EvalArtifacts,
        training_summary: Any,
        session_dir: Path,
        dataset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeInput:
        """Build JudgeInput preferring HPT path; falls back to training_summary path."""
        if eval_artifacts.hpt_results_path and Path(eval_artifacts.hpt_results_path).exists():
            return self._build_from_hpt(
                eval_artifacts=eval_artifacts,
                dataset_id=dataset_id,
                metadata=metadata,
            )
        return self._build_from_training_summary(
            eval_artifacts=eval_artifacts,
            training_summary=training_summary,
            session_dir=session_dir,
            dataset_id=dataset_id,
            metadata=metadata,
        )

    def _build_from_hpt(
        self,
        eval_artifacts: EvalArtifacts,
        dataset_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> JudgeInput:
        """Use adapt_from_hpt_results() which already handles SHAP enrichment."""
        # Use the first non-None shap_dir as root (per-model lookup handled in adapter)
        shap_root_dir: Optional[str] = None
        non_null_shap = [v for v in eval_artifacts.shap_dirs.values() if v is not None]
        if non_null_shap:
            # shap_dirs values are per-model dirs; parent is the shap root
            shap_root_dir = str(Path(non_null_shap[0]).parent)

        return self.adapter.adapt_from_hpt_results(
            hpt_json_path=eval_artifacts.hpt_results_path,
            task_type=self.task_type,
            shap_dir=shap_root_dir,
            dataset_id=dataset_id,
            metadata=metadata,
        )

    def _build_from_training_summary(
        self,
        eval_artifacts: EvalArtifacts,
        training_summary: Any,
        session_dir: Path,
        dataset_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> JudgeInput:
        """Build candidates directly from training_summary + overfitting JSONs."""
        primary_metric = "accuracy" if self.task_type == "classification" else "r2"
        models = getattr(training_summary, "models", []) or []
        candidate_raws: List[Dict[str, Any]] = []

        for model_result in models:
            model_name = model_result.model_name

            # Read overfitting_analysis.json if available
            overfit_json: Dict[str, Any] = {"is_overfitted": False, "primary_metric": primary_metric, "gaps": {primary_metric: 0.0}}
            overfit_dir = eval_artifacts.overfitting_dirs.get(model_name)
            if overfit_dir:
                overfit_json_path = Path(overfit_dir) / "overfitting_analysis.json"
                if overfit_json_path.exists():
                    with overfit_json_path.open(encoding="utf-8") as overfit_file:
                        overfit_json = json.load(overfit_file)

            # Build SHAP summary from per-model CSV
            shap_summary: Optional[Dict[str, Any]] = None
            shap_model_dir = eval_artifacts.shap_dirs.get(model_name)
            if shap_model_dir:
                shap_csv = Path(shap_model_dir) / "csv" / "global_feature_importance.csv"
                shap_summary = self.adapter.build_shap_summary_from_csv(str(shap_csv))

            # Build minimal complexity descriptor from model_name
            complexity_dict = self._infer_complexity(model_name)

            # Build metrics dict from training_summary with all available metrics
            raw_metrics = getattr(model_result, "metrics", {}) or {}
            val_metrics = raw_metrics.get("validation") if isinstance(raw_metrics, dict) and "validation" in raw_metrics else raw_metrics
            
            val_score = getattr(model_result, "validation_score", None)
            metrics_dict: Dict[str, Optional[float]] = {primary_metric: val_score}
            if isinstance(val_metrics, dict):
                for metric_key, metric_value in val_metrics.items():
                    metrics_dict[metric_key] = metric_value

            candidate_raws.append({
                "model_name": model_name,
                "task_type": self.task_type,
                "metrics": metrics_dict,
                "overfitting_json": overfit_json,
                "complexity": complexity_dict,
                "shap_summary": shap_summary,
            })

        return self.adapter.adapt_judge_input(
            candidate_raw_list=candidate_raws,
            dataset_id=dataset_id,
            metadata=metadata,
        )

    def _infer_complexity(self, model_name: str) -> Dict[str, Any]:
        """Infer a rough ComplexityDescriptor from the model name string."""
        # Ordered from simplest to most complex — used as family_rank.
        complexity_rank_map: Dict[str, int] = {
            "LinearRegression": 1, "Ridge": 2, "Lasso": 2, "ElasticNet": 3,
            "LogisticRegression": 1, "RidgeClassifier": 2, "SGD": 2,
            "DecisionTree": 3, "KNeighbors": 3, "NaiveBayes": 2,
            "RandomForest": 5, "ExtraTrees": 5, "Bagging": 5,
            "GradientBoosting": 7, "HistGradientBoosting": 6, "XGB": 8,
            "AdaBoost": 6, "SV": 4, "MLP": 9, "PyTorch": 10,
        }
        family_rank = 5  # default middle rank
        for family_key, rank_value in complexity_rank_map.items():
            if family_key.lower() in model_name.lower():
                family_rank = rank_value
                break
        return {"n_params": 0, "depth": 0, "family_rank": family_rank}


class JudgeLoop:
    """Runs the judge + optional feedback re-selection loop.

    Single-turn usage (no re-training):
        loop = JudgeLoop(task_type="classification", max_turns=1)
        decision = loop.run(eval_artifacts, training_summary, session_dir)

    Multi-turn usage (re-selection + re-training callback):
        loop = JudgeLoop(task_type="classification", max_turns=3)
        decision = loop.run_with_feedback(
            eval_artifacts, training_summary, session_dir,
            training_callback=my_retrain_fn,
            metadata_path=..., feature_selection_path=...,
            mini_data_path=..., model_library_root=...,
        )
    """

    def __init__(
        self,
        task_type: str,
        max_turns: int = 3,
        use_llm: Optional[bool] = None,
        judge_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        self.task_type = task_type
        self.max_turns = max_turns
        self.use_llm = use_llm
        self.verbose = verbose
        self.input_builder = JudgeInputBuilder(task_type=task_type)
        self.judge_agent = JudgeAgent(config=judge_config)

    def run(
        self,
        eval_artifacts: EvalArtifacts,
        training_summary: Any,
        session_dir: Path,
        dataset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeDecision:
        """Run the judge once and return the decision. No feedback loop."""
        judge_input = self.input_builder.build(
            eval_artifacts=eval_artifacts,
            training_summary=training_summary,
            session_dir=session_dir,
            dataset_id=dataset_id,
            metadata=metadata,
        )
        decision = self.judge_agent.judge(judge_input, use_llm=self.use_llm)
        self._persist_decision(decision, session_dir, turn=1)
        logger.info(
            "=> judge turn 1: selected=%s ranked=%d",
            decision.selected_model,
            len(decision.ranked_models),
        )
        return decision

    def run_with_feedback(
        self,
        eval_artifacts: EvalArtifacts,
        training_summary: Any,
        session_dir: Path,
        training_callback: Callable[[List[str]], Any],
        metadata_path: Path,
        feature_selection_path: Path,
        mini_data_path: Path,
        model_library_root: Path,
        max_models: int = 10,
        dataset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeDecision:
        """Run the judge + feedback loop up to max_turns.

        On each turn where all models are rejected the loop:
          1. Collects rejected model names.
          2. Re-invokes select_models() excluding those names.
          3. Calls training_callback(excluded_model_names) to re-train.
          4. Expects training_callback to return a new training_summary.
          5. Re-runs EvalRunner (assumed externally) and retries.

        Args:
            training_callback: fn(excluded_names: List[str]) -> new training_summary
        """
        excluded_model_names: List[str] = []
        current_training_summary = training_summary
        current_eval_artifacts = eval_artifacts
        last_decision: Optional[JudgeDecision] = None

        for turn_number in range(1, self.max_turns + 1):
            judge_input = self.input_builder.build(
                eval_artifacts=current_eval_artifacts,
                training_summary=current_training_summary,
                session_dir=session_dir,
                dataset_id=dataset_id,
                metadata=metadata,
            )

            if not judge_input.candidates:
                logger.warning("=> judge turn %d: no candidates left, stopping loop.", turn_number)
                break

            decision = self.judge_agent.judge(judge_input, use_llm=self.use_llm)
            self._persist_decision(decision, session_dir, turn=turn_number)
            last_decision = decision

            logger.info(
                "=> judge turn %d/%d: selected=%s ranked=%d",
                turn_number,
                self.max_turns,
                decision.selected_model,
                len(decision.ranked_models),
            )

            # If a winner was found, stop the loop.
            if decision.selected_model is not None:
                logger.info("=> judge loop done: selected=%s after %d turn(s)", decision.selected_model, turn_number)
                return decision

            if turn_number >= self.max_turns:
                logger.warning("=> judge loop reached max_turns=%d with no winner.", self.max_turns)
                break

            # All rejected — collect rejected names and re-select
            newly_rejected = [
                ranked_model.model_name
                for ranked_model in decision.ranked_models
                if ranked_model.verdict == "reject"
            ]
            excluded_model_names.extend(newly_rejected)
            excluded_model_names = list(set(excluded_model_names))

            logger.info(
                "=> judge feedback: all rejected on turn %d. Excluded so far: %s",
                turn_number,
                excluded_model_names,
            )

            # Re-invoke model selection excluding all rejected names, then re-train.
            self._reselect_models(
                excluded_model_names=excluded_model_names,
                session_dir=session_dir,
                metadata_path=metadata_path,
                feature_selection_path=feature_selection_path,
                mini_data_path=mini_data_path,
                model_library_root=model_library_root,
                max_models=max_models,
            )

            # Re-train via callback; callback should also re-run EvalRunner.
            current_training_summary = training_callback(excluded_model_names)

        return last_decision or decision  # type: ignore[return-value]

    def _reselect_models(
        self,
        excluded_model_names: List[str],
        session_dir: Path,
        metadata_path: Path,
        feature_selection_path: Path,
        mini_data_path: Path,
        model_library_root: Path,
        max_models: int,
    ) -> None:
        """Re-run select_models() with exclusions and overwrite model_config.json."""
        from backend.agents.model_selection.selector import select_models
        model_config_path = session_dir / "reports" / "model_config.json"
        select_models(
            metadata_path=metadata_path,
            feature_selection_path=feature_selection_path,
            mini_data_path=mini_data_path,
            model_library_root=model_library_root,
            output_path=model_config_path,
            max_models=max_models,
            excluded_model_names=excluded_model_names,
        )
        logger.info("=> re-selected models (excluded %d). Config: %s", len(excluded_model_names), model_config_path)

    def _persist_decision(self, decision: JudgeDecision, session_dir: Path, turn: int) -> None:
        """Write judge_decision.json (overwritten each turn; turn N also archived)."""
        reports_dir = session_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        decision_data = decision.model_dump()
        # Always overwrite the canonical output
        canonical_path = reports_dir / "judge_decision.json"
        canonical_path.write_text(json.dumps(decision_data, indent=2), encoding="utf-8")
        # Archive per-turn output for auditability
        turn_path = reports_dir / f"judge_decision_turn_{turn}.json"
        turn_path.write_text(json.dumps(decision_data, indent=2), encoding="utf-8")
        logger.debug("=> judge decision written: %s", canonical_path)
