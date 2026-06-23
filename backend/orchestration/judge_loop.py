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
from time import monotonic
from typing import Any, Callable, Dict, List, Optional

from backend.agents.evaluation.judge.adapter import UpstreamAdapter
from backend.agents.evaluation.judge.judge_agent import JudgeAgent
from backend.agents.evaluation.judge.schemas import JudgeDecision, JudgeInput
from backend.orchestration.events import TrainingEvent, TrainingEventBus
from backend.orchestration.eval_runner import EvaluationRestartRequested

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
        domain_reasoning: Optional[Dict[str, Any]] = None,
    ) -> JudgeInput:
        """Build JudgeInput preferring HPT path; falls back to training_summary path."""
        if eval_artifacts.hpt_results_path and Path(eval_artifacts.hpt_results_path).exists():
            return self._build_from_hpt(
                eval_artifacts=eval_artifacts,
                dataset_id=dataset_id,
                metadata=metadata,
                domain_reasoning=domain_reasoning,
            )
        return self._build_from_training_summary(
            eval_artifacts=eval_artifacts,
            training_summary=training_summary,
            session_dir=session_dir,
            dataset_id=dataset_id,
            metadata=metadata,
            domain_reasoning=domain_reasoning,
        )

    def _build_from_hpt(
        self,
        eval_artifacts: EvalArtifacts,
        dataset_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        domain_reasoning: Optional[Dict[str, Any]] = None,
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
            domain_reasoning=domain_reasoning,
        )

    def _build_from_training_summary(
        self,
        eval_artifacts: EvalArtifacts,
        training_summary: Any,
        session_dir: Path,
        dataset_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        domain_reasoning: Optional[Dict[str, Any]] = None,
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
            domain_reasoning=domain_reasoning,
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
        event_bus: Optional["TrainingEventBus"] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.task_type = task_type
        self.max_turns = max_turns
        self.use_llm = use_llm
        self.verbose = verbose
        self.input_builder = JudgeInputBuilder(task_type=task_type)
        self.judge_agent = JudgeAgent(config=judge_config)
        self.event_bus = event_bus
        self.session_id = session_id

    def run(
        self,
        eval_artifacts: EvalArtifacts,
        training_summary: Any,
        session_dir: Path,
        dataset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeDecision:
        """Run the judge once and return the decision. No feedback loop."""
        def status_callback(msg: str, details: Optional[Dict[str, Any]] = None) -> None:
            self._write_judge_status(
                session_dir=session_dir,
                turn=1,
                max_turns=1,
                status="running",
                message=msg,
                details=details,
            )

        self._write_judge_status(session_dir, 1, 1, "running", "Building evaluation input from SHAP, overfitting, and training metrics...")
        self._emit_judge_event(
            turn_number=1,
            total_turns=1,
            status="running",
            msg="[JUDGE] Building evaluation input from SHAP, overfitting, and training metrics...",
            pct=10,
        )
        # Domain reasoning is generated once upstream (metadata stage) and is
        # only ever read here, never regenerated -- a single disk read per
        # JudgeLoop.run() call, reused as-is for this single-turn run.
        domain_reasoning = self._load_domain_reasoning(session_dir=session_dir)
        judge_input = self.input_builder.build(
            eval_artifacts=eval_artifacts,
            training_summary=training_summary,
            session_dir=session_dir,
            dataset_id=dataset_id,
            metadata=metadata,
            domain_reasoning=domain_reasoning,
        )
        candidate_count = len(judge_input.candidates) if judge_input else 0
        self._write_judge_status(session_dir, 1, 1, "running", f"Evaluating {candidate_count} model candidate(s) with rule engine + LLM...")
        self._emit_judge_event(
            turn_number=1,
            total_turns=1,
            status="running",
            msg=f"[JUDGE] Evaluating {candidate_count} model candidate(s) with rule engine + LLM...",
            pct=40,
        )
        decision = self.judge_agent.judge(judge_input, use_llm=self.use_llm, status_callback=status_callback)
        self._persist_decision(decision, session_dir, turn=1)
        # Stream per-model Judge findings live to the leaderboard.
        self._emit_findings_stream(decision, turn_number=1, total_turns=1)
        logger.info(
            "=> judge turn 1: selected=%s ranked=%d",
            decision.selected_model,
            len(decision.ranked_models),
        )
        self._write_judge_status(
            session_dir=session_dir,
            turn=1,
            max_turns=1,
            status="completed",
            message=f"Decision: selected model = {decision.selected_model or 'none'} ({len(decision.ranked_models)} models ranked).",
            details={"selected_model": decision.selected_model, "ranked_count": len(decision.ranked_models)},
        )
        self._emit_judge_event(
            turn_number=1,
            total_turns=1,
            status="completed",
            msg=f"[JUDGE] Decision: selected model = {decision.selected_model or 'none'} ({len(decision.ranked_models)} models ranked).",
            pct=100,
            details={"selected_model": decision.selected_model, "ranked_count": len(decision.ranked_models)},
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
        # Typed as Any: at runtime this is a multiprocessing.Manager().Event()
        # proxy, the same instance threaded into EvalRunner.restart_event.
        turn_restart_event: Optional[Any] = None,
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
        accumulated_approved: List[str] = []
        accumulated_rejected: List[str] = []
        current_training_summary = training_summary
        current_eval_artifacts = eval_artifacts
        last_decision: Optional[JudgeDecision] = None
        # Initialize decision to None so the return below is always defined
        # even when the loop breaks before judge_agent.judge() is called.
        decision: Optional[JudgeDecision] = None
        # Loaded exactly once for the whole feedback loop -- every turn reuses
        # the same in-memory dict rather than re-reading or regenerating it.
        domain_reasoning = self._load_domain_reasoning(session_dir=session_dir)

        turn_number = 1
        while turn_number <= self.max_turns:
            try:
                # Capture turn_number by value so the closure uses the current turn,
                # not the loop variable at call time.
                def status_callback(
                    msg: str,
                    details: Optional[Dict[str, Any]] = None,
                    _turn: int = turn_number,
                ) -> None:
                    self._write_judge_status(
                        session_dir=session_dir,
                        turn=_turn,
                        max_turns=self.max_turns,
                        status="running",
                        message=msg,
                        details=details,
                    )
                    # Also emit SSE so the judge stage card updates live during LLM inference
                    # (status_callback is called from inside judge_agent.judge() before the
                    # 30-90 second LLM call, so this gives the frontend a timely update).
                    self._emit_judge_event(
                        turn_number=_turn,
                        total_turns=self.max_turns,
                        status="running",
                        msg=f"[JUDGE] {msg}",
                        pct=max(25, int((_turn - 0.5) / self.max_turns * 70)),
                        details=details,
                    )

                self._write_judge_status(session_dir, turn_number, self.max_turns, "running", f"Turn {turn_number}/{self.max_turns}: Building evaluation input (excluded: {len(excluded_model_names)} models)...")
                self._emit_judge_event(
                    turn_number=turn_number,
                    total_turns=self.max_turns,
                    status="running",
                    msg=f"[JUDGE] Turn {turn_number}/{self.max_turns}: Building evaluation input (excluded: {len(excluded_model_names)} models)...",
                    pct=max(10, int((turn_number - 1) / self.max_turns * 70)),
                )
                build_started = monotonic()
                judge_input = self.input_builder.build(
                    eval_artifacts=current_eval_artifacts,
                    training_summary=current_training_summary,
                    session_dir=session_dir,
                    dataset_id=dataset_id,
                    metadata=metadata,
                    domain_reasoning=domain_reasoning,
                )
                logger.info(
                    "=> judge turn %d: judge_input built in %.1fs",
                    turn_number,
                    monotonic() - build_started,
                )

                if not judge_input.candidates:
                    logger.warning("=> judge turn %d: no candidates left, stopping loop.", turn_number)
                    self._write_judge_status(session_dir, turn_number, self.max_turns, "failed", f"Turn {turn_number}: No candidate models remaining after exclusions. Stopping.")
                    self._emit_judge_event(
                        turn_number=turn_number,
                        total_turns=self.max_turns,
                        status="failed",
                        msg=f"[JUDGE] Turn {turn_number}: No candidate models remaining after exclusions. Stopping.",
                        pct=100,
                    )
                    break

                candidate_count = len(judge_input.candidates)
                self._write_judge_status(session_dir, turn_number, self.max_turns, "running", f"Turn {turn_number}/{self.max_turns}: Evaluating {candidate_count} candidate(s) with LLM...")
                self._emit_judge_event(
                    turn_number=turn_number,
                    total_turns=self.max_turns,
                    status="running",
                    msg=f"[JUDGE] Turn {turn_number}/{self.max_turns}: Evaluating {candidate_count} candidate(s) with LLM...",
                    pct=max(20, int((turn_number - 0.5) / self.max_turns * 70)),
                )
                if turn_restart_event is not None and turn_restart_event.is_set():
                    raise EvaluationRestartRequested()

                judge_started = monotonic()
                decision = self.judge_agent.judge(judge_input, use_llm=self.use_llm, status_callback=status_callback)
                logger.info(
                    "=> judge turn %d: judge_agent.judge() took %.1fs for %d candidate(s)",
                    turn_number,
                    monotonic() - judge_started,
                    candidate_count,
                )
                self._persist_decision(decision, session_dir, turn=turn_number)
                # Stream per-model Judge findings live to the leaderboard.
                self._emit_findings_stream(decision, turn_number=turn_number, total_turns=self.max_turns)
                last_decision = decision

                logger.info(
                    "=> judge turn %d/%d: selected=%s ranked=%d",
                    turn_number,
                    self.max_turns,
                    decision.selected_model,
                    len(decision.ranked_models),
                )

                # Collect rejected model names from this turn.
                rejected_names = [
                    ranked_model.model_name
                    for ranked_model in decision.ranked_models
                    if ranked_model.verdict == "reject"
                ]

                # Update accumulated approved/rejected tracking for reporting.
                for ranked_model in decision.ranked_models:
                    if ranked_model.verdict == "select":
                        if ranked_model.model_name not in accumulated_approved:
                            accumulated_approved.append(ranked_model.model_name)
                        if ranked_model.model_name in accumulated_rejected:
                            accumulated_rejected.remove(ranked_model.model_name)
                    elif ranked_model.verdict == "reject":
                        if ranked_model.model_name not in accumulated_rejected:
                            accumulated_rejected.append(ranked_model.model_name)
                        if ranked_model.model_name in accumulated_approved:
                            accumulated_approved.remove(ranked_model.model_name)

                # Stop when: no rejections (all candidates accepted) OR turn ceiling hit.
                # A winner alone does NOT stop the loop -- if any models are rejected and
                # turns remain, we expand the pool with fresh candidates to find even better.
                if not rejected_names or turn_number >= self.max_turns:
                    final_status = "all_completed" if decision.selected_model is not None else "completed"
                    final_msg = (
                        f"Converged on turn {turn_number}/{self.max_turns}: Winner = {decision.selected_model} ({len(decision.ranked_models)} models ranked)."
                        if decision.selected_model is not None
                        else f"Reached max turns ({self.max_turns}) with no clear winner. Returning best available."
                    )
                    logger.info("=> judge loop done after turn %d: selected=%s", turn_number, decision.selected_model)
                    self._write_judge_status(
                        session_dir=session_dir,
                        turn=turn_number,
                        max_turns=self.max_turns,
                        status=final_status,
                        message=final_msg,
                        details={"selected_model": decision.selected_model, "ranked_count": len(decision.ranked_models), "turn": turn_number},
                    )
                    self._emit_judge_event(
                        turn_number=turn_number,
                        total_turns=self.max_turns,
                        status=final_status,
                        msg=f"[JUDGE] {final_msg}",
                        pct=100,
                        details={"selected_model": decision.selected_model, "ranked_count": len(decision.ranked_models), "turn": turn_number},
                    )
                    return decision

                # There are rejected models and turns remaining: expand the pool.
                # Exclude ALL tried models (approved + rejected) so the next turn brings
                # in genuinely fresh candidates the judge has never seen.
                all_tried_names = [rm.model_name for rm in decision.ranked_models]
                excluded_model_names.extend(all_tried_names)
                excluded_model_names = list(set(excluded_model_names))

                accepted_count = len(decision.ranked_models) - len(rejected_names)
                # Halve the NEW-candidate count each turn: turn 2 => N/2, turn 3 => N/4, etc.
                # The effective_max passed to the selector must also include the approved carries
                # so those don't consume the new-model slots (bug: approved ate slots causing 0 new
                # models in turn 3 and a wasted LLM judge call on an unchanged pool).
                new_models_count = max(2, max_models // (2 ** turn_number))
                effective_max_models = len(accumulated_approved) + new_models_count
                logger.info(
                    "=> judge feedback turn %d: %d accepted, %d rejected. "
                    "Selecting %d new candidates (effective_max=%d). Excluded: %s",
                    turn_number,
                    accepted_count,
                    len(rejected_names),
                    new_models_count,
                    effective_max_models,
                    excluded_model_names,
                )
                feedback_msg = (
                    f"Turn {turn_number}: {accepted_count} accepted, {len(rejected_names)} rejected. "
                    f"Selecting {new_models_count} new candidates for turn {turn_number + 1}..."
                )
                self._write_judge_status(
                    session_dir=session_dir,
                    turn=turn_number,
                    max_turns=self.max_turns,
                    status="running",
                    message=feedback_msg,
                )
                self._emit_judge_event(
                    turn_number=turn_number,
                    total_turns=self.max_turns,
                    status="running",
                    msg=f"[JUDGE] {feedback_msg}",
                    pct=max(30, int(turn_number / self.max_turns * 60)),
                    details={"accepted": accepted_count, "rejected": len(rejected_names)},
                )

                # Re-invoke model selection; pass effective_max so approved carries don't consume
                # the new-model slots (approved + new_models_count slots total).
                reselect_started = monotonic()
                self._reselect_models(
                    approved_model_names=accumulated_approved,
                    rejected_model_names=accumulated_rejected,
                    session_dir=session_dir,
                    metadata_path=metadata_path,
                    feature_selection_path=feature_selection_path,
                    mini_data_path=mini_data_path,
                    model_library_root=model_library_root,
                    max_models=effective_max_models,
                )
                logger.info(
                    "=> judge turn %d: _reselect_models() took %.1fs",
                    turn_number,
                    monotonic() - reselect_started,
                )

                # Track model count before callback to detect the "no new models" skip path.
                models_before_callback = len(getattr(current_training_summary, "models", []) or [])

                # training_callback() retrains the new candidates and blocks on a fresh
                # SHAP/overfitting evaluation pass for them. Emit an explicit status here
                # so the judge panel doesn't sit on turn N's stale "Selecting..." message
                # while the overfitting/SHAP cards are actively running for turn N+1 -- it
                # looks like the judge has already moved to turn N+1, which it has not.
                retrain_msg = (
                    f"Turn {turn_number}: retraining {new_models_count} new candidate(s) and "
                    f"re-evaluating (SHAP + overfitting) before turn {turn_number + 1} begins..."
                )
                self._write_judge_status(
                    session_dir=session_dir,
                    turn=turn_number,
                    max_turns=self.max_turns,
                    status="running",
                    message=retrain_msg,
                )
                self._emit_judge_event(
                    turn_number=turn_number,
                    total_turns=self.max_turns,
                    status="running",
                    msg=f"[JUDGE] {retrain_msg}",
                    pct=max(30, int(turn_number / self.max_turns * 60)),
                )

                # Re-train via callback; callback merges models across turns and re-runs EvalRunner.
                callback_started = monotonic()
                callback_result = training_callback(excluded_model_names)
                logger.info(
                    "=> judge turn %d: training_callback() (retrain + re-eval) took %.1fs",
                    turn_number,
                    monotonic() - callback_started,
                )
                if isinstance(callback_result, tuple):
                    current_training_summary, new_eval_output = callback_result
                    current_eval_artifacts = EvalArtifacts(
                        shap_dirs=new_eval_output.get("shap_dirs", {}),
                        overfitting_dirs=new_eval_output.get("overfitting_dirs", {}),
                        hpt_results_path=new_eval_output.get("hpt_results_path"),
                    )
                else:
                    current_training_summary = callback_result

                # Early-exit if the callback returned no new models (catalog exhausted or all
                # reselected candidates were already trained). Re-running the judge on an
                # identical pool would consume LLM tokens for zero new information.
                models_after_callback = len(getattr(current_training_summary, "models", []) or [])
                if models_after_callback <= models_before_callback:
                    logger.info(
                        "=> judge loop: no new models after turn %d callback (before=%d, after=%d). "
                        "Returning current decision without redundant judge call.",
                        turn_number,
                        models_before_callback,
                        models_after_callback,
                    )
                    self._write_judge_status(
                        session_dir=session_dir,
                        turn=turn_number,
                        max_turns=self.max_turns,
                        status="completed",
                        message=f"Catalog exhausted after turn {turn_number}: no new candidates available. Returning best result.",
                        details={"selected_model": decision.selected_model, "ranked_count": len(decision.ranked_models)},
                    )
                    return decision
            except EvaluationRestartRequested:
                # User clicked "restart this turn": the SHAP/overfitting subprocesses
                # for this turn were just killed. Clear the event and redo turn_number
                # from scratch (same candidate pool, no advance) instead of failing or
                # silently continuing with a partial/aborted evaluation.
                if turn_restart_event is not None:
                    turn_restart_event.clear()
                restart_msg = f"Turn {turn_number}: evaluation restart requested by user. Re-running this turn from scratch..."
                logger.info("=> judge turn %d: restart requested by user, redoing turn.", turn_number)
                self._write_judge_status(session_dir, turn_number, self.max_turns, "running", restart_msg)
                self._emit_judge_event(
                    turn_number=turn_number,
                    total_turns=self.max_turns,
                    status="running",
                    msg=f"[JUDGE] {restart_msg}",
                    pct=max(10, int((turn_number - 1) / self.max_turns * 70)),
                )
                continue

            turn_number += 1

        return last_decision or decision  # type: ignore[return-value]  # decision init'd above

    def _reselect_models(
        self,
        approved_model_names: List[str],
        rejected_model_names: List[str],
        session_dir: Path,
        metadata_path: Path,
        feature_selection_path: Path,
        mini_data_path: Path,
        model_library_root: Path,
        max_models: int,
    ) -> None:
        """Re-run select_models() with judge feedback and overwrite model_config.json."""
        from backend.agents.model_selection.selector import select_models
        model_config_path = session_dir / "reports" / "model_config.json"
        select_models(
            metadata_path=metadata_path,
            feature_selection_path=feature_selection_path,
            mini_data_path=mini_data_path,
            model_library_root=model_library_root,
            output_path=model_config_path,
            max_models=max_models,
            approved_model_names=approved_model_names,
            rejected_model_names=rejected_model_names,
        )
        logger.info("=> re-selected models (approved %d, rejected %d). Config: %s", len(approved_model_names), len(rejected_model_names), model_config_path)

    def _emit_judge_event(
        self,
        turn_number: int,
        total_turns: int,
        status: str,
        msg: str,
        pct: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a judge SSE event if event_bus and session_id are available."""
        if self.event_bus is None or not self.session_id:
            return
        try:
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="judge",
                    level="info",
                    status=status,  # type: ignore[arg-type]
                    msg=msg,
                    pct=pct,
                    details={
                        "turn": turn_number,
                        "total_turns": total_turns,
                        **(details or {}),
                    },
                )
            )
        except Exception as emit_exc:
            logger.debug("=> judge SSE emit failed (non-fatal): %s", emit_exc)

    def _emit_findings_stream(
        self,
        decision: JudgeDecision,
        turn_number: int,
        total_turns: int,
    ) -> None:
        """Replay each model's structured findings as live [Judge] SSE lines.

        Emits, per ranked model: an 'Evaluating <model>' line, one line per
        finding, and a final 'Approving'/'Rejecting' line. Findings are carried
        in event.details so the leaderboard can render the full decision card.
        """
        if self.event_bus is None or not self.session_id:
            return
        # Map verdict => streamed verb (no if-else ladder).
        verdict_verb_map: Dict[str, str] = {
            "select": "Approving",
            "rank_only": "Ranking",
            "reject": "Rejecting",
        }
        for ranked_model in decision.ranked_models:
            self._emit_judge_event(
                turn_number=turn_number,
                total_turns=total_turns,
                status="running",
                msg=f"[Judge] Evaluating {ranked_model.model_name}",
                pct=80,
                details={"model_name": ranked_model.model_name, "phase": "evaluating"},
            )
            for finding in ranked_model.findings:
                self._emit_judge_event(
                    turn_number=turn_number,
                    total_turns=total_turns,
                    status="running",
                    msg=f"[Judge] {finding.label}: {finding.message}",
                    pct=85,
                    details={
                        "model_name": ranked_model.model_name,
                        "phase": "finding",
                        "dimension": finding.dimension,
                        "finding_status": finding.status,
                    },
                )
            verb = verdict_verb_map.get(ranked_model.verdict, "Reviewing")
            self._emit_judge_event(
                turn_number=turn_number,
                total_turns=total_turns,
                status="running",
                msg=f"[Judge] {verb} model {ranked_model.model_name} ({ranked_model.decision})",
                pct=90,
                details={
                    "model_name": ranked_model.model_name,
                    "phase": "decision",
                    "decision": ranked_model.decision,
                },
            )

    @staticmethod
    def _load_domain_reasoning(session_dir: Path) -> Optional[Dict[str, Any]]:
        # domain_reasoning.json is generated exactly once, upstream, during
        # metadata generation (see backend/routers/metadata.py and Stage 1.5
        # of run_pipeline.py). It is read-only here -- the judge never
        # triggers generation itself, and a missing file is tolerated.
        domain_reasoning_path = session_dir / "reports" / "domain_reasoning.json"
        if not domain_reasoning_path.is_file():
            return None
        try:
            return json.loads(domain_reasoning_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "=> failed to read domain_reasoning.json at %s: %s",
                domain_reasoning_path,
                exc,
            )
            return None

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
        self._persist_prompt_transcript(decision, reports_dir, turn)

    # Delimiter JudgeAgent uses to join the rendered system + user jinja prompts
    # into decision_trace.transcript. Splitting on it lets us dump each rendered
    # template separately for inspection.
    _PROMPT_PARTS_DELIMITER = "\n\n---\n\n"

    @classmethod
    def _persist_prompt_transcript(cls, decision: JudgeDecision, reports_dir: Path, turn: int) -> None:
        # The rendered judge prompt is already stored in decision_trace.transcript
        # (the full audit trail); also dump it as standalone text files in the
        # session dir so each rendered jinja prompt is directly inspectable
        # without parsing JSON.
        transcript = decision.decision_trace.transcript
        if not transcript:
            return
        # Combined prompt (system + user), canonical + per-turn archive.
        (reports_dir / "judge_prompt.txt").write_text(transcript, encoding="utf-8")
        (reports_dir / f"judge_prompt_turn_{turn}.txt").write_text(transcript, encoding="utf-8")

        # Also split the two rendered jinja templates into their own files when
        # the delimiter is present (system rubric/schema vs per-turn data).
        prompt_parts = transcript.split(cls._PROMPT_PARTS_DELIMITER, 1)
        if len(prompt_parts) == 2:
            system_prompt_text, user_prompt_text = prompt_parts
            (reports_dir / "judge_prompt_system.txt").write_text(system_prompt_text, encoding="utf-8")
            (reports_dir / "judge_prompt_user.txt").write_text(user_prompt_text, encoding="utf-8")
            (reports_dir / f"judge_prompt_system_turn_{turn}.txt").write_text(system_prompt_text, encoding="utf-8")
            (reports_dir / f"judge_prompt_user_turn_{turn}.txt").write_text(user_prompt_text, encoding="utf-8")

    def _write_judge_status(
        self,
        session_dir: Path,
        turn: int,
        max_turns: int,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        reports_dir = session_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        status_path = reports_dir / "judge_status.json"
        
        # Load existing logs or initialize
        existing_logs = []
        if status_path.is_file():
            try:
                data = json.loads(status_path.read_text(encoding="utf-8"))
                existing_logs = data.get("logs", [])
            except Exception:
                pass
        
        # Add new message to logs if it's different from the last one
        if not existing_logs or existing_logs[-1] != message:
            existing_logs.append(message)
            # Keep last 100 log statements to save space
            if len(existing_logs) > 100:
                existing_logs = existing_logs[-100:]

        try:
            status_path.write_text(json.dumps({
                "status": status,
                "progress": int(100 * (turn / max(1, max_turns))),
                "message": message,
                "turn": turn,
                "max_turns": max_turns,
                "logs": existing_logs,
                **(details or {}),
            }, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to write judge status: %s", e)
