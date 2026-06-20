"""Unified headless CLI runner for the full MITRA pipeline.

Runs the complete DAG from raw dataset to judge decision without the FastAPI
server or browser. All artifacts are written to a session directory under
WORKSPACE_ROOT/<user_id>/<session_id>/.

Usage::

    python -m backend.orchestration.run_pipeline \\
        --dataset path/to/train.csv \\
        --target  target_column \\
        --session-id my_run_001 \\
        [--user-id  cli_user] \\
        [--config   config.ini] \\
        [--provider anthropic] \\
        [--model    claude-sonnet-4-6] \\
        [--mode     local|ray] \\
        [--max-models 10] \\
        [-v]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from backend.config_loader import ConfigLoader
from backend.agents.metadata_gen_agent import (
    MetadataAgentRunner,
    MetadataGenerationInput,
    LlmSettingsResolver,
)
from backend.agents.training_orchestrator import TrainingOrchestrator
from backend.orchestration.eval_runner import EvalRunner
from backend.orchestration.events import TrainingEventBus
from backend.orchestration.judge_loop import EvalArtifacts, JudgeLoop
from backend.orchestration.plotting import PipelinePlotGenerator
from backend.orchestration.d2v_bridge import D2VBridge
from backend.orchestration.token_counter import TokenCounter
from backend.services.pipeline_prep import PipelinePrep

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full MITRA AutoML pipeline headless (--cli mode)."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--target", required=True,
        help="Name of the target / label column.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session identifier. Auto-generated UUID if not supplied.",
    )
    parser.add_argument(
        "--user-id",
        default="cli_user",
        help="User identifier used to scope the session workspace.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.ini. Defaults to config.ini in the repo root.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider override (e.g. 'anthropic', 'openai', 'gemini').",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model override (e.g. 'claude-sonnet-4-6', 'gpt-4o').",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "ray"],
        default="local",
        help="Training execution mode.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Maximum number of models to train (overrides config).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


class PipelineRunner:
    """Runs the full pipeline DAG for one session."""

    def __init__(
        self,
        config_loader: ConfigLoader,
        session_dir: Path,
        session_id: str,
        provider: str | None,
        model: str | None,
        execution_mode: str,
        max_judge_turns: int = 3,
        use_llm_judge: bool = True,
        verbose: bool = False,
    ) -> None:
        self.config_loader = config_loader
        self.session_dir = session_dir
        self.session_id = session_id
        self.provider = provider
        self.model = model
        self.execution_mode = execution_mode
        self.max_judge_turns = max_judge_turns
        self.use_llm_judge = use_llm_judge
        self.verbose = verbose

    def run(
        self,
        dataset_path: Path,
        target_column: str,
        max_models: int | None = None,
    ) -> dict:
        """Execute full pipeline and return a summary dict with artifact paths."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.session_dir / "data"
        reports_dir = self.session_dir / "reports"
        data_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Resolve LLM settings once; shared across all LLM-using agents.
        llm_resolver = LlmSettingsResolver(self.config_loader)
        llm_settings = llm_resolver.resolve(
            provider=self.provider,
            model=self.model,
        )
        logger.info(
            "=> pipeline start: session=%s provider=%s model=%s mode=%s",
            self.session_id,
            llm_settings.provider,
            llm_settings.model,
            self.execution_mode,
        )

        # Stage 1: metadata generation
        logger.info("=> [1/6] metadata generation ...")
        metadata_path = reports_dir / "metadata.json"
        metadata_result = MetadataAgentRunner().generate_metadata(
            MetadataGenerationInput(
                session_id=self.session_id,
                workspace_root=self.session_dir,
                llm_settings=llm_settings,
            )
        )
        metadata_path = metadata_result.metadata_path
        logger.info("=> metadata written: %s", metadata_path)

        # Stage 2: feature engineering -> feature_selection -> train/test split -> model selection
        logger.info("=> [2/6] pre-training prep (feature eng + model selection) ...")
        prep = PipelinePrep(
            config_loader=self.config_loader,
            session_dir=self.session_dir,
            llm_settings=llm_settings,
        )
        model_config_path = prep.run(
            raw_data_path=dataset_path,
            target_column=target_column,
            metadata_path=metadata_path,
            max_models=max_models,
        )
        logger.info("=> model_config written: %s", model_config_path)

        # Stage 3: training
        logger.info("=> [3/6] model training (mode=%s) ...", self.execution_mode)
        event_bus = TrainingEventBus()
        orchestrator = TrainingOrchestrator(
            model_library_root=self.config_loader.training_api.model_library_root,
            event_sink=event_bus,
        )

        train_path = self.session_dir / "data" / "train.csv"
        test_path = self.session_dir / "data" / "test.csv"
        summary_path = reports_dir / "training_summary.json"

        training_fn = {
            "local": orchestrator.prepare_and_execute_local,
            "ray": orchestrator.prepare_and_execute_ray,
        }[self.execution_mode]

        training_summary = training_fn(
            session_id=self.session_id,
            metadata_path=metadata_path,
            model_config_path=model_config_path,
            train_path=train_path,
            test_path=test_path,
            session_dir=self.session_dir,
            target_column=target_column,
            summary_path=summary_path,
        )
        logger.info(
            "=> training complete: %d models, %d completed, %d failed",
            training_summary.total_models,
            training_summary.completed,
            training_summary.failed,
        )

        # Resolve task_type from metadata for eval/judge stages
        task_type = self._read_task_type(metadata_path)
        token_counter = TokenCounter(session_dir=self.session_dir)

        # Stage 4: parallel eval (SHAP || overfitting || HPT)
        logger.info("=> [4/6] parallel evaluation (SHAP + overfitting + HPT) ...")
        engineered_csv = self.session_dir / "data" / "engineered_dataset.csv"
        if not engineered_csv.exists():
            # Fall back to train.csv if engineered version not yet split
            engineered_csv = train_path
        eval_runner = EvalRunner(
            session_id=self.session_id,
            session_dir=self.session_dir,
            task_type=task_type,
            target_column=target_column,
            verbose=self.verbose,
        )
        eval_output = eval_runner.run(
            training_summary=training_summary,
            engineered_dataset_path=engineered_csv,
        )
        logger.info("=> eval complete: shap=%d overfit=%d hpt=%s",
                    sum(1 for v in eval_output["shap_dirs"].values() if v),
                    sum(1 for v in eval_output["overfitting_dirs"].values() if v),
                    "ok" if eval_output["hpt_results_path"] else "skipped")

        # Stage 5: judge + feedback loop
        logger.info("=> [5/6] judge loop (max_turns=%d) ...", self.max_judge_turns)
        eval_artifacts = EvalArtifacts(
            shap_dirs=eval_output["shap_dirs"],
            overfitting_dirs=eval_output["overfitting_dirs"],
            hpt_results_path=eval_output["hpt_results_path"],
        )
        judge_loop = JudgeLoop(
            task_type=task_type,
            max_turns=self.max_judge_turns,
            use_llm=self.use_llm_judge,
            verbose=self.verbose,
        )
        # Load metadata dict for judge context
        metadata_dict = None
        if metadata_path.exists():
            import json as _json
            with metadata_path.open(encoding="utf-8") as meta_file:
                metadata_dict = _json.load(meta_file)

        judge_decision = judge_loop.run(
            eval_artifacts=eval_artifacts,
            training_summary=training_summary,
            session_dir=self.session_dir,
            dataset_id=dataset_path.stem,
            metadata=metadata_dict,
        )
        logger.info(
            "=> judge complete: selected=%s total_ranked=%d",
            judge_decision.selected_model,
            len(judge_decision.ranked_models),
        )

        # Post-hoc visualization stage: read already-written artifacts and dump
        # comprehensive plots. Decoupled and fully non-fatal so a plotting error
        # can never abort the pipeline (log + continue).
        plots_summary: dict[str, list[str]] = {}
        try:
            plots_summary = PipelinePlotGenerator(
                session_dir=self.session_dir
            ).generate_all()
            total_plots = sum(len(paths) for paths in plots_summary.values())
            populated_stages = sum(1 for paths in plots_summary.values() if paths)
            logger.info(
                "=> [plots] wrote %d plots across %d stages",
                total_plots,
                populated_stages,
            )
        except Exception as plotting_error:  # noqa: BLE001 - plotting is non-fatal
            logger.warning("=> [plots] generation failed (non-fatal): %s", plotting_error)

        # Stage 6: dataset2Vec write-back (non-fatal; runs in background)
        db_dir = Path(__file__).resolve().parents[2] / "DB"
        if db_dir.exists():
            logger.info("=> [6/6] dataset2Vec write-back ...")
            d2v_bridge = D2VBridge(db_dir=db_dir)
            d2v_bridge.write_back(
                csv_path=dataset_path,
                target_column=target_column,
                task_type=task_type,
                judge_decision=judge_decision,
            )
        else:
            logger.info("=> [6/6] DB/ dir absent; skipping dataset2Vec write-back.")

        # Log command for reproducibility (REQ #13)
        cmd_log = self.session_dir / "pipeline_command.txt"
        cmd_log.write_text(
            f"python -m backend.orchestration.run_pipeline"
            f" --dataset {dataset_path}"
            f" --target {target_column}"
            f" --session-id {self.session_id}"
            f" --provider {self.provider or llm_settings.provider}"
            f" --model {llm_settings.model}"
            f" --mode {self.execution_mode}",
            encoding="utf-8",
        )
        judge_decision_path = self.session_dir / "reports" / "judge_decision.json"
        logger.info("=> pipeline done. Artifacts in %s", self.session_dir)

        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "metadata_path": str(metadata_path),
            "model_config_path": str(model_config_path),
            "training_summary_path": str(summary_path),
            "judge_decision_path": str(judge_decision_path),
            "selected_model": judge_decision.selected_model,
            "total_models": training_summary.total_models,
            "completed_models": training_summary.completed,
            "failed_models": training_summary.failed,
            "token_usage": token_counter.summary(),
            "plots": plots_summary,
        }

    @staticmethod
    def _read_task_type(metadata_path: Path) -> str:
        """Read task_type from metadata.json; default to classification."""
        import json as _json
        if not metadata_path.exists():
            return "classification"
        with metadata_path.open(encoding="utf-8") as meta_file:
            meta = _json.load(meta_file)
        return meta.get("task_type", "classification")


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config) if args.config else repo_root / "config.ini"
    config_loader = ConfigLoader(config_path=config_path, repo_root=repo_root)

    session_id = args.session_id or str(uuid.uuid4())[:8]
    workspace_root = config_loader.paths.workspace_root
    session_dir = workspace_root / args.user_id / session_id

    runner = PipelineRunner(
        config_loader=config_loader,
        session_dir=session_dir,
        session_id=session_id,
        provider=args.provider,
        model=args.model,
        execution_mode=args.mode,
        verbose=args.verbose,
    )

    result = runner.run(
        dataset_path=Path(args.dataset).resolve(),
        target_column=args.target,
        max_models=args.max_models,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
