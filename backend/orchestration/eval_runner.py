"""Parallel evaluation runner: SHAP || overfitting || HPT.

Runs all three evaluation branches concurrently using ProcessPoolExecutor
(CPU-bound work) so wall-clock time equals the slowest branch, not the sum.
Results are assembled into a dict of artifact paths consumed by JudgeLoop.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Top-level functions needed so ProcessPoolExecutor can pickle them.

def _run_shap_for_model(
    model_name: str,
    model_path: str,
    dataset_path: str,
    target_column: str,
    shap_output_dir: str,
    session_id: str,
    max_shap_samples: int,
) -> Optional[str]:
    """Worker function: runs SHAP for one model in a subprocess."""
    # sys.path bootstrap for the subprocess (inherits parent env on Linux but
    # explicit is safer for cross-platform and future Ray workers).
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from backend.agents.evaluation.shap.runner import SHAPRunner
    runner = SHAPRunner(shap_output_dir=Path(shap_output_dir), session_id=session_id)
    result_dir = runner.run_for_model(
        model_name=model_name,
        model_path=Path(model_path),
        dataset_path=Path(dataset_path),
        target_column=target_column,
        max_shap_samples=max_shap_samples,
    )
    return str(result_dir) if result_dir else None


def _run_overfitting_for_model(
    input_json_path: str,
    output_dir: str,
    verbose: bool,
) -> Optional[str]:
    """Worker function: runs OverfittingAnalyzer for one model in a subprocess."""
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from backend.agents.evaluation.overfitting.overfitting_analysis import OverfittingAnalyzer
    analyzer = OverfittingAnalyzer(
        input_json_path=input_json_path,
        output_dir=output_dir,
        verbose=verbose,
    )
    result = analyzer.run()
    return json.dumps(result)


class OverfittingRunner:
    """Builds overfitting input JSONs from training results and runs OverfittingAnalyzer."""

    @staticmethod
    def _read_target_column(session_dir: Path) -> Optional[str]:
        """Return the target column from session metadata.json, or None if unresolvable."""
        for candidate in [
            session_dir / "reports" / "metadata.json",
            session_dir / "metadata.json",
        ]:
            if candidate.is_file():
                meta = json.loads(candidate.read_text(encoding="utf-8"))
                return meta.get("target_col") or meta.get("target_column")
        return None

    def run(
        self,
        training_summary: Any,
        session_dir: Path,
        dataset_path: Path,
        task_type: str,
        target_column: Optional[str] = None,
        verbose: bool = False,
    ) -> dict[str, Optional[str]]:
        """Run overfitting analysis for all trained models.

        Returns:
            {model_name: output_dir_path or None}
        """
        overfit_dir = session_dir / "evaluation" / "overfitting"
        overfit_dir.mkdir(parents=True, exist_ok=True)

        # Prefer the already-split CSVs so OverfittingAnalyzer gets separate
        # train and test arrays. Fall back to the single engineered CSV.
        train_csv = session_dir / "data" / "train.csv"
        test_csv = session_dir / "data" / "test.csv"
        if train_csv.is_file() and test_csv.is_file():
            primary_dataset_path = str(train_csv)
            test_dataset_path = str(test_csv)
        else:
            primary_dataset_path = str(dataset_path)
            test_dataset_path = None

        # Read target column from session metadata when not passed directly.
        resolved_target = target_column or self._read_target_column(session_dir)

        model_results: dict[str, Optional[str]] = {}
        models = getattr(training_summary, "models", []) or []

        with ProcessPoolExecutor() as pool:
            futures = {}
            for model_result in models:
                model_name = model_result.model_name
                model_output_dir = str(overfit_dir / model_name)
                os.makedirs(model_output_dir, exist_ok=True)

                # Build the input JSON expected by OverfittingAnalyzer.
                input_payload: dict[str, Any] = {
                    "model_type": task_type,
                    "model_name": model_name,
                    "dataset_path": primary_dataset_path,
                }
                if test_dataset_path:
                    input_payload["test_dataset_path"] = test_dataset_path
                if resolved_target:
                    input_payload["target_column"] = resolved_target
                if model_result.validation_score is not None:
                    primary_metric = "accuracy" if task_type == "classification" else "r2"
                    input_payload["test_metrics"] = {primary_metric: model_result.validation_score}

                input_json_path = os.path.join(model_output_dir, "overfitting_input.json")
                with open(input_json_path, "w", encoding="utf-8") as input_file:
                    json.dump(input_payload, input_file, indent=2)

                future = pool.submit(
                    _run_overfitting_for_model,
                    input_json_path=input_json_path,
                    output_dir=model_output_dir,
                    verbose=verbose,
                )
                futures[future] = model_name

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    future.result()
                    model_results[model_name] = str(overfit_dir / model_name)
                    logger.info("=> overfitting done: model=%s", model_name)
                except Exception as exc:
                    logger.warning("=> overfitting failed for %s: %s", model_name, exc)
                    model_results[model_name] = None

        return model_results


class HPTRunner:
    """Runs HyperparameterTuningAgent for the session."""

    def run(
        self,
        session_id: str,
        session_dir: Path,
        verbose: bool = False,
    ) -> Optional[Path]:
        """Run HPT and return path to hpt_results.json, or None on failure."""
        from backend.agents.evaluation.hpt.agent import HyperparameterTuningAgent

        # HPT uses session_root = Path(".mitra") / session_id by default;
        # override by setting MITRA_SESSION_ROOT before init.
        os.environ["MITRA_SESSION_ROOT"] = str(session_dir.parent)

        try:
            hpt_agent = HyperparameterTuningAgent(
                session_id=str(session_dir.name),
                verbose=verbose,
            )
            results = hpt_agent.run()
            hpt_output_path = session_dir / "evaluation" / "hpt" / "hpt_results.json"
            hpt_output_path.parent.mkdir(parents=True, exist_ok=True)
            hpt_output_path.write_text(
                json.dumps({"hpt_results": results}, indent=2), encoding="utf-8"
            )
            logger.info("=> HPT done: %d models tuned", len(results))
            return hpt_output_path
        except Exception as exc:
            logger.warning("=> HPT failed: %s", exc)
            return None


class EvalRunner:
    """Runs SHAP, overfitting, and HPT in parallel and assembles their outputs."""

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        task_type: str,
        target_column: str,
        max_shap_samples: int = 1000,
        verbose: bool = False,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.task_type = task_type
        self.target_column = target_column
        self.max_shap_samples = max_shap_samples
        self.verbose = verbose

    def run(
        self,
        training_summary: Any,
        engineered_dataset_path: Path,
        run_hpt: bool = True,
    ) -> dict[str, Any]:
        """Run all three eval branches in parallel.

        Returns a dict with keys: shap_dirs, overfitting_dirs, hpt_results_path
        consumed by JudgeLoop to build JudgeInput.
        """
        shap_output_dir = self.session_dir / "evaluation" / "shap"
        shap_output_dir.mkdir(parents=True, exist_ok=True)

        models = getattr(training_summary, "models", []) or []

        # Launch SHAP workers (one per model, in subprocess pool)
        shap_dirs: dict[str, Optional[str]] = {}
        with ProcessPoolExecutor() as pool:
            futures = {}
            for model_result in models:
                if not model_result.model_path:
                    logger.warning("=> no model_path for %s, skipping SHAP", model_result.model_name)
                    shap_dirs[model_result.model_name] = None
                    continue
                future = pool.submit(
                    _run_shap_for_model,
                    model_name=model_result.model_name,
                    model_path=model_result.model_path,
                    dataset_path=str(engineered_dataset_path),
                    target_column=self.target_column,
                    shap_output_dir=str(shap_output_dir),
                    session_id=self.session_id,
                    max_shap_samples=self.max_shap_samples,
                )
                futures[future] = model_result.model_name

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    shap_dirs[model_name] = future.result()
                    logger.info("=> SHAP done: model=%s", model_name)
                except Exception as exc:
                    logger.warning("=> SHAP failed for %s: %s", model_name, exc)
                    shap_dirs[model_name] = None

        # Overfitting and HPT run concurrently with each other (via executor)
        overfit_runner = OverfittingRunner()
        hpt_runner = HPTRunner()

        with ProcessPoolExecutor(max_workers=2) as pool:
            overfit_future = pool.submit(
                overfit_runner.run,
                training_summary=training_summary,
                session_dir=self.session_dir,
                dataset_path=engineered_dataset_path,
                task_type=self.task_type,
                target_column=self.target_column,
                verbose=self.verbose,
            )
            if run_hpt:
                hpt_future = pool.submit(
                    hpt_runner.run,
                    session_id=self.session_id,
                    session_dir=self.session_dir,
                    verbose=self.verbose,
                )
            else:
                hpt_future = None

            overfitting_dirs = overfit_future.result()
            hpt_results_path = hpt_future.result() if hpt_future else None

        return {
            "shap_dirs": shap_dirs,
            "overfitting_dirs": overfitting_dirs,
            "hpt_results_path": str(hpt_results_path) if hpt_results_path else None,
        }
