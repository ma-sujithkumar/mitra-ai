"""Parallel evaluation runner: SHAP || overfitting || HPT.

Runs all three evaluation branches concurrently using ProcessPoolExecutor
(CPU-bound work) so wall-clock time equals the slowest branch, not the sum.
Results are assembled into a dict of artifact paths consumed by JudgeLoop.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Empty
from typing import Any, Optional

from backend.orchestration.events import TrainingEvent, TrainingEventBus

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


def _run_shap_for_model_mp(
    result_queue: "multiprocessing.Queue",
    model_name: str,
    **kwargs: Any,
) -> None:
    """Process target: runs _run_shap_for_model and posts the outcome to the
    queue. Used (instead of ProcessPoolExecutor) so the parent can forcibly
    terminate a hung worker via Process.terminate()/kill() -- a stuck
    KernelExplainer otherwise blocks the pool's shutdown(wait=True) forever.
    """
    try:
        result_dir = _run_shap_for_model(model_name=model_name, **kwargs)
        result_queue.put((model_name, "ok", result_dir))
    except Exception as exc:
        result_queue.put((model_name, "error", str(exc)))


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
        timeout_sec: Optional[int] = None,
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
        total_models = len(models)
        completed_count = 0
        status_path = session_dir / "evaluation" / "overfitting_status.json"
        self._write_status(status_path, completed_count, total_models, "running", "Starting overfitting analysis...")

        pool = ProcessPoolExecutor()
        try:
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

            # Monitor the running futures with a configurable timeout
            futures_list = list(futures.keys())
            try:
                # We retrieve completed futures with the timeout limit
                iterator = as_completed(futures_list, timeout=timeout_sec)
                while True:
                    try:
                        completed_future = next(iterator)
                        model_name = futures[completed_future]
                        completed_count += 1
                        try:
                            completed_future.result()
                            model_results[model_name] = str(overfit_dir / model_name)
                            logger.info("=> overfitting done: model=%s", model_name)
                            self._write_status(status_path, completed_count, total_models, "running", f"Overfitting check done for {model_name}")
                        except Exception as exc:
                            logger.warning("=> overfitting failed for %s: %s", model_name, exc)
                            model_results[model_name] = None
                            self._write_status(status_path, completed_count, total_models, "running", f"Overfitting check failed for {model_name}: {exc}")
                        # Remove from the list of pending futures
                        futures_list.remove(completed_future)
                    except StopIteration:
                        break
            except TimeoutError:
                # Handles when the next future doesn't complete within the remaining timeout
                logger.warning("=> overfitting analysis timed out after %ss", timeout_sec)
                for remaining_future in futures_list:
                    m_name = futures[remaining_future]
                    model_results[m_name] = None
                    remaining_future.cancel()
                    self._write_status(status_path, completed_count, total_models, "running", f"Overfitting check timed out for {m_name}")
        finally:
            # Shutdown pool without blocking so we do not freeze the main loop
            pool.shutdown(wait=False, cancel_futures=True)

        self._write_status(status_path, completed_count, total_models, "completed", "Overfitting analysis completed successfully.")
        return model_results

    def _write_status(self, path: Path, completed: int, total: int, status: str, message: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "status": status,
                "progress": int(100 * (completed / max(1, total))),
                "message": message,
                "completed_models": completed,
                "total_models": total
            }, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to write overfitting status: %s", e)


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
        event_bus: Optional[TrainingEventBus] = None,
        shap_timeout_sec: int = 180,
        overfitting_timeout_sec: int = 120,
        shap_skip_model_classes: Optional[list[str]] = None,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.task_type = task_type
        self.target_column = target_column
        self.max_shap_samples = max_shap_samples
        self.verbose = verbose
        self.event_bus = event_bus
        self.shap_timeout_sec = shap_timeout_sec
        self.overfitting_timeout_sec = overfitting_timeout_sec
        # Use a set for O(1) membership checks in the launch loop.
        self.shap_skip_model_classes: set[str] = set(shap_skip_model_classes or [])

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
        if self.event_bus:
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg="Evaluation Started",
                    pct=10,
                )
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg="Loading Validation Dataset",
                    pct=20,
                )
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg="Running Metrics",
                    pct=30,
                )
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg="Generating SHAP Values",
                    pct=40,
                )
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="shap",
                    level="info",
                    status="running",
                    msg="[SHAP EXPLAINER] Starting SHAP explanation analysis for all trained models...",
                    pct=10,
                )
            )

        shap_output_dir = self.session_dir / "evaluation" / "shap"
        shap_output_dir.mkdir(parents=True, exist_ok=True)

        models = getattr(training_summary, "models", []) or []

        # Launch SHAP workers (one per model, each its own subprocess). A plain
        # ProcessPoolExecutor is not used here: its shutdown(wait=True) joins
        # every worker unconditionally, so one hung KernelExplainer (observed
        # with SVC multiclass) would block the pipeline forever. Process gives
        # us terminate()/kill() so a model that exceeds shap_timeout_sec is
        # killed and skipped instead of stalling everything after it.
        shap_dirs: dict[str, Optional[str]] = {}
        queued_models = []
        for model_result in models:
            if not model_result.model_path:
                logger.warning("=> no model_path for %s, skipping SHAP", model_result.model_name)
                shap_dirs[model_result.model_name] = None
                if self.event_bus:
                    self.event_bus.emit(
                        TrainingEvent(
                            session_id=self.session_id,
                            stage="shap",
                            level="warn",
                            status="running",
                            msg=f"[SHAP EXPLAINER] Skipping SHAP for model {model_result.model_name} (failed to train, no model path)",
                            pct=10,
                        )
                    )
                continue
            # Skip models whose class is in the configured skip list (e.g. SVC, NuSVC).
            # These use KernelExplainer which consistently exceeds SHAP_TIMEOUT_SEC and
            # produces no output. Skipping upfront avoids the full timeout wait.
            if model_result.model_name in self.shap_skip_model_classes:
                logger.info(
                    "=> SHAP skip: model=%s is in shap_skip_model_classes, skipping SHAP",
                    model_result.model_name,
                )
                shap_dirs[model_result.model_name] = None
                if self.event_bus:
                    self.event_bus.emit(
                        TrainingEvent(
                            session_id=self.session_id,
                            stage="shap",
                            level="info",
                            status="running",
                            msg=f"[SHAP EXPLAINER] SHAP skipped for model {model_result.model_name} (KernelSVM - not supported efficiently, skipped per configuration)",
                            pct=10,
                        )
                    )
                continue
            queued_models.append(model_result)

        total_models = len(queued_models)
        completed_count = 0
        heartbeat_interval_sec = 10.0
        max_concurrent = max(1, os.cpu_count() or 4)

        self._write_shap_status(completed_count, total_models or 1, "running", "Starting SHAP analysis...")

        result_queue: "multiprocessing.Queue" = multiprocessing.Queue()
        running_processes: dict[str, multiprocessing.Process] = {}
        start_times: dict[str, float] = {}

        def _launch(model_result: Any) -> None:
            process = multiprocessing.Process(
                target=_run_shap_for_model_mp,
                kwargs=dict(
                    result_queue=result_queue,
                    model_name=model_result.model_name,
                    model_path=model_result.model_path,
                    dataset_path=str(engineered_dataset_path),
                    target_column=self.target_column,
                    shap_output_dir=str(shap_output_dir),
                    session_id=self.session_id,
                    max_shap_samples=self.max_shap_samples,
                ),
            )
            process.start()
            running_processes[model_result.model_name] = process
            start_times[model_result.model_name] = time.monotonic()

        while queued_models and len(running_processes) < max_concurrent:
            _launch(queued_models.pop(0))

        last_heartbeat = time.monotonic()
        while running_processes:
            try:
                model_name, outcome, payload = result_queue.get(timeout=heartbeat_interval_sec)
            except Empty:
                model_name = None
                outcome = None
                payload = None

            now = time.monotonic()
            progress_pct = 10 + int(80 * (completed_count / max(1, total_models)))

            if model_name is not None:
                process = running_processes.pop(model_name, None)
                if process is not None:
                    process.join()
                start_times.pop(model_name, None)
                completed_count += 1
                progress_pct = 10 + int(80 * (completed_count / max(1, total_models)))
                if outcome == "ok":
                    shap_dirs[model_name] = payload
                    logger.info("=> SHAP done: model=%s", model_name)
                    if self.event_bus:
                        self.event_bus.emit(
                            TrainingEvent(
                                session_id=self.session_id,
                                stage="shap",
                                level="info",
                                status="running",
                                msg=f"[SHAP EXPLAINER] Finished SHAP value computation for model: {model_name} ({completed_count}/{total_models})",
                                pct=progress_pct,
                            )
                        )
                    self._write_shap_status(completed_count, total_models, "running", f"Finished SHAP value computation for model: {model_name} ({completed_count}/{total_models})")
                else:
                    shap_dirs[model_name] = None
                    logger.warning("=> SHAP failed for %s: %s", model_name, payload)
                    if self.event_bus:
                        self.event_bus.emit(
                            TrainingEvent(
                                session_id=self.session_id,
                                stage="shap",
                                level="warn",
                                status="running",
                                msg=f"[SHAP EXPLAINER] SHAP analysis failed for model {model_name}: {payload}",
                                pct=progress_pct,
                            )
                        )
                    self._write_shap_status(completed_count, total_models, "running", f"SHAP analysis failed for model {model_name}: {payload}")
                if queued_models:
                    _launch(queued_models.pop(0))

            # Kill any model whose SHAP run has exceeded the hard timeout.
            for timed_out_name in [
                name for name, started_at in start_times.items()
                if now - started_at > self.shap_timeout_sec
            ]:
                process = running_processes.pop(timed_out_name, None)
                start_times.pop(timed_out_name, None)
                if process is not None:
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                completed_count += 1
                progress_pct = 10 + int(80 * (completed_count / max(1, total_models)))
                shap_dirs[timed_out_name] = None
                logger.warning(
                    "=> SHAP timed out for %s after %ds, process killed",
                    timed_out_name, self.shap_timeout_sec,
                )
                if self.event_bus:
                    self.event_bus.emit(
                        TrainingEvent(
                            session_id=self.session_id,
                            stage="shap",
                            level="warn",
                            status="running",
                            msg=f"[SHAP EXPLAINER] SHAP analysis for model {timed_out_name} exceeded {self.shap_timeout_sec}s timeout, skipped",
                            pct=progress_pct,
                        )
                    )
                self._write_shap_status(completed_count, total_models, "running", f"SHAP analysis for model {timed_out_name} timed out, skipped")
                if queued_models:
                    _launch(queued_models.pop(0))

            if (
                model_name is None
                and not any(now - started_at > self.shap_timeout_sec for started_at in start_times.values())
                and self.event_bus
                and running_processes
                and now - last_heartbeat >= heartbeat_interval_sec
            ):
                still_running_names = list(running_processes.keys())
                self.event_bus.emit(
                    TrainingEvent(
                        session_id=self.session_id,
                        stage="shap",
                        level="info",
                        status="running",
                        msg=f"[SHAP EXPLAINER] Still computing SHAP values... ({completed_count}/{total_models} done, running: {', '.join(still_running_names[:3])})",
                        pct=10 + int(80 * (completed_count / max(1, total_models))),
                    )
                )
                self._write_shap_status(completed_count, total_models, "running", f"Still computing SHAP values... ({completed_count}/{total_models} done)")
                last_heartbeat = now

        self._write_shap_status(completed_count, total_models or 1, "completed", "SHAP explanation analysis completed successfully.")

        if self.event_bus:
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="shap",
                    level="info",
                    status="completed",
                    msg="[SHAP EXPLAINER] SHAP explanation analysis completed successfully.",
                    pct=100,
                )
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg="Computing Drift Metrics",
                    pct=60,
                )
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="evaluation",
                    level="info",
                    status="running",
                    msg="Computing Overfitting Gaps",
                    pct=80,
                )
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="overfitting",
                    level="info",
                    status="running",
                    msg="[OVERFITTING] Starting overfitting analysis (checking train vs. validation gaps)...",
                    pct=10,
                )
            )

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
                timeout_sec=self.overfitting_timeout_sec,
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

        if self.event_bus:
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="overfitting",
                    level="info",
                    status="completed",
                    msg="[OVERFITTING] Overfitting analysis completed successfully.",
                    pct=100,
                )
            )
            self.event_bus.emit(
                TrainingEvent(
                    session_id=self.session_id,
                    stage="evaluation",
                    level="info",
                    status="completed",
                    msg="Evaluation Completed",
                    pct=100,
                )
            )

        return {
            "shap_dirs": shap_dirs,
            "overfitting_dirs": overfitting_dirs,
            "hpt_results_path": str(hpt_results_path) if hpt_results_path else None,
        }

    def _write_shap_status(self, completed: int, total: int, status: str, message: str) -> None:
        try:
            status_path = self.session_dir / "evaluation" / "shap_status.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps({
                "status": status,
                "progress": int(100 * (completed / max(1, total))),
                "message": message,
                "completed_models": completed,
                "total_models": total
            }, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to write SHAP status: %s", e)
