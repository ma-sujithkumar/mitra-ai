import configparser
import gc
import glob
import logging
import os
import shutil
import sys
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

# Bootstrap sys.path so model_library imports resolve from any cwd, same
# convention as epic_4/overfitting_analysis_tool/overfitting_analysis.py.
_INI_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.ini")
_boot_parser = configparser.ConfigParser()
_boot_parser.read(_INI_PATH)
_MODEL_LIBRARY_ROOT = _boot_parser.get("paths", "model_library_root")
if _MODEL_LIBRARY_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_LIBRARY_ROOT)

# Ray spawns worker processes as fresh subprocesses -- they inherit
# environment variables but NOT this process's in-memory sys.path. Ray
# deserializes a remote call's arguments (e.g. CommonData, which needs
# `core.data_bundle` to unpickle) before it even imports this module, so
# sys.path.insert above is not enough: PYTHONPATH must be set so every
# worker's own interpreter startup adds model_library_root to sys.path.
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _MODEL_LIBRARY_ROOT not in _existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(
        path_entry for path_entry in [_MODEL_LIBRARY_ROOT, _existing_pythonpath] if path_entry
    )

# By default Ray blanks CUDA_VISIBLE_DEVICES for any worker assigned
# num_gpus=0 (the xgboost/sklearn case in MODEL_FAMILY_RESOURCES below). A
# CUDA-enabled xgboost build still probes the CUDA driver on init regardless
# of device="cpu", and an empty CUDA_VISIBLE_DEVICES makes that probe raise
# cudaErrorNoDevice instead of finding zero devices. Opting out of the
# override keeps the GPU visible (but unused, since device="cpu" is forced
# in execute_model_trial) so the probe succeeds.
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Ray already owns parallelism across (dataset, model) units via n_parallel
# workers. Without these caps, EACH worker's model would ALSO fan out: BLAS/
# OpenMP backends (numpy, HistGradientBoosting, MLP, GradientBoosting) spawn one
# thread per core, and joblib-based estimators (RandomForest/ExtraTrees/KNN/
# Bagging/XGB with n_jobs=-1) fork one SUBPROCESS per core. On a 16-core box
# that is up to n_parallel x 16 processes/threads -- a memory fork-bomb (each
# child copies the working set) and CPU oversubscription. These env vars cap the
# BLAS/OpenMP thread fan-out to 1; the joblib PROCESS fan-out is capped
# separately by forcing n_jobs=1 (MODEL_NAME_FORCED_HYPERPARAMETERS below), since
# joblib counts os.cpu_count() regardless of these thread vars.
for _thread_env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env_var] = "1"

import numpy as np
import optuna
import ray
import torch
from core.data_bundle import CommonData, DataBundle
from metrics.evaluators import MetricsResult, compute_metrics
from ml_kit import MLKit

from d2v_core.sampling import infer_task_type, standardize_columns
from d2v_core.schema import LeaderboardEntry, LeaderboardRecord
from d2v_core.store import MetaKnowledgeStore, utc_now_isoformat

logger = logging.getLogger(__name__)

OPTUNA_SAMPLER_DISPATCH: dict = {
    "tpe": optuna.samplers.TPESampler,
    "random": optuna.samplers.RandomSampler,
}

OPTUNA_PRUNER_DISPATCH: dict = {
    "median": optuna.pruners.MedianPruner,
    "none": optuna.pruners.NopPruner,
}

METRIC_DIRECTION_TO_OPTUNA: dict[str, str] = {"max": "maximize", "min": "minimize"}

# xgboost is dispatched with num_gpus=0.0 (CPU-only, freely parallel) per
# MODEL_FAMILY_RESOURCES below, but a CUDA-enabled xgboost build still probes
# the CUDA driver on init even with device unset, which raises cudaErrorNoDevice
# once Ray hides the GPU from a num_gpus=0 worker. Forcing device="cpu" skips
# that probe entirely.
MODEL_FAMILY_FORCED_HYPERPARAMETERS: dict[str, dict] = {
    "xgboost": {"device": "cpu"},
}

# Per-model-name forced overrides, applied on TOP of the family ones. These are
# the exact estimators that model_library defaults to n_jobs=-1 (verified
# against model_library/config/config.yaml). Forcing n_jobs=1 stops each one
# from forking ~os.cpu_count() joblib subprocesses inside an already-parallel
# Ray worker (the real cause of the 6 GB transient memory spike that OOM'd the
# node). XGB also accepts n_jobs and is included here.
MODEL_NAME_FORCED_HYPERPARAMETERS: dict[str, dict] = {
    "RandomForestClassifier": {"n_jobs": 1},
    "ExtraTreesClassifier": {"n_jobs": 1},
    "KNeighborsClassifier": {"n_jobs": 1},
    "BaggingClassifier": {"n_jobs": 1},
    "XGBClassifier": {"n_jobs": 1},
}

# Family classification by model_name prefix -> Ray resource request + Optuna
# concurrency policy. Checked in order; anything not matched is "sklearn"
# (CPU-only, freely parallel).
MODEL_FAMILY_PREFIXES: dict[str, str] = {
    "PyTorch": "pytorch",
    "XGB": "xgboost",
}

MODEL_FAMILY_RESOURCES: dict[str, dict[str, float]] = {
    "pytorch": {"num_cpus": 1, "num_gpus": 1.0},
    "xgboost": {"num_cpus": 1, "num_gpus": 0.0},
    "sklearn": {"num_cpus": 1, "num_gpus": 0.0},
}

METRIC_DIRECTION: dict[str, str] = {
    "accuracy": "max",
    "f1_macro": "max",
    "f1_weighted": "max",
    "precision_macro": "max",
    "recall_macro": "max",
    "mse": "min",
    "rmse": "min",
    "mae": "min",
    "r2": "max",
}


def classify_model_family(model_name: str) -> str:
    for prefix, family in MODEL_FAMILY_PREFIXES.items():
        if model_name.startswith(prefix):
            return family
    return "sklearn"


def metrics_result_to_dict(metrics: MetricsResult) -> dict:
    """Drops None-valued fields (the opposite task type's metrics) so only the
    metrics relevant to this task_type are stored in the leaderboard."""
    full_dict = asdict(metrics)
    return {key: value for key, value in full_dict.items() if value is not None and key not in ("task_type", "model_name")}


def execute_model_trial(
    model_name: str,
    hyperparameters: dict,
    common: CommonData,
    task_type: str,
) -> MetricsResult:
    """The SINGLE shared training core used by both the Optuna sweep
    (core/sweep.py -> run_optuna_trial) and Phase 3 verification
    (core/verify.py -> run_verification_trial). Trains model_name with the
    given hyperparameters on `common`, evaluates on common.X_test/y_test, and
    frees all GPU/CPU memory before returning -- no trained model is ever
    persisted here."""
    # Family overrides (e.g. xgboost device=cpu) first, then per-model-name
    # overrides (e.g. n_jobs=1) on top -- both win over any suggested value.
    family_overrides = MODEL_FAMILY_FORCED_HYPERPARAMETERS.get(classify_model_family(model_name), {})
    name_overrides = MODEL_NAME_FORCED_HYPERPARAMETERS.get(model_name, {})
    hyperparameters = {**hyperparameters, **family_overrides, **name_overrides}
    data_bundle = DataBundle(common=common, hyperparameters=hyperparameters)
    kit = MLKit(model_name=model_name, data=data_bundle, training_mode="full_train")
    kit.train()
    predictions = kit.test()
    metrics = compute_metrics(common.y_test, predictions, task_type, model_name)

    del kit
    del data_bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def build_suggestions(trial, model_name: str, search_spaces: dict) -> dict:
    """Converts one model's search_spaces.json entries into optuna
    trial.suggest_* calls. Models with no entry in search_spaces.json (or an
    empty list) get an empty dict -- MLKit then trains with the model's
    config.yaml defaults (a single default-config trial)."""
    parameter_specs = search_spaces.get(model_name, [])
    suggested_hyperparameters: dict = {}

    PARAM_TYPE_SUGGEST_DISPATCH = {
        "int": lambda spec: trial.suggest_int(spec["name"], spec["low"], spec["high"], log=spec.get("log", False)),
        "float": lambda spec: trial.suggest_float(spec["name"], spec["low"], spec["high"], log=spec.get("log", False)),
    }

    for parameter_spec in parameter_specs:
        suggest_fn = PARAM_TYPE_SUGGEST_DISPATCH[parameter_spec["type"]]
        suggested_hyperparameters[parameter_spec["name"]] = suggest_fn(parameter_spec)

    return suggested_hyperparameters


def _optuna_objective(
    trial: optuna.Trial,
    model_name: str,
    search_spaces: dict,
    common: CommonData,
    task_type: str,
    primary_metric: str,
) -> float:
    hyperparameters = build_suggestions(trial, model_name, search_spaces)
    metrics = execute_model_trial(model_name, hyperparameters, common, task_type)
    metrics_dict = metrics_result_to_dict(metrics)
    trial.set_user_attr("hyperparameters", hyperparameters)
    trial.set_user_attr("metrics", metrics_dict)
    return metrics_dict[primary_metric]


def run_optuna_study(
    dataset_id: str,
    model_name: str,
    common: CommonData,
    task_type: str,
    search_spaces: dict,
    optuna_storage: str,
    n_trials: int,
    primary_metric: str,
    optuna_sampler: str,
    optuna_pruner: str,
    study_timeout_seconds: int,
) -> Optional[LeaderboardEntry]:
    """Wraps execute_model_trial in an Optuna study for one (dataset_id,
    model_name) unit. Returns None when every trial failed (e.g. NuSVC with
    infeasible nu for this dataset's class distribution) so the caller can
    skip it rather than crashing the entire sweep."""
    study_name = f"{dataset_id}__{model_name}"
    direction = METRIC_DIRECTION_TO_OPTUNA[METRIC_DIRECTION[primary_metric]]
    sampler = OPTUNA_SAMPLER_DISPATCH[optuna_sampler](seed=42)
    pruner = OPTUNA_PRUNER_DISPATCH[optuna_pruner]()

    study = optuna.create_study(
        study_name=study_name,
        storage=optuna_storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Count ALL past trial states (COMPLETE + FAIL + PRUNED) toward the budget
    # so that models that always fail (e.g. NuSVC infeasible nu) do not loop
    # forever trying to accumulate n_trials COMPLETE results.
    n_all_past_trials = len(study.trials)
    n_remaining_trials = max(0, n_trials - n_all_past_trials)
    if n_remaining_trials > 0:
        study.optimize(
            lambda trial: _optuna_objective(
                trial, model_name, search_spaces, common, task_type, primary_metric
            ),
            n_trials=n_remaining_trials,
            # Wall-time cap: stops after study_timeout_seconds even if n_trials
            # haven't all completed. Prevents runaway trials on large datasets.
            timeout=study_timeout_seconds,
            # Catch all sklearn/torch model errors (infeasible params, singular
            # matrices, convergence failures, etc.) so a single bad trial does
            # not abort the Optuna study and crash the Ray task.
            catch=(Exception,),
        )

    complete_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        logger.warning(
            "=> run_optuna_study: all %d trials failed/timed-out for dataset_id='%s' model='%s' -- skipping.",
            len(study.trials), dataset_id, model_name,
        )
        return None

    best_trial = study.best_trial

    # Guard 1: best trial must have user_attrs set by _optuna_objective.
    # A trial caught mid-execution by catch=(Exception,) is marked FAIL (not
    # COMPLETE), so this should always be present -- but defend anyway.
    if "hyperparameters" not in best_trial.user_attrs or "metrics" not in best_trial.user_attrs:
        logger.warning(
            "=> run_optuna_study: best trial for dataset_id='%s' model='%s' "
            "is missing user_attrs (incomplete execution) -- skipping.",
            dataset_id, model_name,
        )
        return None

    # Guard 2: for maximize metrics, a best value of 0.0 means the model
    # predicted a single class on every trial (zero_division=0 in f1_score).
    # That result is degenerate and should not pollute the leaderboard.
    metric_direction = METRIC_DIRECTION[primary_metric]
    best_value = best_trial.value
    if metric_direction == "max" and best_value <= 0.0:
        logger.warning(
            "=> run_optuna_study: best trial value %.4f <= 0.0 for maximize metric '%s' "
            "on dataset_id='%s' model='%s' -- degenerate result, skipping.",
            best_value, primary_metric, dataset_id, model_name,
        )
        return None

    return LeaderboardEntry(
        rank=1,
        model_name=model_name,
        hyperparameters=best_trial.user_attrs["hyperparameters"],
        metrics=best_trial.user_attrs["metrics"],
        n_trials=len(study.trials),
    )


def load_dataset_common_data(corpus_dir: str, dataset_id: str) -> tuple[CommonData, str]:
    """Reads {dataset_id}.npz directly (X_train/y_train/X_test/y_test) for the
    sweep/verify training path. Reuses infer_task_type/standardize_columns from
    core/sampling.py rather than duplicating that logic."""
    npz_path = os.path.join(corpus_dir, f"{dataset_id}.npz")
    npz_data = np.load(npz_path, allow_pickle=True)
    X_train = standardize_columns(np.asarray(npz_data["X_train"], dtype=np.float64))
    X_test = standardize_columns(np.asarray(npz_data["X_test"], dtype=np.float64))
    y_train = np.asarray(npz_data["y_train"], dtype=np.float64).reshape(-1)
    y_test = np.asarray(npz_data["y_test"], dtype=np.float64).reshape(-1)

    task_type = str(npz_data["task_type"]) if "task_type" in npz_data else infer_task_type(y_train)
    common = CommonData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return common, task_type


@ray.remote
def _run_sweep_unit_remote(
    dataset_id: str,
    model_name: str,
    common: CommonData,
    task_type: str,
    search_spaces: dict,
    optuna_storage: str,
    n_trials: int,
    primary_metric: str,
    optuna_sampler: str,
    optuna_pruner: str,
    study_timeout_seconds: int,
) -> Optional[LeaderboardEntry]:
    return run_optuna_study(
        dataset_id=dataset_id,
        model_name=model_name,
        common=common,
        task_type=task_type,
        search_spaces=search_spaces,
        optuna_storage=optuna_storage,
        n_trials=n_trials,
        primary_metric=primary_metric,
        optuna_sampler=optuna_sampler,
        optuna_pruner=optuna_pruner,
        study_timeout_seconds=study_timeout_seconds,
    )


class MemoryJanitor:
    """Hourly driver-side safety net: gc.collect() + torch.cuda.empty_cache() +
    wipe scratch_dir. Runs as a daemon thread in the driver process (per-trial
    cleanup already happens inside execute_model_trial on every Ray worker --
    this is a backstop, not the primary cleanup mechanism)."""

    def __init__(self, scratch_dir: str, cleanup_interval_seconds: int) -> None:
        self.scratch_dir = scratch_dir
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _cleanup_once(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.path.isdir(self.scratch_dir):
            for entry_name in os.listdir(self.scratch_dir):
                entry_path = os.path.join(self.scratch_dir, entry_name)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)
        logger.info("=> janitor: ran cleanup at %s.", utc_now_isoformat())

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.cleanup_interval_seconds):
            self._cleanup_once()

    def start(self) -> None:
        os.makedirs(self.scratch_dir, exist_ok=True)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("=> janitor thread started (interval=%ds).", self.cleanup_interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("=> janitor thread stopped.")


class LeaderboardSweep:
    """Enumerates (dataset_id, model_name) units not yet completed, dispatches
    each to Ray with resources from MODEL_FAMILY_RESOURCES (sklearn/xgboost run
    freely parallel on CPU; pytorch requests num_gpus=1.0 so Ray's scheduler
    serializes them on the single GPU), then groups results per dataset_id into
    a LeaderboardRecord and writes it to the store."""

    def __init__(self, store: MetaKnowledgeStore, search_spaces: dict, sweep_config: dict) -> None:
        self.store = store
        self.search_spaces = search_spaces
        self.sweep_config = sweep_config

    def enumerate_units(self, dataset_ids: list[str], model_names: list[str]) -> list[tuple[str, str]]:
        completed = self.store.completed_units()
        return [
            (dataset_id, model_name)
            for dataset_id in dataset_ids
            for model_name in model_names
            if (dataset_id, model_name) not in completed
        ]

    def run(self, corpus_dir: str, dataset_ids: list[str], model_names: list[str]) -> int:
        if not ray.is_initialized():
            # Each worker resident-imports torch+xgboost+sklearn (~660 MB) BEFORE
            # any training. On a small-RAM box, n_parallel workers + the plasma
            # object store must fit in total RAM. We cap the plasma store
            # explicitly (our CommonData objects are tiny -- KB to a few MB) so
            # Ray does not grab its default 30% of RAM and starve the workers.
            object_store_memory_bytes = int(self.sweep_config["object_store_memory_gb"] * 1024 ** 3)
            ray.init(
                num_cpus=self.sweep_config.get("n_parallel"),
                num_gpus=1 if torch.cuda.is_available() else 0,
                object_store_memory=object_store_memory_bytes,
            )

        units = self.enumerate_units(dataset_ids, model_names)
        if len(units) == 0:
            logger.info("=> LeaderboardSweep.run: no remaining units (all already completed).")
            return 0

        # Pre-init SQLite schema in driver before workers touch the DB to avoid
        # concurrent CREATE TABLE races across Ray worker processes.
        optuna.storages.RDBStorage(self.sweep_config["optuna_storage"])

        primary_metric = self.sweep_config["primary_metric"]
        is_maximize = METRIC_DIRECTION[primary_metric] == "max"
        leaderboard_top_n = self.sweep_config["leaderboard_top_n"]
        study_timeout_seconds = self.sweep_config["study_timeout_seconds"]
        large_dataset_row_threshold = self.sweep_config["large_dataset_row_threshold"]
        large_dataset_n_parallel = self.sweep_config["large_dataset_n_parallel"]
        # Use a set for O(1) membership checks in the dispatch loop
        large_dataset_skip_models = set(self.sweep_config.get("large_dataset_skip_models", []))

        # Group pending units by dataset so we can process one dataset at a time.
        # Dispatching all 2420 units (121 datasets x 20 models) simultaneously
        # fills Ray's object store with all 121 CommonData arrays at once => OOM.
        # Chunked dispatch keeps at most one dataset's data live in the plasma
        # store at any given time.
        units_by_dataset: dict[str, list[str]] = {}
        for dataset_id, model_name in units:
            units_by_dataset.setdefault(dataset_id, []).append(model_name)

        total_units_completed = 0
        n_datasets_total = len(units_by_dataset)

        for dataset_index, (dataset_id, pending_model_names) in enumerate(units_by_dataset.items(), start=1):
            logger.info(
                "=> LeaderboardSweep: dataset %d/%d id='%s' dispatching %d models.",
                dataset_index, n_datasets_total, dataset_id, len(pending_model_names),
            )

            common, task_type = load_dataset_common_data(corpus_dir, dataset_id)
            # Save metadata scalars before handing CommonData to Ray so we can
            # build the LeaderboardRecord after freeing the plasma object.
            n_rows = int(common.X_train.shape[0])
            n_cols = int(common.X_train.shape[1])
            target_cardinality = len(np.unique(common.y_train)) if task_type == "classification" else 0
            # Total rows (train + test) used to decide if this is a large dataset.
            n_rows_total = n_rows + int(common.X_test.shape[0])
            is_large_dataset = n_rows_total >= large_dataset_row_threshold

            if is_large_dataset and large_dataset_skip_models:
                skip_here = [m for m in pending_model_names if m in large_dataset_skip_models]
                if skip_here:
                    logger.info(
                        "=> dataset_id='%s' has %d rows (>= %d threshold): skipping O(n^2) models %s.",
                        dataset_id, n_rows_total, large_dataset_row_threshold, skip_here,
                    )
                pending_model_names = [m for m in pending_model_names if m not in large_dataset_skip_models]

            # ray.put() places CommonData once in the plasma store; all model
            # workers for this dataset share the same object ref (no N-way copies).
            common_ref = ray.put(common)
            del common  # drop local reference; plasma holds the canonical copy

            # For large datasets, dispatch only large_dataset_n_parallel tasks
            # at a time and collect before dispatching the next window. This
            # prevents 15 x GBM-on-130K-rows workers from running concurrently
            # and exhausting node RAM. Small datasets dispatch all at once.
            dispatch_window_size = large_dataset_n_parallel if is_large_dataset else len(pending_model_names)
            entries: list[LeaderboardEntry] = []
            models_remaining = list(pending_model_names)

            while models_remaining:
                window_models = models_remaining[:dispatch_window_size]
                models_remaining = models_remaining[dispatch_window_size:]

                window_futures: list[tuple[str, ray.ObjectRef]] = []
                for model_name in window_models:
                    resources = MODEL_FAMILY_RESOURCES[classify_model_family(model_name)]
                    future = _run_sweep_unit_remote.options(
                        num_cpus=resources["num_cpus"], num_gpus=resources["num_gpus"]
                    ).remote(
                        dataset_id,
                        model_name,
                        common_ref,  # shared plasma ref, not a copy per task
                        task_type,
                        self.search_spaces,
                        self.sweep_config["optuna_storage"],
                        self.sweep_config["n_trials_per_model"],
                        primary_metric,
                        self.sweep_config["optuna_sampler"],
                        self.sweep_config["optuna_pruner"],
                        study_timeout_seconds,
                    )
                    window_futures.append((model_name, future))

                # Collect this window fully before dispatching the next batch.
                for model_name, future in window_futures:
                    leaderboard_entry = ray.get(future)
                    if leaderboard_entry is None:
                        logger.warning(
                            "=> skipping model='%s' dataset_id='%s': no valid trials.",
                            model_name, dataset_id,
                        )
                        continue
                    entries.append(leaderboard_entry)
                    logger.info(
                        "=> completed unit dataset_id='%s' model='%s' %s=%.4f.",
                        dataset_id, model_name, primary_metric,
                        leaderboard_entry.metrics.get(primary_metric, float("nan")),
                    )

            # Release the plasma entry now that all model futures are collected.
            del common_ref

            if not entries:
                logger.warning(
                    "=> dataset_id='%s': all models produced zero successful trials -- no leaderboard record written.",
                    dataset_id,
                )
                total_units_completed += len(pending_model_names)
                continue

            entries.sort(key=lambda entry: entry.metrics[primary_metric], reverse=is_maximize)
            entries = entries[:leaderboard_top_n]
            for rank, entry in enumerate(entries, start=1):
                entry.rank = rank

            record = LeaderboardRecord(
                dataset_id=dataset_id,
                encoder_version="n/a",
                embedding=None,
                task_type=task_type,
                n_rows=n_rows,
                n_cols=n_cols,
                target_cardinality=target_cardinality,
                primary_metric=primary_metric,
                leaderboard=entries,
                best_model=entries[0].model_name,
                created_at=utc_now_isoformat(),
            )
            self.store.write_leaderboard_record(record.model_dump(exclude={"embedding"}))
            logger.info(
                "=> wrote leaderboard record for dataset_id='%s' (%d entries, best=%s).",
                dataset_id, len(entries), entries[0].model_name,
            )
            total_units_completed += len(pending_model_names)

        self.store.build_meta_kb()
        return total_units_completed
