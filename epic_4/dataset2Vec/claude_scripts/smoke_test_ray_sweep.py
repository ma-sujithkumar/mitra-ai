import logging
import os
import shutil
import sys
import tempfile
import time

import pandas as pd
import ray

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from d2v_core.schema import load_search_spaces
from d2v_core.store import MetaKnowledgeStore
from d2v_core.sweep import LeaderboardSweep, MemoryJanitor

logging.basicConfig(level=logging.INFO, format="%(message)s")

TOOL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "toy_corpus")


@ray.remote(num_gpus=1.0)
def _gpu_marker_task(task_id: int, sleep_seconds: float) -> tuple[int, float, float]:
    start_time = time.time()
    time.sleep(sleep_seconds)
    end_time = time.time()
    return task_id, start_time, end_time


def check_gpu_tasks_serialize() -> None:
    """Confirms the mechanism the design relies on: two tasks each requesting
    num_gpus=1.0 on a 1-GPU Ray cluster never run concurrently."""
    futures = [_gpu_marker_task.remote(0, 1.0), _gpu_marker_task.remote(1, 1.0)]
    results = sorted(ray.get(futures), key=lambda r: r[1])
    (_, start_0, end_0), (_, start_1, end_1) = results
    assert start_1 >= end_0, (
        f"GPU tasks overlapped: task0=({start_0},{end_0}) task1=({start_1},{end_1})"
    )
    print("=> confirmed: num_gpus=1.0 tasks are serialized by Ray's scheduler.")


def main() -> None:
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=1)

    check_gpu_tasks_serialize()

    temp_store_dir = tempfile.mkdtemp(prefix="ray_sweep_smoke_")
    scratch_dir = os.path.join(temp_store_dir, "scratch")
    janitor = MemoryJanitor(scratch_dir=scratch_dir, cleanup_interval_seconds=3600)
    janitor.start()
    assert janitor._thread is not None and janitor._thread.is_alive()

    search_spaces = load_search_spaces(os.path.join(TOOL_ROOT, "config", "config.ini"))
    store = MetaKnowledgeStore(store_dir=temp_store_dir, faiss_metric="ip", normalize_embeddings=True)

    sweep_config = {
        "n_parallel": 4,
        "n_trials_per_model": 2,
        "primary_metric": "f1_macro",
        "leaderboard_top_n": 10,
        "optuna_sampler": "tpe",
        "optuna_pruner": "median",
        "optuna_storage": f"sqlite:///{os.path.join(temp_store_dir, 'optuna.db')}",
    }
    sweep = LeaderboardSweep(store=store, search_spaces=search_spaces, sweep_config=sweep_config)

    dataset_ids = ["iris", "wine"]
    model_names = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"]
    n_units_run = sweep.run(CORPUS_DIR, dataset_ids, model_names)
    assert n_units_run == 6, n_units_run

    leaderboards_path = os.path.join(temp_store_dir, "leaderboards.parquet")
    leaderboards_df = pd.read_parquet(leaderboards_path)
    assert set(leaderboards_df["dataset_id"]) == {"iris", "wine"}
    for _, row in leaderboards_df.iterrows():
        assert len(row["leaderboard"]) == 3, row["leaderboard"]

    completed_units = store.completed_units()
    assert completed_units == {
        ("iris", "LogisticRegression"), ("iris", "RandomForestClassifier"), ("iris", "XGBClassifier"),
        ("wine", "LogisticRegression"), ("wine", "RandomForestClassifier"), ("wine", "XGBClassifier"),
    }, completed_units

    # re-run with the same units -- should skip all of them (already completed).
    n_units_second_run = sweep.run(CORPUS_DIR, dataset_ids, model_names)
    assert n_units_second_run == 0, n_units_second_run

    janitor.stop()
    assert not janitor._thread.is_alive()

    shutil.rmtree(temp_store_dir)
    print("=> smoke test passed: Ray parallel dispatch (6 units) + MemoryJanitor start/stop OK.")


if __name__ == "__main__":
    main()
