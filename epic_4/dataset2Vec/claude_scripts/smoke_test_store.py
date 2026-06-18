import logging
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from d2v_core.store import MetaKnowledgeStore

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    temp_store_dir = tempfile.mkdtemp(prefix="meta_kb_smoke_")
    store = MetaKnowledgeStore(
        store_dir=temp_store_dir, faiss_metric="ip", normalize_embeddings=True
    )

    fake_embedding_rows = [
        {
            "dataset_id": "iris",
            "encoder_version": "v0",
            "embedding": list(np.random.RandomState(1).rand(8)),
            "n_rows": 150,
            "n_cols": 4,
            "task_type": "classification",
            "target_cardinality": 3,
            "created_at": "2026-06-16T00:00:00+00:00",
        },
        {
            "dataset_id": "wine",
            "encoder_version": "v0",
            "embedding": list(np.random.RandomState(2).rand(8)),
            "n_rows": 178,
            "n_cols": 13,
            "task_type": "classification",
            "target_cardinality": 3,
            "created_at": "2026-06-16T00:00:00+00:00",
        },
        {
            "dataset_id": "diabetes",
            "encoder_version": "v0",
            "embedding": list(np.random.RandomState(3).rand(8)),
            "n_rows": 442,
            "n_cols": 10,
            "task_type": "regression",
            "target_cardinality": 0,
            "created_at": "2026-06-16T00:00:00+00:00",
        },
    ]
    store.write_embeddings(fake_embedding_rows)

    fake_leaderboard_records = [
        {
            "dataset_id": "iris",
            "encoder_version": "v0",
            "task_type": "classification",
            "n_rows": 150,
            "n_cols": 4,
            "target_cardinality": 3,
            "primary_metric": "accuracy",
            "leaderboard": [
                {
                    "rank": 1,
                    "model_name": "RandomForestClassifier",
                    "hyperparameters": {"n_estimators": 200},
                    "metrics": {"accuracy": 0.97},
                    "n_trials": 20,
                }
            ],
            "best_model": "RandomForestClassifier",
            "created_at": "2026-06-16T00:00:00+00:00",
        },
        {
            "dataset_id": "wine",
            "encoder_version": "v0",
            "task_type": "classification",
            "n_rows": 178,
            "n_cols": 13,
            "target_cardinality": 3,
            "primary_metric": "accuracy",
            "leaderboard": [
                {
                    "rank": 1,
                    "model_name": "LogisticRegression",
                    "hyperparameters": {"C": 1.5},
                    "metrics": {"accuracy": 0.95},
                    "n_trials": 15,
                }
            ],
            "best_model": "LogisticRegression",
            "created_at": "2026-06-16T00:00:00+00:00",
        },
        {
            "dataset_id": "diabetes",
            "encoder_version": "v0",
            "task_type": "regression",
            "n_rows": 442,
            "n_cols": 10,
            "target_cardinality": 0,
            "primary_metric": "rmse",
            "leaderboard": [
                {
                    "rank": 1,
                    "model_name": "XGBRegressor",
                    "hyperparameters": {"max_depth": 4},
                    "metrics": {"rmse": 52.3},
                    "n_trials": 25,
                }
            ],
            "best_model": "XGBRegressor",
            "created_at": "2026-06-16T00:00:00+00:00",
        },
    ]
    for record in fake_leaderboard_records:
        store.write_leaderboard_record(record)

    joined_row_count = store.build_meta_kb()
    assert joined_row_count == 3, f"expected 3 joined rows, got {joined_row_count}"
    assert not store.is_empty()

    query_vector = np.random.RandomState(1).rand(8)
    same_task_results = store.search(
        query_vector, top_k=5, task_type="classification", same_task_only=True
    )
    assert len(same_task_results) == 2, same_task_results
    assert same_task_results[0][0] == "iris", same_task_results

    all_task_results = store.search(
        query_vector, top_k=5, task_type="classification", same_task_only=False
    )
    assert len(all_task_results) == 3, all_task_results

    completed = store.completed_units()
    assert completed == set(), completed

    shutil.rmtree(temp_store_dir)
    print("=> smoke test passed: store.py join/search/completed_units all correct.")


if __name__ == "__main__":
    main()
