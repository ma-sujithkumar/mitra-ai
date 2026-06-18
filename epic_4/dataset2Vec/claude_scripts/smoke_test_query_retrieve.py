import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from d2v_core.encoder import Dataset2VecEncoder
from d2v_core.store import MetaKnowledgeStore

TOOL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_BIN = "/home/sujithma/venv/bin/python"
TOY_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "toy_corpus")

_boot_sys_path_entry = os.path.join(TOOL_ROOT, "..", "..", "model_library")
sys.path.insert(0, os.path.normpath(_boot_sys_path_entry))
from core.validators import validate_model_name  # noqa: E402  (path bootstrap above)


def write_temp_config(temp_root: str) -> str:
    """Mirrors the temp-config-directory construction pattern from
    smoke_test_phase1_e2e.py: config.yaml and search_spaces.json get a
    'config/' prefix inside config.ini, resolved relative to the tool root
    (parent of the ini's own directory)."""
    config_dir = os.path.join(temp_root, "config")
    os.makedirs(config_dir, exist_ok=True)

    with open(os.path.join(TOOL_ROOT, "config", "config.yaml"), "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    full_config["training"]["corpus_dir"] = TOY_CORPUS_DIR
    full_config["training"]["device"] = "cpu"
    full_config["training"]["n_instances_sample"] = 64
    full_config["training"]["n_features_sample"] = 8

    with open(os.path.join(config_dir, "config.yaml"), "w") as yaml_file:
        yaml.safe_dump(full_config, yaml_file)
    shutil.copy(
        os.path.join(TOOL_ROOT, "config", "search_spaces.json"),
        os.path.join(config_dir, "search_spaces.json"),
    )

    config_ini_content = f"""[python]
PYTHON = {PYTHON_BIN}

[paths]
model_library_root = /home/sujithma/mitra/model_library
config_yaml = config/config.yaml
search_spaces_json = config/search_spaces.json
store_dir = store
"""
    config_ini_path = os.path.join(config_dir, "config.ini")
    with open(config_ini_path, "w") as ini_file:
        ini_file.write(config_ini_content)
    return config_ini_path, full_config


def run_query(config_ini_path: str, input_npz_path: str, output_dir: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            PYTHON_BIN, os.path.join(TOOL_ROOT, "query.py"),
            "-c", config_ini_path,
            "-i", input_npz_path,
            "-o", output_dir,
            "-v",
        ],
        cwd=TOOL_ROOT,
        capture_output=True,
        text=True,
    )


def scenario_a_cold_start() -> None:
    """Empty temp store_dir (no parquet files at all) -- query.py must early-
    return a cold_start=true prior rather than raising."""
    temp_root = tempfile.mkdtemp(prefix="query_smoke_cold_")
    config_ini_path, _full_config = write_temp_config(temp_root)
    output_dir = os.path.join(temp_root, "output")

    result = run_query(config_ini_path, os.path.join(TOY_CORPUS_DIR, "iris.npz"), output_dir)
    print(result.stdout[-3000:])
    print(result.stderr[-3000:])
    assert result.returncode == 0, "query.py exited non-zero on cold-start scenario"

    prior_path = os.path.join(output_dir, "dataset_prior.json")
    assert os.path.isfile(prior_path), f"dataset_prior.json not written at '{prior_path}'"
    with open(prior_path, "r") as prior_file:
        prior_payload = json.load(prior_file)

    assert prior_payload["cold_start"] is True, prior_payload
    assert prior_payload["neighbors"] == [], prior_payload["neighbors"]
    assert prior_payload["ranked_models"] == [], prior_payload["ranked_models"]
    assert len(prior_payload["caveats"]) >= 1, prior_payload["caveats"]

    shutil.rmtree(temp_root)
    print("=> smoke test passed: scenario A (cold start, empty store) -> cold_start=true, empty neighbors/ranked_models.")


def build_fake_warm_store(store_dir: str, embedding_dim: int) -> None:
    """Hand-builds a small fake meta_kb mirroring smoke_test_store.py's pattern:
    a few rows of train_embeddings.parquet + leaderboards.parquet, joined via
    build_meta_kb() so meta_kb.parquet + index.faiss exist. Embedding values are
    random but dimensionally consistent with the real (untrained) encoder."""
    store = MetaKnowledgeStore(store_dir=store_dir, faiss_metric="ip", normalize_embeddings=True)

    fake_embedding_rows = [
        {
            "dataset_id": "iris", "encoder_version": "d2v-v1",
            "embedding": list(np.random.RandomState(1).rand(embedding_dim)),
            "n_rows": 150, "n_cols": 4, "task_type": "classification",
            "target_cardinality": 3, "created_at": "2026-06-17T00:00:00+00:00",
        },
        {
            "dataset_id": "wine", "encoder_version": "d2v-v1",
            "embedding": list(np.random.RandomState(2).rand(embedding_dim)),
            "n_rows": 178, "n_cols": 13, "task_type": "classification",
            "target_cardinality": 3, "created_at": "2026-06-17T00:00:00+00:00",
        },
        {
            "dataset_id": "breast_cancer", "encoder_version": "d2v-v1",
            "embedding": list(np.random.RandomState(3).rand(embedding_dim)),
            "n_rows": 569, "n_cols": 30, "task_type": "classification",
            "target_cardinality": 2, "created_at": "2026-06-17T00:00:00+00:00",
        },
        {
            "dataset_id": "diabetes", "encoder_version": "d2v-v1",
            "embedding": list(np.random.RandomState(4).rand(embedding_dim)),
            "n_rows": 442, "n_cols": 10, "task_type": "regression",
            "target_cardinality": 0, "created_at": "2026-06-17T00:00:00+00:00",
        },
    ]
    store.write_embeddings(fake_embedding_rows)

    fake_leaderboard_records = [
        {
            "dataset_id": "iris", "encoder_version": "d2v-v1", "task_type": "classification",
            "n_rows": 150, "n_cols": 4, "target_cardinality": 3, "primary_metric": "f1_macro",
            "leaderboard": [
                {"rank": 1, "model_name": "RandomForestClassifier",
                 "hyperparameters": {"n_estimators": 200}, "metrics": {"f1_macro": 0.97}, "n_trials": 20},
                {"rank": 2, "model_name": "LogisticRegression",
                 "hyperparameters": {"C": 1.0}, "metrics": {"f1_macro": 0.93}, "n_trials": 20},
            ],
            "best_model": "RandomForestClassifier", "created_at": "2026-06-17T00:00:00+00:00",
        },
        {
            "dataset_id": "wine", "encoder_version": "d2v-v1", "task_type": "classification",
            "n_rows": 178, "n_cols": 13, "target_cardinality": 3, "primary_metric": "f1_macro",
            "leaderboard": [
                {"rank": 1, "model_name": "LogisticRegression",
                 "hyperparameters": {"C": 1.5}, "metrics": {"f1_macro": 0.95}, "n_trials": 15},
                {"rank": 2, "model_name": "RandomForestClassifier",
                 "hyperparameters": {"n_estimators": 100}, "metrics": {"f1_macro": 0.90}, "n_trials": 15},
            ],
            "best_model": "LogisticRegression", "created_at": "2026-06-17T00:00:00+00:00",
        },
        {
            "dataset_id": "breast_cancer", "encoder_version": "d2v-v1", "task_type": "classification",
            "n_rows": 569, "n_cols": 30, "target_cardinality": 2, "primary_metric": "f1_macro",
            "leaderboard": [
                {"rank": 1, "model_name": "XGBClassifier",
                 "hyperparameters": {"max_depth": 4}, "metrics": {"f1_macro": 0.96}, "n_trials": 25},
            ],
            "best_model": "XGBClassifier", "created_at": "2026-06-17T00:00:00+00:00",
        },
        {
            "dataset_id": "diabetes", "encoder_version": "d2v-v1", "task_type": "regression",
            "n_rows": 442, "n_cols": 10, "target_cardinality": 0, "primary_metric": "rmse",
            "leaderboard": [
                {"rank": 1, "model_name": "XGBRegressor",
                 "hyperparameters": {"max_depth": 4}, "metrics": {"rmse": 52.3}, "n_trials": 25},
            ],
            "best_model": "XGBRegressor", "created_at": "2026-06-17T00:00:00+00:00",
        },
    ]
    for record in fake_leaderboard_records:
        store.write_leaderboard_record(record)

    joined_row_count = store.build_meta_kb()
    assert joined_row_count == 4, f"expected 4 joined rows, got {joined_row_count}"


def scenario_b_warm_path() -> None:
    """Hand-built fake meta_kb (random but dimensionally-consistent embeddings)
    + a real (untrained, random-init) encoder.pt saved with the real config's
    architecture -- enough to prove query.py's embed+search+rank plumbing works
    end-to-end without needing a full training run."""
    temp_root = tempfile.mkdtemp(prefix="query_smoke_warm_")
    config_ini_path, full_config = write_temp_config(temp_root)
    store_dir = os.path.join(temp_root, "store")
    os.makedirs(store_dir, exist_ok=True)

    embedding_dim = full_config["encoder"]["embedding_dim"]
    n_classes_sample = full_config["training"]["n_classes_sample"]

    encoder = Dataset2VecEncoder(full_config["encoder"], n_classes_sample=n_classes_sample)
    encoder_dir = os.path.join(store_dir, "encoder")
    os.makedirs(encoder_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(encoder_dir, "encoder.pt"))
    with open(os.path.join(encoder_dir, "encoder_version.json"), "w") as version_file:
        json.dump(
            {
                "encoder_version": full_config["encoder"]["encoder_version"],
                "embedding_dim": embedding_dim,
                "final_epoch": 0,
                "best_loss": 0.0,
                "created_at": "2026-06-17T00:00:00+00:00",
            },
            version_file,
            indent=2,
        )

    build_fake_warm_store(store_dir, embedding_dim)

    output_dir = os.path.join(temp_root, "output")
    result = run_query(config_ini_path, os.path.join(TOY_CORPUS_DIR, "iris.npz"), output_dir)
    print(result.stdout[-4000:])
    print(result.stderr[-4000:])
    assert result.returncode == 0, "query.py exited non-zero on warm-path scenario"

    prior_path = os.path.join(output_dir, "dataset_prior.json")
    assert os.path.isfile(prior_path), f"dataset_prior.json not written at '{prior_path}'"
    with open(prior_path, "r") as prior_file:
        prior_payload = json.load(prior_file)

    assert prior_payload["cold_start"] is False, prior_payload
    neighbors = prior_payload["neighbors"]
    ranked_models = prior_payload["ranked_models"]
    assert len(neighbors) > 0, "expected non-empty neighbors in warm path"
    assert len(ranked_models) > 0, "expected non-empty ranked_models in warm path"

    neighbor_similarities = [neighbor["similarity"] for neighbor in neighbors]
    assert neighbor_similarities == sorted(neighbor_similarities, reverse=True), neighbor_similarities

    ranked_scores = [entry["score"] for entry in ranked_models]
    assert ranked_scores == sorted(ranked_scores, reverse=True), ranked_scores

    for ranked_entry in ranked_models:
        validate_model_name(ranked_entry["model_name"], training_mode="fine_tune")

    shutil.rmtree(temp_root)
    print(
        "=> smoke test passed: scenario B (warm path) -> cold_start=false, "
        f"{len(neighbors)} neighbors (sorted desc), {len(ranked_models)} ranked_models "
        "(sorted desc, all model_names valid)."
    )


def main() -> None:
    scenario_a_cold_start()
    scenario_b_warm_path()


if __name__ == "__main__":
    main()
