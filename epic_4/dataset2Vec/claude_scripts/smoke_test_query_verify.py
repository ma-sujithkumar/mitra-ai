import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from d2v_core.encoder import Dataset2VecEncoder
from d2v_core.store import MetaKnowledgeStore

TOOL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_BIN = "/home/sujithma/venv/bin/python"
TOY_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "toy_corpus")


def write_temp_config(temp_root: str) -> tuple[str, dict]:
    config_dir = os.path.join(temp_root, "config")
    os.makedirs(config_dir, exist_ok=True)

    with open(os.path.join(TOOL_ROOT, "config", "config.yaml"), "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    full_config["training"]["corpus_dir"] = TOY_CORPUS_DIR
    full_config["training"]["device"] = "cpu"
    full_config["training"]["n_instances_sample"] = 64
    full_config["training"]["n_features_sample"] = 8
    full_config["retrieval"]["verify_tolerance"] = 0.5

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


def build_fake_warm_store(store_dir: str, embedding_dim: int) -> None:
    """A single fake neighbor ('wine') whose top-2 leaderboard entries are
    real, cheaply-trainable sklearn models with realistic hyperparameters --
    --verify will actually train these on the query dataset (iris)."""
    import numpy as np

    store = MetaKnowledgeStore(store_dir=store_dir, faiss_metric="ip", normalize_embeddings=True)
    store.write_embeddings(
        [
            {
                "dataset_id": "wine", "encoder_version": "d2v-v1",
                "embedding": list(np.random.RandomState(2).rand(embedding_dim)),
                "n_rows": 178, "n_cols": 13, "task_type": "classification",
                "target_cardinality": 3, "created_at": "2026-06-17T00:00:00+00:00",
            },
        ]
    )
    store.write_leaderboard_record(
        {
            "dataset_id": "wine", "encoder_version": "d2v-v1", "task_type": "classification",
            "n_rows": 178, "n_cols": 13, "target_cardinality": 3, "primary_metric": "f1_macro",
            "leaderboard": [
                {"rank": 1, "model_name": "LogisticRegression",
                 "hyperparameters": {"C": 1.0, "max_iter": 500}, "metrics": {"f1_macro": 0.95}, "n_trials": 15},
                {"rank": 2, "model_name": "RandomForestClassifier",
                 "hyperparameters": {"n_estimators": 50, "max_depth": 5}, "metrics": {"f1_macro": 0.90}, "n_trials": 15},
            ],
            "best_model": "LogisticRegression", "created_at": "2026-06-17T00:00:00+00:00",
        }
    )
    joined_row_count = store.build_meta_kb()
    assert joined_row_count == 1, f"expected 1 joined row, got {joined_row_count}"


def run_query(config_ini_path: str, input_npz_path: str, output_dir: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            PYTHON_BIN, os.path.join(TOOL_ROOT, "query.py"),
            "-c", config_ini_path,
            "-i", input_npz_path,
            "-o", output_dir,
            "--verify",
            "-v",
        ],
        cwd=TOOL_ROOT,
        capture_output=True,
        text=True,
    )


def main() -> None:
    temp_root = tempfile.mkdtemp(prefix="query_smoke_verify_")
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
    print(result.stdout[-5000:])
    print(result.stderr[-5000:])
    assert result.returncode == 0, "query.py --verify exited non-zero"

    prior_path = os.path.join(output_dir, "dataset_prior.json")
    with open(prior_path, "r") as prior_file:
        prior_payload = json.load(prior_file)

    assert prior_payload["cold_start"] is False, prior_payload
    ranked_models = prior_payload["ranked_models"]
    assert len(ranked_models) >= 2, ranked_models

    verification_summary = prior_payload["verification_summary"]
    assert verification_summary is not None, prior_payload
    assert verification_summary["n_verified"] == len(ranked_models), verification_summary
    assert verification_summary["best_achieved"] is not None, verification_summary
    assert verification_summary["mean_abs_delta"] is not None, verification_summary

    for ranked_entry in ranked_models:
        verification = ranked_entry["verification"]
        assert verification is not None, ranked_entry
        assert verification["trained"] is True, verification
        assert verification["achieved_metric"] is not None, verification
        assert verification["delta_vs_expected"] is not None, verification
        assert verification["within_tolerance"] in (True, False), verification

    shutil.rmtree(temp_root)
    print(
        "=> smoke test passed: --verify actually trained "
        f"{verification_summary['n_verified']} model(s), "
        f"{verification_summary['n_within_tolerance']} within tolerance="
        f"{verification_summary['tolerance']}, mean_abs_delta="
        f"{verification_summary['mean_abs_delta']:.4f}."
    )


if __name__ == "__main__":
    main()
