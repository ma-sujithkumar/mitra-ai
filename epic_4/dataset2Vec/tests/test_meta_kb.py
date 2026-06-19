import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd
import yaml

TOOL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_BIN = "/home/sujithma/venv/bin/python"
TOY_CORPUS_DIR = os.path.join(TOOL_ROOT, "claude_scripts", "toy_corpus")
MODEL_LIBRARY_ROOT = "/home/sujithma/mitra/model_library"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end test of the Dataset2Vec meta-knowledge base across "
        "all 3 phases, in an isolated temp store_dir, CPU-only, on the 5 toy datasets."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args()


def write_temp_config(temp_root: str) -> tuple[str, dict]:
    """Mirrors the temp-config-directory construction pattern used by every
    claude_scripts/smoke_test_*.py script: config.yaml and search_spaces.json
    live under a 'config/' subdirectory, referenced with that prefix inside
    the generated config.ini, matching d2v_core/schema.py's tool-root-relative
    path resolution."""
    config_dir = os.path.join(temp_root, "config")
    os.makedirs(config_dir, exist_ok=True)

    with open(os.path.join(TOOL_ROOT, "config", "config.yaml"), "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    full_config["training"]["corpus_dir"] = TOY_CORPUS_DIR
    full_config["training"]["device"] = "cpu"
    full_config["training"]["epochs"] = 15
    full_config["training"]["checkpoint_every"] = 5
    full_config["training"]["es_patience"] = 1000
    full_config["training"]["n_instances_sample"] = 64
    full_config["training"]["n_features_sample"] = 8
    full_config["training"]["pairs_per_batch"] = 10

    full_config["sweep"]["corpus_dir"] = TOY_CORPUS_DIR
    full_config["sweep"]["models"] = ["LogisticRegression", "RandomForestClassifier"]
    full_config["sweep"]["n_parallel"] = 4
    full_config["sweep"]["n_trials_per_model"] = 2

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
model_library_root = {MODEL_LIBRARY_ROOT}
config_yaml = config/config.yaml
search_spaces_json = config/search_spaces.json
store_dir = store
"""
    config_ini_path = os.path.join(config_dir, "config.ini")
    with open(config_ini_path, "w") as ini_file:
        ini_file.write(config_ini_content)
    return config_ini_path, full_config


def run_cli(*args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [PYTHON_BIN, *args], cwd=TOOL_ROOT, capture_output=True, text=True
    )
    logger.debug("=> ran %s -> exit=%d", args, result.returncode)
    if result.returncode != 0:
        logger.error("stdout:\n%s", result.stdout[-4000:])
        logger.error("stderr:\n%s", result.stderr[-4000:])
    return result


def test_cold_start(config_ini_path: str, output_dir: str) -> None:
    """Phase 3 against a freshly-created, completely empty store_dir must
    early-return cold_start=true rather than raising."""
    result = run_cli(
        os.path.join(TOOL_ROOT, "query.py"),
        "-c", config_ini_path,
        "-i", os.path.join(TOY_CORPUS_DIR, "iris.npz"),
        "-o", output_dir,
    )
    assert result.returncode == 0, "query.py exited non-zero on the cold-start store"

    with open(os.path.join(output_dir, "dataset_prior.json"), "r") as prior_file:
        prior_payload = json.load(prior_file)
    assert prior_payload["cold_start"] is True, prior_payload
    assert prior_payload["neighbors"] == [], prior_payload
    assert prior_payload["ranked_models"] == [], prior_payload
    logger.info("=> test_cold_start passed.")


def test_phase1_train_encoder(config_ini_path: str, store_dir: str) -> None:
    """PHASE 1: trains the encoder on the toy corpus and embeds it. Asserts
    encoder.pt/encoder_version.json/train_embeddings.parquet exist with the
    expected dataset_ids and a fixed-length embedding per dataset."""
    result = run_cli(os.path.join(TOOL_ROOT, "train_encoder.py"), "-c", config_ini_path, "-v")
    assert result.returncode == 0, "train_encoder.py exited non-zero"

    assert os.path.isfile(os.path.join(store_dir, "encoder", "encoder.pt"))
    assert os.path.isfile(os.path.join(store_dir, "encoder", "encoder_version.json"))

    embeddings_df = pd.read_parquet(os.path.join(store_dir, "train_embeddings.parquet"))
    assert set(embeddings_df["dataset_id"]) == {
        "iris", "wine", "breast_cancer", "diabetes", "synthetic_blob"
    }, embeddings_df["dataset_id"].tolist()

    with open(os.path.join(store_dir, "encoder", "encoder_version.json"), "r") as version_file:
        embedding_dim = json.load(version_file)["embedding_dim"]
    assert all(len(row) == embedding_dim for row in embeddings_df["embedding"])
    logger.info("=> test_phase1_train_encoder passed (%d datasets embedded).", len(embeddings_df))


def test_phase2_build_leaderboard_db(config_ini_path: str, store_dir: str) -> None:
    """PHASE 2: sweeps 2 classification datasets (iris, wine) x 2 sklearn
    models. Only classification datasets are swept here -- diabetes is
    regression and was never registered with classifier models, which proves
    build_meta_kb()'s inner join correctly excludes a dataset embedded in
    Phase 1 but never swept in Phase 2."""
    result = run_cli(
        os.path.join(TOOL_ROOT, "build_leaderboard_db.py"),
        "-c", config_ini_path, "--datasets", "iris,wine", "--resume", "-v",
    )
    assert result.returncode == 0, "build_leaderboard_db.py exited non-zero"

    leaderboards_df = pd.read_parquet(os.path.join(store_dir, "leaderboards.parquet"))
    assert set(leaderboards_df["dataset_id"]) == {"iris", "wine"}, leaderboards_df["dataset_id"].tolist()

    meta_kb_df = pd.read_parquet(os.path.join(store_dir, "meta_kb.parquet"))
    assert set(meta_kb_df["dataset_id"]) == {"iris", "wine"}, (
        "meta_kb.parquet should only contain datasets present in BOTH "
        f"train_embeddings.parquet and leaderboards.parquet, got {meta_kb_df['dataset_id'].tolist()}"
    )
    assert os.path.isfile(os.path.join(store_dir, "index.faiss"))

    completed_units_before_resume = set(leaderboards_df["dataset_id"])

    # --resume re-run: must skip every already-completed unit (no new trials).
    second_result = run_cli(
        os.path.join(TOOL_ROOT, "build_leaderboard_db.py"),
        "-c", config_ini_path, "--datasets", "iris,wine", "--resume", "-v",
    )
    assert second_result.returncode == 0, "build_leaderboard_db.py --resume re-run exited non-zero"
    assert "no remaining units" in second_result.stdout + second_result.stderr, (
        "expected the resumed re-run to report no remaining units"
    )

    leaderboards_df_after_resume = pd.read_parquet(os.path.join(store_dir, "leaderboards.parquet"))
    assert set(leaderboards_df_after_resume["dataset_id"]) == completed_units_before_resume
    logger.info("=> test_phase2_build_leaderboard_db passed (2 datasets x 2 models, --resume verified).")


def test_phase3_retrieve(config_ini_path: str, output_dir: str) -> None:
    """PHASE 3 retrieve-only: query with breast_cancer.npz -- a dataset that
    Phase 1 embedded but Phase 2 never swept -- so its only possible neighbors
    are iris/wine (proving cross-dataset retrieval, not a degenerate
    self-match)."""
    result = run_cli(
        os.path.join(TOOL_ROOT, "query.py"),
        "-c", config_ini_path,
        "-i", os.path.join(TOY_CORPUS_DIR, "breast_cancer.npz"),
        "-o", output_dir, "-v",
    )
    assert result.returncode == 0, "query.py exited non-zero on the warm retrieve path"

    with open(os.path.join(output_dir, "dataset_prior.json"), "r") as prior_file:
        prior_payload = json.load(prior_file)

    assert prior_payload["cold_start"] is False, prior_payload
    neighbor_ids = {neighbor["dataset_id"] for neighbor in prior_payload["neighbors"]}
    assert neighbor_ids.issubset({"iris", "wine"}), neighbor_ids
    assert len(prior_payload["ranked_models"]) > 0, prior_payload

    similarities = [neighbor["similarity"] for neighbor in prior_payload["neighbors"]]
    assert similarities == sorted(similarities, reverse=True), similarities
    scores = [entry["score"] for entry in prior_payload["ranked_models"]]
    assert scores == sorted(scores, reverse=True), scores

    sys.path.insert(0, MODEL_LIBRARY_ROOT)
    from core.validators import validate_model_name

    for ranked_entry in prior_payload["ranked_models"]:
        validate_model_name(ranked_entry["model_name"], training_mode="fine_tune")
    logger.info(
        "=> test_phase3_retrieve passed (%d neighbors, %d ranked models).",
        len(prior_payload["neighbors"]), len(prior_payload["ranked_models"]),
    )


def test_phase3_verify(config_ini_path: str, output_dir: str) -> None:
    """PHASE 3 with --verify: ACTUALLY TRAINS every recommended model on
    breast_cancer.npz and checks achieved metrics against expected_metric."""
    result = run_cli(
        os.path.join(TOOL_ROOT, "query.py"),
        "-c", config_ini_path,
        "-i", os.path.join(TOY_CORPUS_DIR, "breast_cancer.npz"),
        "-o", output_dir, "--verify", "-v",
    )
    assert result.returncode == 0, "query.py --verify exited non-zero"

    with open(os.path.join(output_dir, "dataset_prior.json"), "r") as prior_file:
        prior_payload = json.load(prior_file)

    verification_summary = prior_payload["verification_summary"]
    assert verification_summary is not None, prior_payload
    assert verification_summary["n_verified"] == len(prior_payload["ranked_models"])
    assert verification_summary["best_achieved"] is not None, verification_summary

    for ranked_entry in prior_payload["ranked_models"]:
        verification = ranked_entry["verification"]
        assert verification["trained"] is True, verification
        assert verification["achieved_metric"] is not None, verification
        assert verification["within_tolerance"] in (True, False), verification
    logger.info(
        "=> test_phase3_verify passed (%d/%d models within tolerance, mean_abs_delta=%.4f).",
        verification_summary["n_within_tolerance"], verification_summary["n_verified"],
        verification_summary["mean_abs_delta"],
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    started_at = time.time()
    temp_root = tempfile.mkdtemp(prefix="test_meta_kb_e2e_")
    config_ini_path, _full_config = write_temp_config(temp_root)
    store_dir = os.path.join(temp_root, "store")

    test_cold_start(config_ini_path, os.path.join(temp_root, "output_cold_start"))
    test_phase1_train_encoder(config_ini_path, store_dir)
    test_phase2_build_leaderboard_db(config_ini_path, store_dir)
    test_phase3_retrieve(config_ini_path, os.path.join(temp_root, "output_retrieve"))
    test_phase3_verify(config_ini_path, os.path.join(temp_root, "output_verify"))

    shutil.rmtree(temp_root)
    elapsed_seconds = time.time() - started_at
    print(f"=> ALL TESTS PASSED in {elapsed_seconds:.1f}s (isolated temp store_dir, CPU-only).")


if __name__ == "__main__":
    main()
