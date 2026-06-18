import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd
import yaml

TOOL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_BIN = "/home/sujithma/venv/bin/python"


def main() -> None:
    temp_root = tempfile.mkdtemp(prefix="phase1_e2e_smoke_")
    config_dir = os.path.join(temp_root, "config")
    os.makedirs(config_dir, exist_ok=True)

    with open(os.path.join(TOOL_ROOT, "config", "config.yaml"), "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    full_config["training"]["corpus_dir"] = os.path.join(os.path.dirname(__file__), "toy_corpus")
    full_config["training"]["device"] = "cpu"
    full_config["training"]["epochs"] = 15
    full_config["training"]["checkpoint_every"] = 5
    full_config["training"]["es_patience"] = 1000
    full_config["training"]["n_instances_sample"] = 64
    full_config["training"]["n_features_sample"] = 8
    full_config["training"]["pairs_per_batch"] = 10

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

    result = subprocess.run(
        [PYTHON_BIN, os.path.join(TOOL_ROOT, "train_encoder.py"), "-c", config_ini_path, "-v"],
        cwd=TOOL_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-3000:])
    print(result.stderr[-3000:])
    assert result.returncode == 0, "train_encoder.py exited non-zero"

    store_dir = os.path.join(temp_root, "store")
    assert os.path.isfile(os.path.join(store_dir, "encoder", "encoder.pt"))
    assert os.path.isfile(os.path.join(store_dir, "encoder", "encoder_version.json"))
    embeddings_path = os.path.join(store_dir, "train_embeddings.parquet")
    assert os.path.isfile(embeddings_path)

    embeddings_df = pd.read_parquet(embeddings_path)
    assert set(embeddings_df["dataset_id"]) == {
        "iris", "wine", "breast_cancer", "diabetes", "synthetic_blob"
    }, embeddings_df["dataset_id"].tolist()
    assert all(len(row) == full_config["encoder"]["embedding_dim"] for row in embeddings_df["embedding"])

    shutil.rmtree(temp_root)
    print("=> smoke test passed: train_encoder.py end-to-end produced encoder.pt + embeddings.")


if __name__ == "__main__":
    main()
