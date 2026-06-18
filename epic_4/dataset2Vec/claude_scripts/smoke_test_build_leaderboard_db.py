import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd
import yaml

TOOL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_BIN = "/home/sujithma/venv/bin/python"


def build_temp_config(temp_root: str) -> str:
    """Mirrors smoke_test_phase1_e2e.py's temp-config-construction pattern:
    config.yaml and search_spaces.json live under temp_root/config and are
    referenced with a 'config/' prefix in the generated ini, matching the
    tool_root-relative resolution logic in d2v_core/schema.py."""
    config_dir = os.path.join(temp_root, "config")
    os.makedirs(config_dir, exist_ok=True)

    with open(os.path.join(TOOL_ROOT, "config", "config.yaml"), "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    full_config["sweep"]["corpus_dir"] = os.path.join(os.path.dirname(__file__), "toy_corpus")
    full_config["sweep"]["n_parallel"] = 4
    full_config["sweep"]["n_trials_per_model"] = 2
    full_config["sweep"]["models"] = ["LogisticRegression", "RandomForestClassifier"]
    # optuna_storage / scratch_dir left as relative defaults ("sqlite:///store/optuna.db",
    # "store/scratch") to exercise build_leaderboard_db.py's path-resolution code.

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
    return config_ini_path


def run_cli(config_ini_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            PYTHON_BIN,
            os.path.join(TOOL_ROOT, "build_leaderboard_db.py"),
            "-c", config_ini_path,
            "--datasets", "iris,wine",
            "--resume",
            "-v",
        ],
        cwd=TOOL_ROOT,
        capture_output=True,
        text=True,
    )


def main() -> None:
    temp_root = tempfile.mkdtemp(prefix="build_leaderboard_db_smoke_")
    config_ini_path = build_temp_config(temp_root)
    store_dir = os.path.join(temp_root, "store")
    leaderboards_path = os.path.join(store_dir, "leaderboards.parquet")

    first_result = run_cli(config_ini_path)
    print(first_result.stdout[-4000:])
    print(first_result.stderr[-4000:])
    assert first_result.returncode == 0, "=> first build_leaderboard_db.py run exited non-zero"

    assert os.path.isfile(leaderboards_path), f"=> {leaderboards_path} was not created"
    leaderboards_df = pd.read_parquet(leaderboards_path)
    assert len(leaderboards_df) == 2, f"=> expected 2 rows, got {len(leaderboards_df)}"
    assert set(leaderboards_df["dataset_id"]) == {"iris", "wine"}, leaderboards_df["dataset_id"].tolist()
    for _, row in leaderboards_df.iterrows():
        assert len(row["leaderboard"]) > 0, f"=> empty leaderboard for dataset_id='{row['dataset_id']}'"

    n_trials_first_run = {
        row["dataset_id"]: [entry["n_trials"] for entry in row["leaderboard"]]
        for _, row in leaderboards_df.iterrows()
    }

    second_result = run_cli(config_ini_path)
    print(second_result.stdout[-4000:])
    print(second_result.stderr[-4000:])
    assert second_result.returncode == 0, "=> second build_leaderboard_db.py run exited non-zero"

    combined_second_output = second_result.stdout + second_result.stderr
    assert "no remaining units" in combined_second_output, (
        "=> expected LeaderboardSweep.run's 'no remaining units' log line on the resumed run"
    )

    leaderboards_df_after_resume = pd.read_parquet(leaderboards_path)
    assert len(leaderboards_df_after_resume) == 2, len(leaderboards_df_after_resume)
    n_trials_second_run = {
        row["dataset_id"]: [entry["n_trials"] for entry in row["leaderboard"]]
        for _, row in leaderboards_df_after_resume.iterrows()
    }
    assert n_trials_first_run == n_trials_second_run, (
        f"=> trial counts changed across resumed run: {n_trials_first_run} vs {n_trials_second_run}"
    )

    shutil.rmtree(temp_root)
    print(
        "=> smoke test passed: build_leaderboard_db.py CLI produced a 2-row leaderboards.parquet "
        "and the resumed re-run skipped all already-completed units (no new optuna trials)."
    )


if __name__ == "__main__":
    main()
