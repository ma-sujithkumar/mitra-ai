import logging
import os
import shutil
import sys
import tempfile

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from d2v_core.schema import LeaderboardEntry, load_search_spaces
from d2v_core.sweep import run_optuna_study

sys.path.insert(0, "/home/sujithma/mitra/model_library")
from core.data_bundle import CommonData

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    common = CommonData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    tool_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    search_spaces = load_search_spaces(os.path.join(tool_root, "config", "config.ini"))

    temp_dir = tempfile.mkdtemp(prefix="optuna_smoke_")
    optuna_storage = f"sqlite:///{os.path.join(temp_dir, 'optuna.db')}"

    leaderboard_entry = run_optuna_study(
        dataset_id="iris",
        model_name="RandomForestClassifier",
        common=common,
        task_type="classification",
        search_spaces=search_spaces,
        optuna_storage=optuna_storage,
        n_trials=3,
        primary_metric="f1_macro",
        optuna_sampler="tpe",
        optuna_pruner="median",
    )
    assert isinstance(leaderboard_entry, LeaderboardEntry)
    assert leaderboard_entry.n_trials == 3, leaderboard_entry.n_trials
    assert leaderboard_entry.metrics["f1_macro"] > 0.7, leaderboard_entry.metrics
    print(f"=> leaderboard entry after 3 trials: {leaderboard_entry}")

    # re-run with n_trials=5 -- should top up by 2 more, not restart from 0.
    topped_up_entry = run_optuna_study(
        dataset_id="iris",
        model_name="RandomForestClassifier",
        common=common,
        task_type="classification",
        search_spaces=search_spaces,
        optuna_storage=optuna_storage,
        n_trials=5,
        primary_metric="f1_macro",
        optuna_sampler="tpe",
        optuna_pruner="median",
    )
    assert topped_up_entry.n_trials == 5, topped_up_entry.n_trials

    shutil.rmtree(temp_dir)
    print("=> smoke test passed: Optuna study wrapping + resume top-up work correctly.")


if __name__ == "__main__":
    main()
