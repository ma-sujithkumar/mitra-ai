from pathlib import Path

import pandas as pd
import pytest
import ray

from backend.agents.feature_engineering.config import load_config
from backend.agents.feature_engineering.orchestrator import (
    DEFAULT_CONFIG_PATH,
    FeatureEngineerOrchestrator,
)
from backend.agents.feature_engineering.state import PipelineState


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster():
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True, log_to_driver=False)
    yield


def _make_state(tmp_path: Path) -> PipelineState:
    config = load_config(DEFAULT_CONFIG_PATH)
    frame = pd.DataFrame(
        {
            "feature_a": [float(i) for i in range(1, 21)],
            "feature_b": [float(i * 2) for i in range(1, 21)],
            "category_c": (["x", "y"] * 10),
        }
    )
    output_dir = tmp_path / "fe"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    state = PipelineState(
        df=frame,
        target=None,
        task="unsupervised",
        target_column=None,
        run_id="test_run",
        config=config,
        output_dir=output_dir,
    )
    state.stats_dir = stats_dir
    return state


def test_unsupervised_pipeline_runs_without_target(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    orchestrator = FeatureEngineerOrchestrator(
        data_path=tmp_path / "unused.csv",
        target_column=None,
        task="unsupervised",
        model_string="openai/test-model",
        llm_settings=None,
    )

    # select_features (the only target-dependent LLM step) is skipped for
    # unsupervised, so a no-op model_call is sufficient.
    orchestrator._run_pipeline(state, model_call=lambda prompt: "ok")

    # All engineered columns are kept as the selected feature set.
    assert state.selection_method == "unsupervised_all"
    assert state.selected_columns is not None
    assert len(state.selected_columns) >= 2
    # Validator coerced + filled, so no residual NaNs and no target column added.
    assert state.df.isna().sum().sum() == 0
    assert state.target is None


def test_orchestrator_constructor_accepts_unsupervised_and_none_target() -> None:
    orchestrator = FeatureEngineerOrchestrator(
        data_path="x.csv",
        target_column=None,
        task="unsupervised",
        model_string="openai/test-model",
        llm_settings=None,
    )
    assert orchestrator.target_column is None
    assert orchestrator.task == "unsupervised"
