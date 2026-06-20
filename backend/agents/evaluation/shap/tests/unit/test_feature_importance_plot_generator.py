"""Unit tests for backend.agents.evaluation.shap.visualizations.feature_importance_plot_generator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.evaluation.shap.errors import VisualizationError
from backend.agents.evaluation.shap.visualizations.feature_importance_plot_generator import (
    FeatureImportancePlotGenerator,
    _FEATURE_IMPORTANCE_BAR_FILENAME,
)
from backend.agents.evaluation.shap.tests.fixtures.fixture_factory import FixtureFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHAP_SUMMARY_PLOT_TARGET = (
    "backend.agents.evaluation.shap.visualizations.feature_importance_plot_generator.shap.summary_plot"
)


def _make_generator(
    tmp_path: Path, max_display_features: int = 10
) -> FeatureImportancePlotGenerator:
    return FeatureImportancePlotGenerator(
        execution_logger=FixtureFactory.make_execution_logger(tmp_path),
        plot_format="PNG",
        max_display_features=max_display_features,
    )


# ---------------------------------------------------------------------------
# File creation and return value
# ---------------------------------------------------------------------------

def test_render_creates_nonempty_file(tmp_path: Path) -> None:
    """render() must create a non-empty PNG file at the specified output path."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET):
        generator.render(shap_result, output_path)

    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_render_returns_output_path(tmp_path: Path) -> None:
    """render() must return the path that was written."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET):
        returned_path = generator.render(shap_result, output_path)

    assert returned_path == output_path


def test_render_creates_parent_directory_if_missing(tmp_path: Path) -> None:
    """render() must mkdir -p the parent directory if it does not exist."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "nested" / "deep" / _FEATURE_IMPORTANCE_BAR_FILENAME

    assert not output_path.parent.exists()

    with patch(_SHAP_SUMMARY_PLOT_TARGET):
        generator.render(shap_result, output_path)

    assert output_path.exists()


# ---------------------------------------------------------------------------
# Prediction type support
# ---------------------------------------------------------------------------

def test_render_binary_prediction_type(tmp_path: Path) -> None:
    """render() must complete successfully for binary_classification SHAPResult."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET) as mock_summary:
        generator.render(shap_result, output_path)

    mock_summary.assert_called_once()
    assert output_path.exists()


def test_render_regression_prediction_type(tmp_path: Path) -> None:
    """render() must complete successfully for regression SHAPResult."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_regression()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET):
        result = generator.render(shap_result, output_path)

    assert result == output_path
    assert output_path.exists()


def test_render_multiclass_prediction_type(tmp_path: Path) -> None:
    """render() must complete successfully for multiclass_classification SHAPResult."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_multiclass()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET):
        result = generator.render(shap_result, output_path)

    assert result == output_path
    assert output_path.exists()


# ---------------------------------------------------------------------------
# SHAP call arguments
# ---------------------------------------------------------------------------

def test_render_passes_bar_plot_type(tmp_path: Path) -> None:
    """render() must pass plot_type='bar' to shap.summary_plot()."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET) as mock_summary:
        generator.render(shap_result, output_path)

    _, kwargs = mock_summary.call_args
    assert kwargs["plot_type"] == "bar"


def test_render_passes_shap_values_array(tmp_path: Path) -> None:
    """render() must pass shap_result.shap_values_array as the first positional arg."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET) as mock_summary:
        generator.render(shap_result, output_path)

    args, _ = mock_summary.call_args
    assert args[0] is shap_result.shap_values_array


def test_render_passes_feature_names(tmp_path: Path) -> None:
    """render() must pass list(shap_result.feature_names) as feature_names kwarg."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET) as mock_summary:
        generator.render(shap_result, output_path)

    _, kwargs = mock_summary.call_args
    assert kwargs["feature_names"] == list(shap_result.feature_names)


def test_render_passes_features_none(tmp_path: Path) -> None:
    """render() must pass features=None since bar chart does not use feature values."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET) as mock_summary:
        generator.render(shap_result, output_path)

    _, kwargs = mock_summary.call_args
    assert kwargs["features"] is None


def test_render_passes_show_false(tmp_path: Path) -> None:
    """render() must pass show=False to suppress GUI display."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET) as mock_summary:
        generator.render(shap_result, output_path)

    _, kwargs = mock_summary.call_args
    assert kwargs["show"] is False


def test_render_passes_max_display_features(tmp_path: Path) -> None:
    """render() must forward max_display_features as max_display kwarg."""
    generator = _make_generator(tmp_path, max_display_features=8)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET) as mock_summary:
        generator.render(shap_result, output_path)

    _, kwargs = mock_summary.call_args
    assert kwargs["max_display"] == 8


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_render_raises_visualization_error_on_shap_failure(tmp_path: Path) -> None:
    """VisualizationError must be raised when shap.summary_plot() raises."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    with patch(_SHAP_SUMMARY_PLOT_TARGET, side_effect=RuntimeError("shap failed")):
        with pytest.raises(VisualizationError, match="Failed to generate feature importance bar"):
            generator.render(shap_result, output_path)


def test_render_raises_visualization_error_on_save_failure(tmp_path: Path) -> None:
    """VisualizationError must be raised when plt.savefig() raises."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    save_target = (
        "backend.agents.evaluation.shap.visualizations.feature_importance_plot_generator.plt.savefig"
    )
    with patch(_SHAP_SUMMARY_PLOT_TARGET):
        with patch(save_target, side_effect=OSError("disk full")):
            with pytest.raises(VisualizationError, match="Failed to generate feature importance bar"):
                generator.render(shap_result, output_path)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def test_render_logs_plot_generation_events(tmp_path: Path) -> None:
    """render() must call log_plot_generation at least twice (start and done)."""
    logger_mock = MagicMock()
    generator = FeatureImportancePlotGenerator(
        execution_logger=logger_mock,
        plot_format="PNG",
        max_display_features=10,
    )
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with patch(_SHAP_SUMMARY_PLOT_TARGET):
        generator.render(shap_result, output_path)

    assert logger_mock.log_plot_generation.call_count >= 2


# ---------------------------------------------------------------------------
# Figure lifecycle
# ---------------------------------------------------------------------------

def test_no_figure_leak_after_render(tmp_path: Path) -> None:
    """plt.close('all') must be called after render to prevent figure accumulation."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    close_target = (
        "backend.agents.evaluation.shap.visualizations.feature_importance_plot_generator.plt.close"
    )
    with patch(_SHAP_SUMMARY_PLOT_TARGET):
        with patch(close_target) as mock_close:
            generator.render(shap_result, output_path)

    mock_close.assert_called_with("all")


def test_figures_closed_even_on_error(tmp_path: Path) -> None:
    """plt.close('all') must be called in the finally block even when render fails."""
    generator = _make_generator(tmp_path)
    shap_result = FixtureFactory.make_shap_result_binary()
    output_path = tmp_path / "plots" / _FEATURE_IMPORTANCE_BAR_FILENAME

    close_target = (
        "backend.agents.evaluation.shap.visualizations.feature_importance_plot_generator.plt.close"
    )
    with patch(_SHAP_SUMMARY_PLOT_TARGET, side_effect=RuntimeError("shap failed")):
        with patch(close_target) as mock_close:
            with pytest.raises(VisualizationError):
                generator.render(shap_result, output_path)

    mock_close.assert_called_with("all")
