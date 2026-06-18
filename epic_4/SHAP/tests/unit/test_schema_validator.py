"""Unit tests for shap_explainability.validators.schema_validator."""

import pandas as pd
import pytest

from shap_explainability.errors import SchemaValidationError
from shap_explainability.session_context import ExecutionStatus
from shap_explainability.validators.schema_validator import SchemaValidationResult, SchemaValidator
from tests.fixtures.fixture_factory import FixtureFactory

_DEFAULT_CANDIDATES: tuple[str, ...] = ("target", "label", "outcome")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_validator(
    candidates: tuple[str, ...] = _DEFAULT_CANDIDATES,
    tmp_path=None,
) -> SchemaValidator:
    """Creates a SchemaValidator with a test logger.

    tmp_path is required to create a logger; pass pytest's tmp_path fixture.
    """
    import tempfile
    from pathlib import Path

    base_path = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    logger = FixtureFactory.make_execution_logger(base_path)
    return SchemaValidator(execution_logger=logger, target_column_candidates=candidates)


# ---------------------------------------------------------------------------
# Target column identification (Sec 11)
# ---------------------------------------------------------------------------

def test_target_column_identified_from_first_candidate_match(tmp_path) -> None:
    """First candidate present in the dataset is identified as the target column."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model()

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    assert result.target_column_name == "target"


def test_second_candidate_matched_when_first_absent(tmp_path) -> None:
    """When the first candidate is absent, the next candidate in order is matched."""
    validator = _make_validator(candidates=("target", "label", "outcome"), tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    # Dataset uses "label" instead of "target".
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="label",
    )
    loaded_model = FixtureFactory.make_loaded_model()

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    assert result.target_column_name == "label"


def test_target_column_excluded_from_feature_dataframe(tmp_path) -> None:
    """The target column must not appear in the returned feature_dataframe (Sec 11)."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model()

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    assert "target" not in result.feature_dataframe.columns
    assert set(result.feature_names) == {"feature_a", "feature_b"}


def test_no_target_column_found_adds_warning_to_session_context(tmp_path) -> None:
    """When no candidate matches, a non-terminating warning is recorded."""
    validator = _make_validator(candidates=("target",), tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    # Dataset has no column named "target".
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=False,
    )
    loaded_model = FixtureFactory.make_loaded_model()

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    assert len(session_context.warnings) >= 1
    assert result.target_column_name is None


def test_no_target_column_found_does_not_raise(tmp_path) -> None:
    """Missing target column is a non-terminating warning, not a failure (Sec 11)."""
    validator = _make_validator(candidates=("target",), tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=False,
    )
    loaded_model = FixtureFactory.make_loaded_model()

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    assert not session_context.has_failed()
    assert result is not None


# ---------------------------------------------------------------------------
# SessionContext population
# ---------------------------------------------------------------------------

def test_feature_names_populated_on_session_context(tmp_path) -> None:
    """SchemaValidator writes feature_names to session_context after validation."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model()

    validator.validate(loaded_dataset, loaded_model, session_context)

    assert session_context.feature_names == ["feature_a", "feature_b"]


def test_num_samples_and_num_features_populated_on_session_context(tmp_path) -> None:
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        num_rows=20,
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model()

    validator.validate(loaded_dataset, loaded_model, session_context)

    assert session_context.num_samples == 20
    assert session_context.num_features == 2


def test_target_column_name_populated_on_session_context(tmp_path) -> None:
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a"],
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model()

    validator.validate(loaded_dataset, loaded_model, session_context)

    assert session_context.target_column_name == "target"


# ---------------------------------------------------------------------------
# Zero features after target exclusion (Sec 9 terminating failure)
# ---------------------------------------------------------------------------

def test_only_target_column_in_dataset_raises_schema_validation_error(tmp_path) -> None:
    """If the dataset has only the target column, zero features remain — must fail (Sec 9)."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()

    # Build a dataset with only the target column and no feature columns.
    target_only_dataframe = pd.DataFrame({"target": [0, 1, 0, 1]})
    from shap_explainability.loaders.dataset_loader import LoadedDataset
    loaded_dataset = LoadedDataset(
        dataframe=target_only_dataframe,
        column_names=("target",),
        num_rows=4,
        num_columns=1,
    )
    loaded_model = FixtureFactory.make_loaded_model()

    with pytest.raises(SchemaValidationError):
        validator.validate(loaded_dataset, loaded_model, session_context)


def test_zero_features_marks_session_context_failed(tmp_path) -> None:
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()

    target_only_dataframe = pd.DataFrame({"target": [0, 1, 0, 1]})
    from shap_explainability.loaders.dataset_loader import LoadedDataset
    loaded_dataset = LoadedDataset(
        dataframe=target_only_dataframe,
        column_names=("target",),
        num_rows=4,
        num_columns=1,
    )
    loaded_model = FixtureFactory.make_loaded_model()

    with pytest.raises(SchemaValidationError):
        validator.validate(loaded_dataset, loaded_model, session_context)

    assert session_context.has_failed()


# ---------------------------------------------------------------------------
# Schema compatibility validation (Sec 12)
# ---------------------------------------------------------------------------

def test_no_model_metadata_skips_compatibility_and_adds_warning(tmp_path) -> None:
    """When model has no feature metadata, schema validation is skipped with a warning."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model(
        feature_names_from_model=None,
        num_features_from_model=None,
    )

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    # Validation should succeed with a warning about missing metadata.
    assert result is not None
    assert any("no feature metadata" in warning.lower() for warning in session_context.warnings)


def test_matching_feature_names_passes_validation(tmp_path) -> None:
    """When model feature names match the dataset, validation succeeds without warnings."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model(
        feature_names_from_model=("feature_a", "feature_b"),
        num_features_from_model=2,
    )

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    assert not session_context.has_failed()
    assert result.num_features == 2


def test_feature_count_mismatch_raises_schema_validation_error(tmp_path) -> None:
    """When dataset feature count differs from model metadata count, raise (Sec 12)."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="target",
    )
    # Model expects 5 features but dataset has 2.
    loaded_model = FixtureFactory.make_loaded_model(
        feature_names_from_model=None,
        num_features_from_model=5,
    )

    with pytest.raises(SchemaValidationError, match="mismatch"):
        validator.validate(loaded_dataset, loaded_model, session_context)


def test_feature_count_mismatch_marks_session_context_failed(tmp_path) -> None:
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model(
        feature_names_from_model=None,
        num_features_from_model=5,
    )

    with pytest.raises(SchemaValidationError):
        validator.validate(loaded_dataset, loaded_model, session_context)

    assert session_context.has_failed()


def test_feature_name_mismatch_raises_schema_validation_error(tmp_path) -> None:
    """When model feature names differ from dataset column names, raise (Sec 12)."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="target",
    )
    # Model was trained with different column names.
    loaded_model = FixtureFactory.make_loaded_model(
        feature_names_from_model=("col_x", "col_y"),
        num_features_from_model=2,
    )

    with pytest.raises(SchemaValidationError, match="mismatch"):
        validator.validate(loaded_dataset, loaded_model, session_context)


def test_count_only_metadata_validates_count(tmp_path) -> None:
    """When model only has num_features_from_model, count is validated without names."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a", "feature_b"],
        include_target_column=True,
        target_column_name="target",
    )
    # Model has count only — matches dataset.
    loaded_model = FixtureFactory.make_loaded_model(
        feature_names_from_model=None,
        num_features_from_model=2,
    )

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    assert not session_context.has_failed()
    assert result.num_features == 2


# ---------------------------------------------------------------------------
# Column ordering and Sec 13 dynamic feature names
# ---------------------------------------------------------------------------

def test_column_ordering_preserved_in_result(tmp_path) -> None:
    """Feature column order from the dataset must be preserved exactly (Sec 10, 13)."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["beta", "alpha", "gamma"],
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model()

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    assert list(result.feature_names) == ["beta", "alpha", "gamma"]
    assert list(result.feature_dataframe.columns) == ["beta", "alpha", "gamma"]


def test_schema_validation_result_is_correct_type(tmp_path) -> None:
    """validate() must return a SchemaValidationResult instance."""
    validator = _make_validator(tmp_path=tmp_path)
    session_context = FixtureFactory.make_session_context()
    loaded_dataset = FixtureFactory.make_loaded_dataset(
        feature_names=["feature_a"],
        include_target_column=True,
        target_column_name="target",
    )
    loaded_model = FixtureFactory.make_loaded_model()

    result = validator.validate(loaded_dataset, loaded_model, session_context)

    assert isinstance(result, SchemaValidationResult)
    assert isinstance(result.feature_dataframe, pd.DataFrame)
    assert isinstance(result.feature_names, tuple)
