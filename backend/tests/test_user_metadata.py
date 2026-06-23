import json
from pathlib import Path

from backend.services.training_service import TrainingService
from backend.user_metadata import parse_user_metadata


def test_parse_json_columns_list(tmp_path: Path) -> None:
    metadata_path = tmp_path / "user_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "columns": [
                    {"name": "age", "description": "Age in years", "important": True},
                    {"name": "city", "description": "City of residence"},
                ]
            }
        ),
        encoding="utf-8",
    )

    hints = parse_user_metadata(metadata_path=metadata_path)

    assert hints.descriptions == {"age": "Age in years", "city": "City of residence"}
    assert hints.important_cols == ["age"]


def test_parse_json_column_to_description_map(tmp_path: Path) -> None:
    metadata_path = tmp_path / "user_metadata.json"
    metadata_path.write_text(
        json.dumps({"age": "Age in years", "important_cols": ["age"]}),
        encoding="utf-8",
    )

    hints = parse_user_metadata(metadata_path=metadata_path)

    assert hints.descriptions == {"age": "Age in years"}
    assert hints.important_cols == ["age"]


def test_parse_csv_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "user_metadata.csv"
    metadata_path.write_text(
        "column,description,important\nage,Age in years,yes\ncity,City,no\n",
        encoding="utf-8",
    )

    hints = parse_user_metadata(metadata_path=metadata_path)

    assert hints.descriptions == {"age": "Age in years", "city": "City"}
    assert hints.important_cols == ["age"]


def test_parse_malformed_json_returns_empty(tmp_path: Path) -> None:
    metadata_path = tmp_path / "user_metadata.json"
    metadata_path.write_text("{not valid", encoding="utf-8")

    hints = parse_user_metadata(metadata_path=metadata_path)

    assert hints.descriptions == {}
    assert hints.important_cols == []


def test_legacy_problem_type_supervised_uses_subtype() -> None:
    assert TrainingService._legacy_problem_type(
        payload={"problem_type": "supervised", "problem_subtype": "regression"}
    ) == "regression"


def test_legacy_problem_type_supervised_falls_back_to_target_type() -> None:
    assert TrainingService._legacy_problem_type(
        payload={"problem_type": "supervised", "target_col_type": "numeric"}
    ) == "regression"
    assert TrainingService._legacy_problem_type(
        payload={"problem_type": "supervised", "target_col_type": "categorical"}
    ) == "classification"


def test_legacy_problem_type_unsupervised() -> None:
    assert TrainingService._legacy_problem_type(
        payload={"problem_type": "unsupervised"}
    ) == "unsupervised"


def test_legacy_problem_type_already_legacy_returns_none() -> None:
    # Old metadata.json files already use the legacy enum; nothing to translate.
    assert TrainingService._legacy_problem_type(
        payload={"problem_type": "classification"}
    ) is None
