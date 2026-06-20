import json
from pathlib import Path

from backend.validator import DataValidator


def write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def check_status(results: list[object], key: str) -> str:
    check_result = next(result for result in results if result.key == key)
    return check_result.status


def build_validator(
    min_rows: int = 10,
    null_threshold: float = 0.8,
    null_drop_threshold: float = 0.5,
    pii_patterns: list[str] | None = None,
    metadata_match_min_overlap: float = 0.5,
) -> DataValidator:
    return DataValidator(
        min_rows=min_rows,
        null_threshold=null_threshold,
        null_drop_threshold=null_drop_threshold,
        pii_patterns=pii_patterns if pii_patterns is not None else ["(?i)email"],
        metadata_match_min_overlap=metadata_match_min_overlap,
    )


def test_validator_passes_clean_dataset(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    write_csv(
        path=data_file,
        content=(
            "feature_one,feature_two,target\n"
            "1,2,a\n2,3,a\n3,4,b\n4,5,b\n5,6,c\n6,7,c\n"
            "7,8,a\n8,9,b\n9,10,c\n10,11,a\n"
        ),
    )

    validator = build_validator(min_rows=10)
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
        )
    )

    # No metadata file supplied, so the optional metadata_match check is skipped.
    expected_keys = [key for key in DataValidator.check_order if key != "metadata_match"]
    assert [result.key for result in results] == expected_keys
    assert all(result.status == "pass" for result in results)


def test_validator_warns_and_autodrops_null_heavy_feature(tmp_path: Path) -> None:
    # A sparse non-target column should WARN (and be auto-dropped) rather than
    # block the run, matching the feature-engineering imputer behaviour.
    data_file = tmp_path / "data.csv"
    write_csv(
        path=data_file,
        content="feature,target\n,x\n,y\n,z\n,w\n1,v\n",
    )

    validator = build_validator(min_rows=1, null_threshold=0.5, null_drop_threshold=0.5)
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
        )
    )

    nulls_check = next(result for result in results if result.key == "nulls")
    assert nulls_check.status == "warn"
    assert nulls_check.meta is not None
    sparse_columns = [item["column"] for item in nulls_check.meta["columns"]]
    assert "feature" in sparse_columns
    assert nulls_check.meta["columns"][0]["action"] == "auto-drop"


def test_validator_blocks_null_heavy_target(tmp_path: Path) -> None:
    # The target itself cannot be dropped, so an empty target is a hard block.
    data_file = tmp_path / "data.csv"
    write_csv(
        path=data_file,
        content="feature,target\n1,\n2,\n3,\n4,\n5,v\n",
    )

    validator = build_validator(min_rows=1, null_threshold=0.5, null_drop_threshold=0.5)
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
        )
    )

    assert check_status(results=results, key="nulls") == "fail"


def test_validator_fails_constant_numeric_column(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    write_csv(
        path=data_file,
        content="constant,feature,target\n1,2,a\n1,3,b\n1,4,c\n",
    )

    validator = build_validator(min_rows=1, pii_patterns=[])
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
        )
    )

    assert check_status(results=results, key="variance") == "fail"


def test_validator_warns_for_pii_columns(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    write_csv(
        path=data_file,
        content="email,feature,target\na@example.com,1,a\nb@example.com,2,b\n",
    )

    validator = build_validator(min_rows=1)
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
        )
    )

    assert check_status(results=results, key="pii") == "warn"


def test_validator_fails_missing_target(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    write_csv(
        path=data_file,
        content="feature,other\n1,a\n2,b\n",
    )

    validator = build_validator(min_rows=1, pii_patterns=[])
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
        )
    )

    assert check_status(results=results, key="target") == "fail"


def test_metadata_match_skipped_when_no_metadata_file(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    write_csv(path=data_file, content="feature,target\n1,a\n2,b\n")

    validator = build_validator(min_rows=1, pii_patterns=[])
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
            user_metadata_path=None,
        )
    )

    # The optional metadata_match check should not run without a metadata file.
    assert "metadata_match" not in [result.key for result in results]


def test_metadata_match_passes_for_related_json(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    write_csv(
        path=data_file,
        content="age,income,target\n30,50000,a\n40,60000,b\n",
    )
    metadata_file = tmp_path / "user_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {"columns": [{"name": "age"}, {"name": "income"}, {"name": "target"}]}
        ),
        encoding="utf-8",
    )

    validator = build_validator(min_rows=1, pii_patterns=[])
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
            user_metadata_path=metadata_file,
        )
    )

    assert check_status(results=results, key="metadata_match") == "pass"


def test_metadata_match_fails_for_unrelated_file(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    write_csv(
        path=data_file,
        content="age,income,target\n30,50000,a\n40,60000,b\n",
    )
    metadata_file = tmp_path / "user_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {"columns": [{"name": "temperature"}, {"name": "humidity"}]}
        ),
        encoding="utf-8",
    )

    validator = build_validator(min_rows=1, pii_patterns=[])
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
            user_metadata_path=metadata_file,
        )
    )

    assert check_status(results=results, key="metadata_match") == "fail"


def test_metadata_match_fails_for_malformed_json(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    write_csv(path=data_file, content="age,target\n30,a\n40,b\n")
    metadata_file = tmp_path / "user_metadata.json"
    metadata_file.write_text("{not valid json", encoding="utf-8")

    validator = build_validator(min_rows=1, pii_patterns=[])
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
            user_metadata_path=metadata_file,
        )
    )

    assert check_status(results=results, key="metadata_match") == "fail"
