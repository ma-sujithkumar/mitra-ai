from pathlib import Path

from backend.validator import DataValidator


def write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def check_status(results: list[object], key: str) -> str:
    check_result = next(result for result in results if result.key == key)
    return check_result.status


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

    validator = DataValidator(
        min_rows=10,
        null_threshold=0.8,
        pii_patterns=["(?i)email"],
    )
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
        )
    )

    assert [result.key for result in results] == DataValidator.check_order
    assert all(result.status == "pass" for result in results)


def test_validator_fails_null_heavy_column(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    write_csv(
        path=data_file,
        content="feature,target\n,x\n,y\n,z\n,w\n1,v\n",
    )

    validator = DataValidator(
        min_rows=1,
        null_threshold=0.5,
        pii_patterns=["(?i)email"],
    )
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

    validator = DataValidator(
        min_rows=1,
        null_threshold=0.8,
        pii_patterns=[],
    )
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

    validator = DataValidator(
        min_rows=1,
        null_threshold=0.8,
        pii_patterns=["(?i)email"],
    )
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

    validator = DataValidator(
        min_rows=1,
        null_threshold=0.8,
        pii_patterns=[],
    )
    results = list(
        validator.validate(
            data_file=data_file,
            session_id="sid",
            target_col="target",
        )
    )

    assert check_status(results=results, key="target") == "fail"
