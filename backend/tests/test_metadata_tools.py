import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from backend.agents.metadata_gen_agent import MetadataAgentToolAdapter
from backend.agents.tools import MetadataTools
from backend.session import SessionManager


def valid_metadata(session_id: str) -> dict[str, object]:
    return {
        "session_id": session_id,
        "problem_type": "supervised",
        "problem_subtype": "classification",
        "target_col": "target",
        "target_col_type": "categorical",
        "input_cols": [
            {
                "name": "feature",
                "col_type": "numeric",
            }
        ],
        "cols_to_drop": [],
        "statistics": {
            "feature": {
                "count": 2,
                "mean": 1.5,
                "std": 0.5,
                "min": 1,
                "25%": 1.25,
                "50%": 1.5,
                "75%": 1.75,
                "max": 2,
                "top": None,
                "freq": None,
            }
        },
    }


def create_session_with_mini_data(workspace_root: Path) -> str:
    session_manager = SessionManager(workspace_root=workspace_root)
    session_info = session_manager.create_session(original_filename="dataset.csv")
    (session_info.data_dir / "mini_data.csv").write_text(
        ",count,mean\nfeature,2,1.5\n",
        encoding="utf-8",
    )
    (session_info.data_dir / "data.csv").write_text(
        "feature,target\n1,a\n2,b\n",
        encoding="utf-8",
    )
    return session_info.session_id


def test_read_mini_data_reads_only_session_mini_data(tmp_path: Path) -> None:
    session_id = create_session_with_mini_data(workspace_root=tmp_path)
    tools = MetadataTools(workspace_root=tmp_path)

    mini_data = tools.read_mini_data(session_id=session_id)

    assert "feature,2,1.5" in mini_data
    assert "target" not in mini_data


def test_read_mini_data_rejects_path_traversal(tmp_path: Path) -> None:
    tools = MetadataTools(workspace_root=tmp_path)

    with pytest.raises(ValueError):
        tools.read_mini_data(session_id="../outside")


def test_write_metadata_validates_schema_and_writes_report(tmp_path: Path) -> None:
    session_id = create_session_with_mini_data(workspace_root=tmp_path)
    tools = MetadataTools(workspace_root=tmp_path)

    result = tools.write_metadata(
        session_id=session_id,
        metadata=valid_metadata(session_id=session_id),
    )

    assert result.metadata_path == tmp_path / session_id / "reports" / "metadata.json"
    assert result.metadata_path.is_file()


def test_write_metadata_rejects_invalid_metadata(tmp_path: Path) -> None:
    session_id = create_session_with_mini_data(workspace_root=tmp_path)
    tools = MetadataTools(workspace_root=tmp_path)
    invalid_metadata = valid_metadata(session_id=session_id)
    invalid_metadata.pop("statistics")

    with pytest.raises(ValidationError):
        tools.write_metadata(
            session_id=session_id,
            metadata=invalid_metadata,
        )


def create_session_with_columns(workspace_root: Path, columns: list[str]) -> str:
    session_manager = SessionManager(workspace_root=workspace_root)
    session_info = session_manager.create_session(original_filename="dataset.csv")
    mini_data_rows = "\n".join(f"{column},2,1.5" for column in columns)
    (session_info.data_dir / "mini_data.csv").write_text(
        ",count,mean\n" + mini_data_rows + "\n",
        encoding="utf-8",
    )
    return session_info.session_id


def read_metadata(workspace_root: Path, session_id: str) -> dict[str, object]:
    metadata_path = workspace_root / session_id / "reports" / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def test_adapter_drops_pii_and_excluded_columns(tmp_path: Path) -> None:
    session_id = create_session_with_columns(
        workspace_root=tmp_path,
        columns=["email", "age", "target"],
    )
    adapter = MetadataAgentToolAdapter(
        metadata_tools=MetadataTools(
            workspace_root=tmp_path,
            pii_patterns=["(?i)email"],
        )
    )
    adapter.write_metadata(
        session_id=session_id,
        metadata={
            "session_id": session_id,
            "problem_type": "supervised",
            "problem_subtype": "classification",
            "target_col": "target",
            "target_col_type": "categorical",
            "input_cols": [
                {"name": "email", "col_type": "categorical"},
                {"name": "age", "col_type": "numeric"},
            ],
            # User asked to ignore "age"; "email" is caught deterministically as PII.
            "cols_to_drop": ["age"],
            "statistics": {},
        },
    )

    metadata = read_metadata(workspace_root=tmp_path, session_id=session_id)
    assert metadata["cols_to_drop"] == ["age", "email"]
    assert metadata["input_cols"] == []
    assert "email" not in metadata["statistics"]
    assert "age" not in metadata["statistics"]
    assert "target" in metadata["statistics"]
    # mini_data.csv is pruned of dropped columns.
    mini_data_text = (tmp_path / session_id / "data" / "mini_data.csv").read_text("utf-8")
    assert "email" not in mini_data_text
    assert "age" not in mini_data_text
    assert "target" in mini_data_text


def test_adapter_injects_descriptions_and_important_only_from_user(tmp_path: Path) -> None:
    session_id = create_session_with_columns(
        workspace_root=tmp_path,
        columns=["age", "target"],
    )
    adapter = MetadataAgentToolAdapter(
        metadata_tools=MetadataTools(workspace_root=tmp_path),
        user_metadata_descriptions={"age": "Age in years"},
        user_metadata_important_cols=["age"],
    )
    adapter.write_metadata(
        session_id=session_id,
        metadata={
            "session_id": session_id,
            "problem_type": "supervised",
            "problem_subtype": "regression",
            "target_col": "target",
            "target_col_type": "numeric",
            "input_cols": [{"name": "age", "col_type": "numeric"}],
            "cols_to_drop": [],
            "important_cols": [],
            "statistics": {},
        },
    )

    metadata = read_metadata(workspace_root=tmp_path, session_id=session_id)
    assert metadata["statistics"]["age"]["description"] == "Age in years"
    assert "description" not in metadata["statistics"]["target"]
    assert metadata["important_cols"] == ["age"]


def test_adapter_maps_legacy_problem_type_to_supervised_subtype(tmp_path: Path) -> None:
    session_id = create_session_with_columns(
        workspace_root=tmp_path,
        columns=["age", "target"],
    )
    adapter = MetadataAgentToolAdapter(
        metadata_tools=MetadataTools(workspace_root=tmp_path)
    )
    adapter.write_metadata(
        session_id=session_id,
        metadata={
            "session_id": session_id,
            "problem_type": "classification",
            "target_col": "target",
            "target_col_type": "categorical",
            "input_cols": [{"name": "age", "col_type": "numeric"}],
            "cols_to_drop": [],
            "statistics": {},
        },
    )

    metadata = read_metadata(workspace_root=tmp_path, session_id=session_id)
    assert metadata["problem_type"] == "supervised"
    assert metadata["problem_subtype"] == "classification"
