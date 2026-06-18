import json
from pathlib import Path

from backend.session import SessionManager


def test_create_session_uses_timestamp_slug_uuid(tmp_path: Path) -> None:
    manager = SessionManager(workspace_root=tmp_path)

    session = manager.create_session(original_filename="Iris Data.csv")

    session_id_parts = session.session_id.split("_")
    assert len(session_id_parts[0]) == 8
    assert len(session_id_parts[1]) == 6
    assert "_iris_data_" in session.session_id
    assert len(session_id_parts[-1]) == 8
    assert (tmp_path / session.session_id / "data").is_dir()
    assert (tmp_path / session.session_id / "reports").is_dir()
    assert (tmp_path / session.session_id / "session.json").is_file()


def test_write_session_metadata_merges_existing_values(tmp_path: Path) -> None:
    manager = SessionManager(workspace_root=tmp_path)
    session = manager.create_session(original_filename="iris.csv")

    manager.write_session_metadata(
        session_id=session.session_id,
        updates={"row_count": 150, "column_count": 5},
    )

    session_metadata = json.loads(
        (tmp_path / session.session_id / "session.json").read_text(encoding="utf-8")
    )
    assert session_metadata["original_filename"] == "iris.csv"
    assert session_metadata["row_count"] == 150
    assert session_metadata["column_count"] == 5


def test_list_recent_uploads_uses_session_metadata_and_data_file(tmp_path: Path) -> None:
    manager = SessionManager(workspace_root=tmp_path)
    older_session = manager.create_session(original_filename="older.csv")
    newer_session = manager.create_session(original_filename="newer.csv")

    (tmp_path / older_session.session_id / "data" / "data.csv").write_text(
        "a,b\n1,2\n", encoding="utf-8"
    )
    (tmp_path / newer_session.session_id / "data" / "data.csv").write_text(
        "a,b\n3,4\n", encoding="utf-8"
    )

    recent_uploads = manager.list_recent_uploads(limit=1)

    assert len(recent_uploads) == 1
    assert recent_uploads[0]["session_id"] == newer_session.session_id
    assert recent_uploads[0]["original_filename"] == "newer.csv"


def test_list_recent_uploads_skips_sessions_without_data_csv(tmp_path: Path) -> None:
    manager = SessionManager(workspace_root=tmp_path)
    session = manager.create_session(original_filename="pending.csv")

    recent_uploads = manager.list_recent_uploads(limit=5)

    assert session.session_id not in [
        upload["session_id"]
        for upload in recent_uploads
    ]
