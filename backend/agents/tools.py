from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from backend.session import SessionManager


@dataclass(frozen=True)
class MetadataWriteResult:
    session_id: str
    metadata_path: Path


class MetadataTools:
    def __init__(self, workspace_root: Path, schema_path: Path | None = None) -> None:
        self.session_manager = SessionManager(workspace_root=workspace_root)
        self.schema_path = (
            schema_path
            or Path(__file__).resolve().parents[1]
            / "schemas"
            / "metadata_schema.json"
        )
        schema_payload = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.validator = Draft7Validator(schema_payload)

    def read_mini_data(self, session_id: str) -> str:
        session_path = self.session_manager.get_session_path(session_id=session_id)
        mini_data_path = session_path / "data" / "mini_data.csv"
        if not mini_data_path.is_file():
            raise FileNotFoundError(
                f"mini_data.csv not found for session: {session_id}"
            )
        return mini_data_path.read_text(encoding="utf-8")

    def write_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> MetadataWriteResult:
        session_path = self.session_manager.get_session_path(session_id=session_id)
        reports_dir = session_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        self.validator.validate(metadata)

        metadata_path = reports_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return MetadataWriteResult(
            session_id=session_id,
            metadata_path=metadata_path,
        )
