from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    session_path: Path
    data_dir: Path
    reports_dir: Path
    original_filename: str
    uploaded_at: str


class SessionManager:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def create_session(self, original_filename: str) -> SessionInfo:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_slug = self._slugify_dataset_name(original_filename=original_filename)
        uuid_suffix = uuid.uuid4().hex[:8]
        session_id = f"{timestamp}_{dataset_slug}_{uuid_suffix}"

        session_path = self.workspace_root / session_id
        data_dir = session_path / "data"
        reports_dir = session_path / "reports"
        data_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        uploaded_at = datetime.now().isoformat(timespec="seconds")
        metadata = {
            "session_id": session_id,
            "original_filename": original_filename,
            "uploaded_at": uploaded_at,
            "created_at_epoch": time(),
        }
        self._write_json(path=session_path / "session.json", data=metadata)

        return SessionInfo(
            session_id=session_id,
            session_path=session_path,
            data_dir=data_dir,
            reports_dir=reports_dir,
            original_filename=original_filename,
            uploaded_at=uploaded_at,
        )

    def get_session_path(self, session_id: str) -> Path:
        if Path(session_id).name != session_id or ".." in Path(session_id).parts:
            raise ValueError(f"Invalid session_id: {session_id}")
        return self.workspace_root / session_id

    def write_session_metadata(
        self,
        session_id: str,
        updates: dict[str, object],
    ) -> None:
        session_path = self.get_session_path(session_id=session_id)
        metadata_path = session_path / "session.json"
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        merged_metadata = {**existing_metadata, **updates}
        self._write_json(path=metadata_path, data=merged_metadata)

    def list_recent_uploads(self, limit: int) -> list[dict[str, object]]:
        if not self.workspace_root.exists():
            return []

        upload_records: list[dict[str, object]] = []
        for session_path in self.workspace_root.iterdir():
            metadata_path = session_path / "session.json"
            data_file_path = session_path / "data" / "data.csv"
            if not session_path.is_dir() or not metadata_path.is_file():
                continue
            if not data_file_path.is_file():
                continue

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            upload_records.append(
                {
                    "session_id": metadata.get("session_id", session_path.name),
                    "original_filename": metadata.get("original_filename", "data.csv"),
                    "uploaded_at": metadata.get("uploaded_at", ""),
                    "created_at_epoch": metadata.get("created_at_epoch", 0),
                    "file_size_bytes": data_file_path.stat().st_size,
                    "row_count": metadata.get("row_count"),
                    "column_count": metadata.get("column_count"),
                    "task_type": metadata.get("task_type"),
                }
            )

        sorted_records = sorted(
            upload_records,
            key=lambda upload_record: float(upload_record["created_at_epoch"]),
            reverse=True,
        )
        return sorted_records[:limit]

    @staticmethod
    def _slugify_dataset_name(original_filename: str) -> str:
        filename_stem = Path(original_filename).stem.lower()
        normalized_name = re.sub(r"[^a-z0-9]+", "_", filename_stem).strip("_")
        return normalized_name or "dataset"

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
