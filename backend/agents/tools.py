from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft7Validator

from backend.session import SessionManager


# Statistics fields, split by the JSON type the metadata schema expects for each.
STATISTIC_NUMBER_KEYS = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "freq"]
STATISTIC_STRING_KEYS = ["top"]


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
        return self._mini_data_path(session_id=session_id).read_text(encoding="utf-8")

    def build_statistics(self, session_id: str) -> dict[str, dict[str, Any]]:
        # mini_data.csv is already df.describe(include="all").transpose(), so the
        # per-column statistics are read straight back rather than asking the LLM
        # to transcribe them (which it does unreliably).
        mini_data_path = self._mini_data_path(session_id=session_id)
        describe_frame = pd.read_csv(mini_data_path, index_col=0)
        statistics: dict[str, dict[str, Any]] = {}
        for column_name, column_row in describe_frame.iterrows():
            column_statistics: dict[str, Any] = {}
            for number_key in STATISTIC_NUMBER_KEYS:
                if number_key in column_row.index:
                    column_statistics[number_key] = self._to_number_or_none(
                        value=column_row[number_key]
                    )
            for string_key in STATISTIC_STRING_KEYS:
                if string_key in column_row.index:
                    column_statistics[string_key] = self._to_string_or_none(
                        value=column_row[string_key]
                    )
            statistics[str(column_name)] = column_statistics
        return statistics

    def _mini_data_path(self, session_id: str) -> Path:
        session_path = self.session_manager.get_session_path(session_id=session_id)
        mini_data_path = session_path / "data" / "mini_data.csv"
        if not mini_data_path.is_file():
            raise FileNotFoundError(
                f"mini_data.csv not found for session: {session_id}"
            )
        return mini_data_path

    @staticmethod
    def _to_number_or_none(value: Any) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _to_string_or_none(value: Any) -> str | None:
        if pd.isna(value):
            return None
        return str(value)

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
