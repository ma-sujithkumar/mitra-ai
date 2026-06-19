from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import pandas as pd


# Candidate filenames for an optional user-uploaded metadata file, in probe order.
# Saved by the upload router as data/user_metadata.<ext> (see routers/upload.py).
USER_METADATA_FILENAMES = ["user_metadata.json", "user_metadata.csv"]

# Accepted column-header aliases when parsing a CSV/JSON metadata file.
NAME_FIELD_ALIASES = ["name", "column", "col", "feature", "field"]
DESCRIPTION_FIELD_ALIASES = ["description", "desc", "definition", "notes", "meaning"]
IMPORTANT_FIELD_ALIASES = ["important", "is_important", "key", "keep"]
TRUTHY_VALUES = {"true", "yes", "1", "y", "t", "important", "keep"}


@dataclass(frozen=True)
class UserMetadataHints:
    # Per-column descriptions and user-flagged important columns parsed from the
    # optional uploaded metadata file. Used to inject descriptions deterministically
    # and to seed important_cols (the LLM must not invent descriptions).
    descriptions: dict[str, str] = field(default_factory=dict)
    important_cols: list[str] = field(default_factory=list)


def find_user_metadata_path(session_path: Path) -> Path | None:
    # Single source of truth for locating the optional metadata file so the
    # validate and metadata routers do not duplicate the probe order.
    for metadata_filename in USER_METADATA_FILENAMES:
        metadata_path = session_path / "data" / metadata_filename
        if metadata_path.is_file():
            return metadata_path
    return None


def parse_user_metadata(metadata_path: Path) -> UserMetadataHints:
    # Best-effort extraction of column descriptions and important columns. A file
    # that cannot be parsed yields empty hints rather than raising, since these
    # are optional enrichments and the metadata file itself is already validated
    # for dataset relatedness elsewhere.
    if metadata_path.suffix.lower() == ".json":
        return _parse_json_metadata(metadata_path=metadata_path)
    return _parse_csv_metadata(metadata_path=metadata_path)


def _parse_json_metadata(metadata_path: Path) -> UserMetadataHints:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return UserMetadataHints()

    descriptions: dict[str, str] = {}
    important_cols: list[str] = []

    column_entries = _json_column_entries(payload=payload)
    for column_name, column_value in column_entries:
        description, is_important = _description_and_importance(value=column_value)
        if description:
            descriptions[column_name] = description
        if is_important:
            important_cols.append(column_name)

    # Allow an explicit top-level list as well.
    if isinstance(payload, dict):
        explicit_important = payload.get("important_cols")
        if isinstance(explicit_important, list):
            important_cols.extend(str(item) for item in explicit_important if str(item).strip())

    return UserMetadataHints(
        descriptions=descriptions,
        important_cols=_dedupe(important_cols),
    )


def _json_column_entries(payload: Any) -> list[tuple[str, Any]]:
    # Normalizes the common JSON shapes into (column_name, column_value) pairs.
    if isinstance(payload, dict) and isinstance(payload.get("columns"), list):
        entries: list[tuple[str, Any]] = []
        for item in payload["columns"]:
            if isinstance(item, dict):
                column_name = _first_alias_value(item, NAME_FIELD_ALIASES)
                if column_name:
                    entries.append((column_name, item))
        return entries
    if isinstance(payload, dict):
        return [
            (str(column_name), column_value)
            for column_name, column_value in payload.items()
            if str(column_name) not in {"columns", "important_cols"}
        ]
    return []


def _description_and_importance(value: Any) -> tuple[str | None, bool]:
    if isinstance(value, str):
        return (value.strip() or None, False)
    if isinstance(value, dict):
        description = _first_alias_value(value, DESCRIPTION_FIELD_ALIASES)
        important_raw = _first_alias_value(value, IMPORTANT_FIELD_ALIASES)
        is_important = important_raw is not None and important_raw.strip().lower() in TRUTHY_VALUES
        return (description, is_important)
    return (None, False)


def _parse_csv_metadata(metadata_path: Path) -> UserMetadataHints:
    try:
        metadata_frame = pd.read_csv(metadata_path, dtype=str).fillna("")
    except (ValueError, UnicodeDecodeError):
        return UserMetadataHints()

    lowered_columns = {str(column).strip().lower(): column for column in metadata_frame.columns}
    name_column = _first_present(lowered_columns, NAME_FIELD_ALIASES)
    description_column = _first_present(lowered_columns, DESCRIPTION_FIELD_ALIASES)
    important_column = _first_present(lowered_columns, IMPORTANT_FIELD_ALIASES)
    if name_column is None:
        return UserMetadataHints()

    descriptions: dict[str, str] = {}
    important_cols: list[str] = []
    for _, row in metadata_frame.iterrows():
        column_name = str(row[name_column]).strip()
        if not column_name:
            continue
        if description_column is not None:
            description_text = str(row[description_column]).strip()
            if description_text:
                descriptions[column_name] = description_text
        if important_column is not None and str(row[important_column]).strip().lower() in TRUTHY_VALUES:
            important_cols.append(column_name)

    return UserMetadataHints(
        descriptions=descriptions,
        important_cols=_dedupe(important_cols),
    )


def _first_alias_value(mapping: dict[str, Any], aliases: list[str]) -> str | None:
    lowered = {str(key).strip().lower(): value for key, value in mapping.items()}
    for alias in aliases:
        if alias in lowered and lowered[alias] is not None:
            text = str(lowered[alias]).strip()
            if text:
                return text
    return None


def _first_present(lowered_columns: dict[str, Any], aliases: list[str]) -> Any | None:
    for alias in aliases:
        if alias in lowered_columns:
            return lowered_columns[alias]
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
