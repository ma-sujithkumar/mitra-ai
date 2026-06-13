import json
import logging
from pathlib import Path

import jsonschema

from session import SessionManager

logger = logging.getLogger(__name__)

METADATA_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "metadata_schema.json"


def read_mini_data(session_id: str) -> str:
    mini_data_path = SessionManager.get_session_path(session_id, "data/mini_data.csv")
    if not mini_data_path.exists():
        raise FileNotFoundError(f"mini_data.csv not found for session {session_id}")
    content = mini_data_path.read_text(encoding="utf-8")
    logger.info(f"=> read_mini_data: returned {len(content)} chars for session {session_id}")
    return content


def write_metadata(session_id: str, metadata: dict) -> None:
    with open(METADATA_SCHEMA_PATH, "r") as schema_file:
        schema = json.load(schema_file)

    jsonschema.validate(instance=metadata, schema=schema)

    output_path = SessionManager.get_session_path(session_id, "reports/metadata.json")
    with open(output_path, "w") as output_file:
        json.dump(metadata, output_file, indent=2)

    logger.info(f"=> write_metadata: metadata.json written for session {session_id}")
