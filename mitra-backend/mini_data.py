import logging
from pathlib import Path

import pandas

from config_loader import ConfigLoader
from session import SessionManager

logger = logging.getLogger(__name__)


class MiniDataGenerator:

    def generate(self, session_id: str, data_csv_path: Path) -> Path:
        chunk_size = ConfigLoader.get_int("upload", "CHUNK_SIZE_ROWS")
        sample_rows = ConfigLoader.get_int("upload", "MINI_DATA_SAMPLE_ROWS")

        logger.info(f"=> Generating mini_data.csv for session {session_id}")

        sampled_frames = []
        rows_collected = 0

        for chunk in pandas.read_csv(data_csv_path, chunksize=chunk_size):
            remaining = sample_rows - rows_collected
            if remaining <= 0:
                break
            take = min(len(chunk), remaining)
            sampled_frames.append(chunk.iloc[:take])
            rows_collected += take

        if not sampled_frames:
            raise ValueError(f"No data could be sampled from {data_csv_path}")

        sample_df = pandas.concat(sampled_frames, ignore_index=True)
        stats_df = sample_df.describe(include="all").transpose()

        output_path = SessionManager.get_session_path(session_id, "data/mini_data.csv")
        stats_df.to_csv(output_path)

        logger.info(f"=> mini_data.csv written: {output_path} ({len(sample_df)} rows sampled)")
        return output_path
