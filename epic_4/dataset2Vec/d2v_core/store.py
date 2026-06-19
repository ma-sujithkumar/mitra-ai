import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EMBEDDINGS_PARQUET = "train_embeddings.parquet"
LEADERBOARDS_PARQUET = "leaderboards.parquet"
META_KB_PARQUET = "meta_kb.parquet"
FAISS_INDEX_FILE = "index.faiss"


def utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_leaderboard_entries(leaderboard_entries: list[dict]) -> list[dict]:
    """Parquet/PyArrow unifies struct schemas across every leaderboard entry in
    the file -- since different models have different hyperparameters (and
    different metrics), a nested struct column ends up padding each entry with
    None for keys that only belong to OTHER models, and can even upcast a
    field's type (e.g. an int hyperparameter becomes float once some other
    entry leaves it null). JSON-encoding hyperparameters/metrics as plain
    strings sidesteps struct unification entirely -- the column type is always
    str, so no padding or type coercion can happen."""
    encoded_entries = []
    for entry in leaderboard_entries:
        encoded_entry = dict(entry)
        encoded_entry["hyperparameters"] = json.dumps(entry["hyperparameters"])
        encoded_entry["metrics"] = json.dumps(entry["metrics"])
        encoded_entries.append(encoded_entry)
    return encoded_entries


def _decode_leaderboard_entries(leaderboard_entries: list[dict]) -> list[dict]:
    decoded_entries = []
    for entry in leaderboard_entries:
        decoded_entry = dict(entry)
        decoded_entry["hyperparameters"] = json.loads(entry["hyperparameters"])
        decoded_entry["metrics"] = json.loads(entry["metrics"])
        decoded_entries.append(decoded_entry)
    return decoded_entries


class MetaKnowledgeStore:
    """Owns all persistence for the meta-knowledge base: parquet files for
    embeddings and leaderboards, the inner-join view (meta_kb.parquet), and the
    FAISS similarity index. FAISS is treated as a derived artifact, rebuilt from
    meta_kb.parquet every time build_meta_kb() runs -- it is never hand-mutated."""

    def __init__(
        self, store_dir: str, faiss_metric: str, normalize_embeddings: bool
    ) -> None:
        self.store_dir = store_dir
        self.faiss_metric = faiss_metric
        self.normalize_embeddings = normalize_embeddings
        os.makedirs(self.store_dir, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self.store_dir, filename)

    def write_embeddings(self, rows: list[dict]) -> None:
        """Full overwrite of train_embeddings.parquet -- Phase 1 always re-embeds
        the entire corpus when it runs, so a full overwrite is correct and simple."""
        embeddings_df = pd.DataFrame(rows)
        embeddings_path = self._path(EMBEDDINGS_PARQUET)
        embeddings_df.to_parquet(embeddings_path, index=False)
        logger.info(
            "=> wrote %d rows to %s.", len(embeddings_df), EMBEDDINGS_PARQUET
        )

    def write_leaderboard_record(self, record: dict) -> None:
        """Upsert one row (one dataset_id) into leaderboards.parquet. Each
        leaderboard entry's hyperparameters/metrics are JSON-encoded first --
        see _encode_leaderboard_entries for why (existing rows in
        leaderboards.parquet are already encoded from a prior call, so only
        the new record needs encoding here)."""
        record = dict(record)
        record["leaderboard"] = _encode_leaderboard_entries(record["leaderboard"])

        leaderboards_path = self._path(LEADERBOARDS_PARQUET)
        if os.path.isfile(leaderboards_path):
            existing_df = pd.read_parquet(leaderboards_path)
            existing_df = existing_df[
                existing_df["dataset_id"] != record["dataset_id"]
            ]
            combined_df = pd.concat(
                [existing_df, pd.DataFrame([record])], ignore_index=True
            )
        else:
            combined_df = pd.DataFrame([record])
        combined_df.to_parquet(leaderboards_path, index=False)
        logger.info(
            "=> upserted leaderboard record for dataset_id='%s' (%d total rows).",
            record["dataset_id"],
            len(combined_df),
        )

    def build_meta_kb(self) -> int:
        """Inner-join train_embeddings.parquet + leaderboards.parquet on dataset_id
        -> meta_kb.parquet, then rebuild index.faiss from the joined embeddings.
        Idempotent and safe to call after either phase alone: if either side is
        missing, logs a warning and returns 0 rather than raising."""
        embeddings_path = self._path(EMBEDDINGS_PARQUET)
        leaderboards_path = self._path(LEADERBOARDS_PARQUET)
        if not os.path.isfile(embeddings_path) or not os.path.isfile(
            leaderboards_path
        ):
            logger.warning(
                "=> build_meta_kb: embeddings or leaderboards parquet missing "
                "(embeddings_exists=%s, leaderboards_exists=%s); skipping join.",
                os.path.isfile(embeddings_path),
                os.path.isfile(leaderboards_path),
            )
            return 0

        embeddings_df = pd.read_parquet(embeddings_path)
        leaderboards_df = pd.read_parquet(leaderboards_path)
        merged_df = pd.merge(
            embeddings_df,
            leaderboards_df,
            on="dataset_id",
            how="inner",
            suffixes=("", "_leaderboard"),
        )
        meta_kb_path = self._path(META_KB_PARQUET)
        merged_df.to_parquet(meta_kb_path, index=False)
        logger.info("=> wrote %d rows to %s.", len(merged_df), META_KB_PARQUET)

        self._rebuild_faiss_index(merged_df)
        return len(merged_df)

    def _rebuild_faiss_index(self, merged_df: pd.DataFrame) -> None:
        faiss_index_path = self._path(FAISS_INDEX_FILE)
        if len(merged_df) == 0:
            logger.warning("=> meta_kb is empty; skipping FAISS index build.")
            if os.path.isfile(faiss_index_path):
                os.remove(faiss_index_path)
            return

        embedding_matrix = np.stack(
            merged_df["embedding"].apply(lambda row: np.asarray(row, dtype=np.float32))
        )
        if self.normalize_embeddings:
            faiss.normalize_L2(embedding_matrix)

        embedding_dim = embedding_matrix.shape[1]
        faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss_index.add(embedding_matrix)
        faiss.write_index(faiss_index, faiss_index_path)
        logger.info(
            "=> rebuilt %s with %d vectors (dim=%d).",
            FAISS_INDEX_FILE,
            faiss_index.ntotal,
            embedding_dim,
        )

    def load_meta_kb(self) -> pd.DataFrame:
        meta_kb_path = self._path(META_KB_PARQUET)
        if not os.path.isfile(meta_kb_path):
            return pd.DataFrame()
        meta_kb_df = pd.read_parquet(meta_kb_path)
        if "leaderboard" in meta_kb_df.columns:
            meta_kb_df["leaderboard"] = meta_kb_df["leaderboard"].apply(_decode_leaderboard_entries)
        return meta_kb_df

    def is_empty(self) -> bool:
        meta_kb_df = self.load_meta_kb()
        return len(meta_kb_df) == 0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        task_type: str,
        same_task_only: bool,
    ) -> list[tuple[str, float]]:
        """Builds a fresh in-memory FAISS index from the (optionally task-filtered)
        rows of meta_kb.parquet and searches it. Rebuilding on demand instead of
        reusing the persisted index.faiss keeps task-filtering correct without
        needing to persist a row<->dataset_id mapping for every possible filter
        combination -- the corpus sizes this tool targets make this cheap."""
        meta_kb_df = self.load_meta_kb()
        if len(meta_kb_df) == 0:
            return []

        if same_task_only:
            meta_kb_df = meta_kb_df[meta_kb_df["task_type"] == task_type]
        if len(meta_kb_df) == 0:
            return []

        embedding_matrix = np.stack(
            meta_kb_df["embedding"].apply(lambda row: np.asarray(row, dtype=np.float32))
        )
        query_matrix = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if self.normalize_embeddings:
            faiss.normalize_L2(embedding_matrix)
            faiss.normalize_L2(query_matrix)

        embedding_dim = embedding_matrix.shape[1]
        temp_index = faiss.IndexFlatIP(embedding_dim)
        temp_index.add(embedding_matrix)

        effective_top_k = min(top_k, len(meta_kb_df))
        similarity_scores, neighbor_positions = temp_index.search(
            query_matrix, effective_top_k
        )

        dataset_ids = meta_kb_df["dataset_id"].tolist()
        results: list[tuple[str, float]] = []
        for position, score in zip(neighbor_positions[0], similarity_scores[0]):
            if position == -1:
                continue
            results.append((dataset_ids[position], float(score)))
        return results

    def completed_units(self) -> set[tuple[str, str]]:
        """Reads optuna study names of the form '{dataset_id}__{model_name}' that
        already exist in the SQLite Optuna storage, for --resume skip-logic in
        sweep.py. Returns an empty set if the optuna.db file does not yet exist
        (no side-effecting connection is made in that case)."""
        optuna_db_path = self._path("optuna.db")
        if not os.path.isfile(optuna_db_path):
            return set()

        connection = sqlite3.connect(optuna_db_path)
        try:
            cursor = connection.execute("SELECT study_name FROM studies")
            study_names = [row[0] for row in cursor.fetchall()]
        finally:
            connection.close()

        completed: set[tuple[str, str]] = set()
        for study_name in study_names:
            if "__" not in study_name:
                continue
            dataset_id, model_name = study_name.split("__", 1)
            completed.add((dataset_id, model_name))
        return completed
