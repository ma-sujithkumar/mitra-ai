"""Dataset2Vec warm-start bridge.

Two responsibilities:
  1. QUERY (before training): embed a new dataset, search the meta-knowledge
     store for similar past datasets, return a DatasetPrior with ranked_models
     that is fed into model_selection as a warm-start prior.
  2. WRITE-BACK (after judge): append the new dataset's embedding + final
     leaderboard to the DB so future runs benefit from it.

Concurrency:
  - Query is read-only; safe to run concurrently.
  - Write-back acquires a file-based lock (portalocker) so parallel runs
    don't corrupt leaderboards.parquet / meta_kb.parquet.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Bootstrap sys.path so dataset2vec internal imports resolve.
# d2v_bridge lives at backend/orchestration/ (2 levels below repo root).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_D2V_ROOT = str(Path(__file__).resolve().parents[2] / "backend" / "agents" / "dataset2vec")
for _path_entry in [_REPO_ROOT, _D2V_ROOT]:
    if _path_entry not in sys.path:
        sys.path.insert(0, _path_entry)

from d2v_core.encoder import Dataset2VecEncoder, EmbeddingGenerator
from d2v_core.sampling import CorpusSampler
from d2v_core.schema import DatasetPrior, NeighborResult, RankedModelEntry
from d2v_core.store import MetaKnowledgeStore

logger = logging.getLogger(__name__)

# Default FAISS and normalization settings — callers may override via kwargs.
_DEFAULT_FAISS_METRIC = "cosine"
_DEFAULT_NORMALIZE = True
_DEFAULT_TOP_K = 5


class D2VBridge:
    """Encapsulates dataset2Vec query and write-back for the orchestration layer.

    Args:
        db_dir: Path to the DB/ directory containing encoder.pt, *.parquet, index.faiss.
        top_k: Number of similar datasets to retrieve.
        faiss_metric: FAISS distance metric; must match how the index was built.
        normalize_embeddings: Whether to L2-normalise before FAISS search.
    """

    def __init__(
        self,
        db_dir: Path,
        top_k: int = _DEFAULT_TOP_K,
        faiss_metric: str = _DEFAULT_FAISS_METRIC,
        normalize_embeddings: bool = _DEFAULT_NORMALIZE,
    ) -> None:
        self.db_dir = Path(db_dir)
        self.top_k = top_k
        self.faiss_metric = faiss_metric
        self.normalize_embeddings = normalize_embeddings
        self._encoder: Optional[Dataset2VecEncoder] = None
        self._store: Optional[MetaKnowledgeStore] = None

    def _get_store(self) -> MetaKnowledgeStore:
        if self._store is None:
            self._store = MetaKnowledgeStore(
                store_dir=str(self.db_dir),
                faiss_metric=self.faiss_metric,
                normalize_embeddings=self.normalize_embeddings,
            )
        return self._store

    def _load_encoder(self, n_classes_sample: int = 5) -> Dataset2VecEncoder:
        """Load the trained encoder from db_dir/encoder/encoder.pt."""
        import torch

        encoder_path = self.db_dir / "encoder" / "encoder.pt"
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"=> encoder.pt not found at {encoder_path}. Run train_encoder.py first."
            )
        # Minimal encoder config matching the default architecture used at training time.
        # The exact config is baked into the state_dict shapes, so architecture
        # must match. These defaults correspond to the D2V training in Epic 3.
        encoder_config: Dict[str, Any] = {
            "embedding_dim": 128,
            "n_attention_heads": 4,
            "n_encoder_layers": 4,
            "dropout": 0.1,
        }
        encoder = Dataset2VecEncoder(encoder_config, n_classes_sample=n_classes_sample)
        encoder.load_state_dict(torch.load(str(encoder_path), map_location="cpu"))
        encoder.eval()
        logger.info("=> D2VBridge: loaded encoder from %s", encoder_path)
        return encoder

    def _csv_to_npz(self, csv_path: Path, target_column: str, npz_dir: Path) -> Path:
        """Convert a CSV dataset to the NPZ format expected by CorpusSampler.

        Returns path to the written .npz file.
        """
        dataframe = pd.read_csv(csv_path)
        feature_matrix = dataframe.drop(columns=[target_column]).to_numpy(dtype=np.float64)
        target_vector = dataframe[target_column].to_numpy(dtype=np.float64)
        dataset_id = csv_path.stem
        npz_path = npz_dir / f"{dataset_id}.npz"
        np.savez(str(npz_path), X_train=feature_matrix, y_train=target_vector)
        logger.debug("=> D2VBridge: wrote NPZ to %s (shape=%s)", npz_path, feature_matrix.shape)
        return npz_path

    def query(
        self,
        csv_path: Path,
        target_column: str,
        task_type: str,
        n_instances_sample: int = 100,
        n_features_sample: int = 50,
        n_classes_sample: int = 5,
        random_state: int = 42,
    ) -> Optional[DatasetPrior]:
        """Embed a new dataset and retrieve warm-start priors from the meta-KB.

        Returns None if the encoder or meta-KB is absent (cold start) so the
        caller can fall back to default model selection without crashing.
        """
        import torch

        store = self._get_store()
        if store.is_empty():
            logger.info("=> D2VBridge.query: meta-KB is empty, returning cold-start prior.")
            return DatasetPrior(
                query_dataset_id=csv_path.stem,
                encoder_version="unknown",
                top_k=self.top_k,
                primary_metric="accuracy" if task_type == "classification" else "r2",
                neighbors=[],
                ranked_models=[],
                cold_start=True,
                caveats=["meta-KB is empty; no warm-start priors available."],
            )

        encoder_path = self.db_dir / "encoder" / "encoder.pt"
        if not encoder_path.exists():
            logger.warning("=> D2VBridge.query: encoder.pt not found; returning cold-start prior.")
            return DatasetPrior(
                query_dataset_id=csv_path.stem,
                encoder_version="unknown",
                top_k=self.top_k,
                primary_metric="accuracy" if task_type == "classification" else "r2",
                neighbors=[],
                ranked_models=[],
                cold_start=True,
                caveats=["encoder.pt not found; no warm-start priors available."],
            )

        encoder = self._load_encoder(n_classes_sample=n_classes_sample)

        with tempfile.TemporaryDirectory(prefix="d2v_query_") as temp_dir:
            temp_path = Path(temp_dir)
            npz_path = self._csv_to_npz(csv_path=csv_path, target_column=target_column, npz_dir=temp_path)

            training_config: Dict[str, Any] = {
                "n_instances_sample": n_instances_sample,
                "n_features_sample": n_features_sample,
                "n_classes_sample": n_classes_sample,
                "random_state": random_state,
            }
            sampler = CorpusSampler(
                corpus_dir=str(temp_path),
                **training_config,
            )
            embedding_generator = EmbeddingGenerator(encoder, device=torch.device("cpu"))
            embedding_tensor = embedding_generator.encode_dataset(sampler, csv_path.stem)
            embedding_vector = embedding_tensor.cpu().numpy()

        # Search meta-KB for nearest neighbors
        neighbor_tuples = store.search(
            query_vector=embedding_vector,
            top_k=self.top_k,
            task_type=task_type,
            same_task_only=True,
        )
        if not neighbor_tuples:
            logger.info("=> D2VBridge.query: no neighbors found for dataset %s", csv_path.stem)
            return DatasetPrior(
                query_dataset_id=csv_path.stem,
                encoder_version="unknown",
                top_k=self.top_k,
                primary_metric="accuracy" if task_type == "classification" else "r2",
                neighbors=[],
                ranked_models=[],
                cold_start=True,
                caveats=["no similar datasets found in meta-KB"],
            )

        # Build NeighborResult + RankedModelEntry from store results
        meta_kb_df = store.load_meta_kb()
        neighbors: List[NeighborResult] = []
        ranked_models_raw: Dict[str, float] = {}

        for neighbor_dataset_id, similarity_score in neighbor_tuples:
            neighbor_row = meta_kb_df[meta_kb_df["dataset_id"] == neighbor_dataset_id]
            if neighbor_row.empty:
                continue
            neighbor_leaderboard = neighbor_row.iloc[0]["leaderboard"] or []
            best_entry = neighbor_leaderboard[0] if neighbor_leaderboard else {}
            best_model_name = best_entry.get("model_name", "") if isinstance(best_entry, dict) else ""
            neighbors.append(NeighborResult(
                dataset_id=neighbor_dataset_id,
                similarity=float(similarity_score),
                best_model=best_model_name,
                recommended_hyperparameters=best_entry.get("hyperparameters", {}) if isinstance(best_entry, dict) else {},
                metrics=best_entry.get("metrics", {}) if isinstance(best_entry, dict) else {},
            ))
            # Accumulate similarity-weighted score per model name
            ranked_models_raw[best_model_name] = ranked_models_raw.get(best_model_name, 0.0) + float(similarity_score)

        ranked_entries: List[RankedModelEntry] = [
            RankedModelEntry(
                model_name=model_name,
                score=score_value,
                recommended_hyperparameters={},
                expected_metric=0.0,
            )
            for model_name, score_value in sorted(ranked_models_raw.items(), key=lambda pair: pair[1], reverse=True)
        ]

        primary_metric = "accuracy" if task_type == "classification" else "r2"
        dataset_prior = DatasetPrior(
            query_dataset_id=csv_path.stem,
            encoder_version="unknown",
            top_k=self.top_k,
            primary_metric=primary_metric,
            neighbors=neighbors,
            ranked_models=ranked_entries,
            cold_start=False,
            caveats=[],
        )
        logger.info(
            "=> D2VBridge.query: found %d neighbors, %d ranked models for %s",
            len(neighbors),
            len(ranked_entries),
            csv_path.stem,
        )
        return dataset_prior

    def write_back(
        self,
        csv_path: Path,
        target_column: str,
        task_type: str,
        judge_decision: Any,
        n_instances_sample: int = 100,
        n_features_sample: int = 50,
        n_classes_sample: int = 5,
        random_state: int = 42,
    ) -> None:
        """Append the new dataset's embedding + leaderboard to the meta-KB.

        This is a single-writer operation; acquires a lock before writing to
        prevent corruption from concurrent pipeline runs.
        """
        import portalocker

        store = self._get_store()
        lock_path = self.db_dir / ".write.lock"

        # Build embedding for this dataset (same pipeline as query)
        encoder_path = self.db_dir / "encoder" / "encoder.pt"
        embedding_row: Optional[Dict[str, Any]] = None
        if encoder_path.exists():
            import torch
            encoder = self._load_encoder(n_classes_sample=n_classes_sample)
            with tempfile.TemporaryDirectory(prefix="d2v_wb_") as temp_dir:
                temp_path = Path(temp_dir)
                npz_path = self._csv_to_npz(csv_path=csv_path, target_column=target_column, npz_dir=temp_path)
                training_config: Dict[str, Any] = {
                    "n_instances_sample": n_instances_sample,
                    "n_features_sample": n_features_sample,
                    "n_classes_sample": n_classes_sample,
                    "random_state": random_state,
                }
                sampler = CorpusSampler(corpus_dir=str(temp_path), **training_config)
                embedding_generator = EmbeddingGenerator(encoder, device=torch.device("cpu"))
                embedding_tensor = embedding_generator.encode_dataset(sampler, csv_path.stem)
                embedding_vector = embedding_tensor.cpu().numpy().tolist()
            embedding_row = {
                "dataset_id": csv_path.stem,
                "embedding": embedding_vector,
                "task_type": task_type,
            }
        else:
            logger.warning("=> D2VBridge.write_back: no encoder.pt; skipping embedding.")

        # Build leaderboard record from judge_decision
        ranked_models_list = getattr(judge_decision, "ranked_models", []) or []
        primary_metric = "accuracy" if task_type == "classification" else "r2"
        leaderboard_entries: List[Dict[str, Any]] = [
            {
                "rank": ranked_model.rank,
                "model_name": ranked_model.model_name,
                "hyperparameters": {},
                "metrics": {"score": ranked_model.score},
                "n_trials": 1,
            }
            for ranked_model in ranked_models_list
        ]
        best_model_name = getattr(judge_decision, "selected_model", None) or ""

        dataframe = pd.read_csv(csv_path)
        leaderboard_record: Dict[str, Any] = {
            "dataset_id": csv_path.stem,
            "encoder_version": "unknown",
            "embedding": None,  # filled in by build_meta_kb join
            "task_type": task_type,
            "n_rows": len(dataframe),
            "n_cols": dataframe.shape[1] - 1,
            "target_cardinality": int(dataframe[target_column].nunique()),
            "primary_metric": primary_metric,
            "leaderboard": leaderboard_entries,
            "best_model": best_model_name,
            "created_at": "now",
        }

        # Acquire write lock before touching parquet files
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)
            if embedding_row:
                # Append to train_embeddings — read + deduplicate + overwrite
                self._upsert_embedding(store, embedding_row)
            store.write_leaderboard_record(leaderboard_record)
            store.build_meta_kb()
            portalocker.unlock(lock_file)

        logger.info(
            "=> D2VBridge.write_back: wrote embedding + leaderboard for %s (best=%s)",
            csv_path.stem,
            best_model_name,
        )

    def _upsert_embedding(
        self,
        store: MetaKnowledgeStore,
        new_embedding_row: Dict[str, Any],
    ) -> None:
        """Append or update one embedding row in train_embeddings.parquet."""
        from d2v_core.store import EMBEDDINGS_PARQUET
        embeddings_path = Path(store.store_dir) / EMBEDDINGS_PARQUET
        dataset_id = new_embedding_row["dataset_id"]

        if embeddings_path.exists():
            existing_df = pd.read_parquet(str(embeddings_path))
            existing_df = existing_df[existing_df["dataset_id"] != dataset_id]
            updated_df = pd.concat([existing_df, pd.DataFrame([new_embedding_row])], ignore_index=True)
        else:
            updated_df = pd.DataFrame([new_embedding_row])

        store.write_embeddings(updated_df.to_dict(orient="records"))
        logger.debug("=> D2VBridge: upserted embedding for dataset_id=%s", dataset_id)
