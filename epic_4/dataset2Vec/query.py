import argparse
import configparser
import json
import logging
import os
import sys
import tempfile

# Bootstrap sys.path so model_library imports resolve from any cwd, same
# convention as d2v_core/sweep.py.
_INI_BOOTSTRAP_PATH = os.path.join(os.path.dirname(__file__), "config", "config.ini")
_boot_parser = configparser.ConfigParser()
_boot_parser.read(_INI_BOOTSTRAP_PATH)
_MODEL_LIBRARY_ROOT = _boot_parser.get("paths", "model_library_root")
if _MODEL_LIBRARY_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_LIBRARY_ROOT)

import numpy as np
import pandas as pd
import torch
from core.validators import validate_model_name

from d2v_core.encoder import Dataset2VecEncoder, EmbeddingGenerator
from d2v_core.sampling import CorpusSampler
from d2v_core.schema import (
    DatasetPrior,
    NeighborResult,
    RankedModelEntry,
    load_yaml_config,
    resolve_store_dir,
)
from d2v_core.store import MetaKnowledgeStore
from d2v_core.sweep import load_dataset_common_data
from d2v_core.verify import VerificationRunner

logger = logging.getLogger(__name__)

# Single-entry hash-map dispatch for model_vote strategies. Only
# "similarity_weighted" is supported today (config.yaml retrieval.model_vote),
# but this keeps the door open for future strategies without an if-else ladder.
MODEL_VOTE_DISPATCH: dict = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PHASE 3: retrieve a warm-start prior for a new test dataset "
        "from the trained Dataset2Vec encoder + FAISS-backed meta-knowledge store."
    )
    parser.add_argument("-c", "--config", required=True, type=str, help="path to config.ini")
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="path to the new test dataset's .npz file"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="output directory for dataset_prior.json"
    )
    parser.add_argument(
        "-k", "--top-k", required=False, type=int, default=None,
        help="override retrieval.top_k from config.yaml for the neighbor search",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="actually train every recommended model on the query dataset and "
        "compare achieved metrics against the warm-start prior's expected_metric",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args()


def load_encoder(store_dir: str, encoder_config: dict, n_classes_sample: int) -> Dataset2VecEncoder:
    """Reconstructs the trained encoder from <store_dir>/encoder/encoder.pt using the
    SAME architecture config.yaml's encoder: section that produced it (encoder_version.json
    only stores a metadata subset, not the architecture, so it cannot be used to rebuild
    the module graph)."""
    encoder_pt_path = os.path.join(store_dir, "encoder", "encoder.pt")
    if not os.path.isfile(encoder_pt_path):
        raise FileNotFoundError(
            f"=> encoder.pt not found at '{encoder_pt_path}'. Run train_encoder.py first."
        )
    encoder = Dataset2VecEncoder(encoder_config, n_classes_sample=n_classes_sample)
    encoder.load_state_dict(torch.load(encoder_pt_path, map_location="cpu"))
    encoder.eval()
    logger.info("=> loaded encoder state_dict from '%s'.", encoder_pt_path)
    return encoder


def load_encoder_version(store_dir: str) -> str:
    encoder_version_path = os.path.join(store_dir, "encoder", "encoder_version.json")
    if not os.path.isfile(encoder_version_path):
        raise FileNotFoundError(
            f"=> encoder_version.json not found at '{encoder_version_path}'. Run train_encoder.py first."
        )
    with open(encoder_version_path, "r") as version_file:
        version_payload = json.load(version_file)
    return str(version_payload["encoder_version"])


def embed_query_dataset(
    input_npz_path: str,
    encoder: Dataset2VecEncoder,
    training_config: dict,
) -> tuple[np.ndarray, str, str]:
    """Copies the single query .npz into a fresh temp directory and reuses
    CorpusSampler (instead of writing new npz-loading logic) to embed it. Returns
    (embedding_vector, dataset_id, task_type)."""
    temp_corpus_dir = tempfile.mkdtemp(prefix="d2v_query_corpus_")
    dataset_id = os.path.splitext(os.path.basename(input_npz_path))[0]
    temp_npz_path = os.path.join(temp_corpus_dir, f"{dataset_id}.npz")
    with open(input_npz_path, "rb") as source_file, open(temp_npz_path, "wb") as dest_file:
        dest_file.write(source_file.read())

    sampler = CorpusSampler(
        corpus_dir=temp_corpus_dir,
        n_instances_sample=training_config["n_instances_sample"],
        n_features_sample=training_config["n_features_sample"],
        n_classes_sample=training_config["n_classes_sample"],
        random_state=training_config["random_state"],
    )
    embedding_generator = EmbeddingGenerator(encoder, device=torch.device("cpu"))
    embedding_tensor = embedding_generator.encode_dataset(sampler, dataset_id)
    embedding_vector = embedding_tensor.cpu().numpy()
    task_type = sampler.corpus[dataset_id].task_type

    os.remove(temp_npz_path)
    os.rmdir(temp_corpus_dir)
    return embedding_vector, dataset_id, task_type


def write_cold_start_prior(
    output_dir: str,
    dataset_id: str,
    encoder_version: str,
    top_k: int,
    primary_metric: str,
    caveat_message: str,
) -> str:
    """Builds and writes a cold-start DatasetPrior (empty neighbors/ranked_models)
    via an explicit early return rather than raising, per the project's
    early-return-over-try/except convention for control flow."""
    cold_start_prior = DatasetPrior(
        query_dataset_id=dataset_id,
        encoder_version=encoder_version,
        top_k=top_k,
        primary_metric=primary_metric,
        neighbors=[],
        ranked_models=[],
        verification_summary=None,
        cold_start=True,
        caveats=[caveat_message],
    )
    output_path = os.path.join(output_dir, "dataset_prior.json")
    with open(output_path, "w") as output_file:
        output_file.write(cold_start_prior.model_dump_json(indent=2))
    logger.warning("=> cold start: %s. Wrote '%s'.", caveat_message, output_path)
    return output_path


def build_neighbor_results(
    neighbor_search_results: list[tuple[str, float]],
    meta_kb_df: pd.DataFrame,
) -> list[NeighborResult]:
    """For each (dataset_id, similarity) pair, looks up its meta_kb row and builds
    a NeighborResult from the TOP-RANKED (rank == 1, i.e. leaderboard[0]) entry."""
    neighbor_results: list[NeighborResult] = []
    for dataset_id, similarity_score in neighbor_search_results:
        matching_rows = meta_kb_df[meta_kb_df["dataset_id"] == dataset_id]
        neighbor_row = matching_rows.iloc[0]
        top_leaderboard_entry = neighbor_row["leaderboard"][0]
        neighbor_results.append(
            NeighborResult(
                dataset_id=dataset_id,
                similarity=similarity_score,
                best_model=neighbor_row["best_model"],
                recommended_hyperparameters=top_leaderboard_entry["hyperparameters"],
                metrics=top_leaderboard_entry["metrics"],
            )
        )
    return neighbor_results


def similarity_weighted_vote(
    neighbor_search_results: list[tuple[str, float]],
    meta_kb_df: pd.DataFrame,
    primary_metric: str,
    n_recommended_models: int,
) -> list[RankedModelEntry]:
    """Similarity-weighted vote across ALL leaderboard entries of ALL neighbors:
    score[model_name] = sum(similarity * metric) / sum(similarity). The
    recommended_hyperparameters/expected_metric for a model_name come from the
    SINGLE highest-similarity neighbor that used that model (not blended across
    heterogeneous hyperparameter configs)."""
    vote_score_by_model: dict[str, float] = {}
    weight_sum_by_model: dict[str, float] = {}
    best_similarity_by_model: dict[str, float] = {}
    best_record_by_model: dict[str, dict] = {}

    for dataset_id, similarity_score in neighbor_search_results:
        matching_rows = meta_kb_df[meta_kb_df["dataset_id"] == dataset_id]
        neighbor_row = matching_rows.iloc[0]
        neighbor_primary_metric = primary_metric if primary_metric else neighbor_row["primary_metric"]

        for leaderboard_entry in neighbor_row["leaderboard"]:
            model_name = leaderboard_entry["model_name"]
            metric_value = leaderboard_entry["metrics"].get(neighbor_primary_metric)
            if metric_value is None:
                continue

            vote_score_by_model[model_name] = (
                vote_score_by_model.get(model_name, 0.0) + similarity_score * metric_value
            )
            weight_sum_by_model[model_name] = (
                weight_sum_by_model.get(model_name, 0.0) + similarity_score
            )

            if similarity_score > best_similarity_by_model.get(model_name, -float("inf")):
                best_similarity_by_model[model_name] = similarity_score
                best_record_by_model[model_name] = {
                    "recommended_hyperparameters": leaderboard_entry["hyperparameters"],
                    "expected_metric": metric_value,
                }

    ranked_models: list[RankedModelEntry] = []
    for model_name, vote_score in vote_score_by_model.items():
        final_score = vote_score / weight_sum_by_model[model_name]
        best_record = best_record_by_model[model_name]
        ranked_models.append(
            RankedModelEntry(
                model_name=model_name,
                score=final_score,
                recommended_hyperparameters=best_record["recommended_hyperparameters"],
                expected_metric=best_record["expected_metric"],
                verification=None,
            )
        )

    ranked_models.sort(key=lambda entry: entry.score, reverse=True)
    return ranked_models[:n_recommended_models]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    os.makedirs(args.output, exist_ok=True)

    encoder_config = load_yaml_config(args.config, "encoder")
    training_config = load_yaml_config(args.config, "training")
    store_config = load_yaml_config(args.config, "store")
    retrieval_config = load_yaml_config(args.config, "retrieval")

    top_k = args.top_k if args.top_k is not None else retrieval_config["top_k"]
    same_task_only = retrieval_config["same_task_only"]
    n_recommended_models = max(retrieval_config["n_recommended_models"], 5)
    configured_primary_metric = retrieval_config.get("primary_metric")
    model_vote_strategy = retrieval_config["model_vote"]
    MODEL_VOTE_DISPATCH["similarity_weighted"] = similarity_weighted_vote
    if model_vote_strategy not in MODEL_VOTE_DISPATCH:
        raise ValueError(
            f"=> retrieval.model_vote='{model_vote_strategy}' is not supported. "
            f"Supported strategies: {list(MODEL_VOTE_DISPATCH.keys())}"
        )
    vote_fn = MODEL_VOTE_DISPATCH[model_vote_strategy]

    store_dir = resolve_store_dir(args.config)
    store = MetaKnowledgeStore(
        store_dir=store_dir,
        faiss_metric=store_config["faiss_metric"],
        normalize_embeddings=store_config["normalize_embeddings"],
    )

    query_dataset_id = os.path.splitext(os.path.basename(args.input))[0]

    # Cold-start guard #1: the store has no data at all yet.
    if store.is_empty():
        write_cold_start_prior(
            output_dir=args.output,
            dataset_id=query_dataset_id,
            encoder_version=encoder_config["encoder_version"],
            top_k=top_k,
            primary_metric=configured_primary_metric if configured_primary_metric else "unknown",
            caveat_message="meta-knowledge store is empty (no trained datasets yet); "
            "cannot produce a warm-start prior.",
        )
        return

    encoder = load_encoder(
        store_dir, encoder_config, n_classes_sample=training_config["n_classes_sample"]
    )
    encoder_version = load_encoder_version(store_dir)

    query_embedding, embedded_dataset_id, query_task_type = embed_query_dataset(
        args.input, encoder, training_config
    )
    logger.info(
        "=> embedded query dataset '%s' (task_type='%s').", embedded_dataset_id, query_task_type
    )

    neighbor_search_results = store.search(
        query_embedding, top_k=top_k, task_type=query_task_type, same_task_only=same_task_only
    )

    # Cold-start guard #2: store has data, but none matches this task_type (or
    # search otherwise returned nothing).
    if len(neighbor_search_results) == 0:
        write_cold_start_prior(
            output_dir=args.output,
            dataset_id=embedded_dataset_id,
            encoder_version=encoder_version,
            top_k=top_k,
            primary_metric=configured_primary_metric if configured_primary_metric else "unknown",
            caveat_message=f"no neighbors found in the store for task_type='{query_task_type}' "
            f"(same_task_only={same_task_only}); store has data but none matches this query.",
        )
        return

    meta_kb_df = store.load_meta_kb()
    neighbor_results = build_neighbor_results(neighbor_search_results, meta_kb_df)

    # primary_metric resolution: prefer config.yaml's retrieval.primary_metric;
    # fall back to the top neighbor's own primary_metric field if not configured.
    if configured_primary_metric:
        primary_metric = configured_primary_metric
    else:
        top_neighbor_dataset_id = neighbor_search_results[0][0]
        primary_metric = meta_kb_df[
            meta_kb_df["dataset_id"] == top_neighbor_dataset_id
        ].iloc[0]["primary_metric"]

    ranked_models = vote_fn(
        neighbor_search_results, meta_kb_df, primary_metric, n_recommended_models
    )

    for ranked_entry in ranked_models:
        # Sanity assertion, not a silent filter: a validate_model_name failure here
        # means the leaderboard data is corrupted and must raise loudly.
        validate_model_name(ranked_entry.model_name, training_mode="fine_tune")

    caveats: list[str] = []
    if len(neighbor_search_results) < top_k:
        caveats.append(
            f"only {len(neighbor_search_results)} neighbor(s) found, fewer than requested top_k={top_k}."
        )

    verification_summary = None
    if args.verify:
        verification_common_data, verification_task_type = load_dataset_common_data(
            corpus_dir=os.path.dirname(args.input), dataset_id=embedded_dataset_id
        )
        verification_runner = VerificationRunner(
            common=verification_common_data,
            task_type=verification_task_type,
            primary_metric=primary_metric,
            tolerance=retrieval_config["verify_tolerance"],
        )
        verification_summary = verification_runner.run(ranked_models)
        logger.info(
            "=> verification finished: %d/%d model(s) within tolerance=%.4f, mean_abs_delta=%s.",
            verification_summary.n_within_tolerance,
            verification_summary.n_verified,
            verification_summary.tolerance,
            verification_summary.mean_abs_delta,
        )

    dataset_prior = DatasetPrior(
        query_dataset_id=embedded_dataset_id,
        encoder_version=encoder_version,
        top_k=top_k,
        primary_metric=primary_metric,
        neighbors=neighbor_results,
        ranked_models=ranked_models,
        verification_summary=verification_summary,
        cold_start=False,
        caveats=caveats,
    )

    output_path = os.path.join(args.output, "dataset_prior.json")
    with open(output_path, "w") as output_file:
        output_file.write(dataset_prior.model_dump_json(indent=2))
    logger.info(
        "=> wrote dataset_prior.json to '%s' (%d neighbors, %d ranked models).",
        output_path,
        len(neighbor_results),
        len(ranked_models),
    )


if __name__ == "__main__":
    main()
