import argparse
import logging
import os
from datetime import datetime, timezone

import torch

from d2v_core.encoder import Dataset2VecEncoder, EmbeddingGenerator, EncoderTrainer
from d2v_core.sampling import CorpusSampler
from d2v_core.schema import load_yaml_config, resolve_store_dir
from d2v_core.store import MetaKnowledgeStore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PHASE 1: train the Dataset2Vec encoder and embed the training corpus."
    )
    parser.add_argument("-c", "--config", required=True, type=str, help="path to config.ini")
    parser.add_argument(
        "--resume", action="store_true", help="resume from the latest checkpoint if present"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    encoder_config = load_yaml_config(args.config, "encoder")
    training_config = load_yaml_config(args.config, "training")
    store_config = load_yaml_config(args.config, "store")

    corpus_dir = training_config["corpus_dir"]
    if corpus_dir is None:
        raise ValueError(
            "=> training.corpus_dir is null in config.yaml. Set it to a directory of "
            "*.npz training datasets before running train_encoder.py."
        )

    store_dir = resolve_store_dir(args.config)
    encoder_dir = os.path.join(store_dir, "encoder")
    checkpoint_dir = os.path.join(encoder_dir, "checkpoints")
    os.makedirs(encoder_dir, exist_ok=True)

    sampler = CorpusSampler(
        corpus_dir=corpus_dir,
        n_instances_sample=training_config["n_instances_sample"],
        n_features_sample=training_config["n_features_sample"],
        n_classes_sample=training_config["n_classes_sample"],
        random_state=training_config["random_state"],
    )

    encoder = Dataset2VecEncoder(
        encoder_config, n_classes_sample=training_config["n_classes_sample"]
    )
    trainer = EncoderTrainer(encoder, sampler, training_config, checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, EncoderTrainer.CHECKPOINT_FILENAME)
    if args.resume and os.path.isfile(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    elif args.resume:
        logger.warning("=> --resume requested but no checkpoint found at '%s'; starting fresh.", checkpoint_path)

    train_result = trainer.train()
    logger.info("=> training finished: %s", train_result)

    encoder_pt_path = os.path.join(encoder_dir, "encoder.pt")
    torch.save(encoder.state_dict(), encoder_pt_path)

    encoder_version = encoder_config["encoder_version"]
    encoder_version_path = os.path.join(encoder_dir, "encoder_version.json")
    import json

    with open(encoder_version_path, "w") as version_file:
        json.dump(
            {
                "encoder_version": encoder_version,
                "embedding_dim": encoder_config["embedding_dim"],
                "final_epoch": train_result["final_epoch"],
                "best_loss": train_result["best_loss"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            version_file,
            indent=2,
        )
    logger.info("=> saved '%s' and '%s'.", encoder_pt_path, encoder_version_path)

    embedding_generator = EmbeddingGenerator(encoder, device=trainer.device)
    embedding_rows = []
    for dataset_id in sampler.dataset_ids:
        embedding_vector = embedding_generator.encode_dataset(sampler, dataset_id).cpu().tolist()
        corpus_entry = sampler.corpus[dataset_id]
        embedding_rows.append(
            {
                "dataset_id": dataset_id,
                "encoder_version": encoder_version,
                "embedding": embedding_vector,
                "n_rows": corpus_entry.feature_matrix.shape[0],
                "n_cols": corpus_entry.feature_matrix.shape[1],
                "task_type": corpus_entry.task_type,
                "target_cardinality": corpus_entry.n_classes,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info("=> embedded dataset '%s'.", dataset_id)

    store = MetaKnowledgeStore(
        store_dir=store_dir,
        faiss_metric=store_config["faiss_metric"],
        normalize_embeddings=store_config["normalize_embeddings"],
    )
    store.write_embeddings(embedding_rows)
    joined_row_count = store.build_meta_kb()
    logger.info("=> build_meta_kb joined %d rows (0 expected until leaderboards.parquet exists too).", joined_row_count)


if __name__ == "__main__":
    main()
