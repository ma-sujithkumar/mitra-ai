import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from d2v_core.encoder import Dataset2VecEncoder
from d2v_core.sampling import CorpusSampler

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    tool_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(tool_root, "config", "config.yaml"), "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    encoder_config = full_config["encoder"]
    training_config = full_config["training"]
    n_classes_sample = training_config["n_classes_sample"]

    corpus_dir = os.path.join(os.path.dirname(__file__), "toy_corpus")
    sampler = CorpusSampler(
        corpus_dir=corpus_dir,
        n_instances_sample=training_config["n_instances_sample"],
        n_features_sample=training_config["n_features_sample"],
        n_classes_sample=n_classes_sample,
        random_state=training_config["random_state"],
    )
    assert set(sampler.dataset_ids) == {
        "iris", "wine", "breast_cancer", "diabetes", "synthetic_blob"
    }, sampler.dataset_ids

    device = torch.device("cpu")
    encoder = Dataset2VecEncoder(encoder_config, n_classes_sample=n_classes_sample).to(device)
    encoder.eval()

    embedding_dim = encoder_config["embedding_dim"]
    with torch.no_grad():
        for dataset_id in sampler.dataset_ids:
            patch = sampler.sample_patch(dataset_id)
            embedding = encoder(patch, device)
            assert embedding.shape == (embedding_dim,), (dataset_id, embedding.shape)
            assert torch.isfinite(embedding).all(), dataset_id
            print(
                f"=> {dataset_id}: patch shape={patch.feature_matrix.shape}, "
                f"task_type={patch.task_type}, embedding_dim={embedding.shape[0]}"
            )

        pairs = sampler.sample_pair_batch(pairs_per_batch=8)
        assert len(pairs) == 8
        for patch_a, patch_b, label in pairs:
            embedding_a = encoder(patch_a, device)
            embedding_b = encoder(patch_b, device)
            assert embedding_a.shape == (embedding_dim,)
            assert embedding_b.shape == (embedding_dim,)
            assert label in (0, 1)

    print("=> smoke test passed: fixed-length embeddings across varying (n, m, t) shapes.")


if __name__ == "__main__":
    main()
