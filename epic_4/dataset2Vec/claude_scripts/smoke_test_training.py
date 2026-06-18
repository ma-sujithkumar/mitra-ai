import logging
import os
import shutil
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from d2v_core.encoder import Dataset2VecEncoder, EncoderTrainer
from d2v_core.sampling import CorpusSampler

logging.basicConfig(level=logging.INFO, format="%(message)s")


def average_similarity(encoder: Dataset2VecEncoder, sampler: CorpusSampler, device: torch.device, dataset_id_a: str, dataset_id_b: str, n_samples: int = 8) -> float:
    distances = []
    with torch.no_grad():
        for _ in range(n_samples):
            embedding_a = encoder(sampler.sample_patch(dataset_id_a), device)
            embedding_b = encoder(sampler.sample_patch(dataset_id_b), device)
            distances.append(torch.norm(embedding_a - embedding_b, p=2).item())
    mean_distance = sum(distances) / len(distances)
    return mean_distance


def main() -> None:
    tool_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(tool_root, "config", "config.yaml"), "r") as yaml_file:
        full_config = yaml.safe_load(yaml_file)

    encoder_config = full_config["encoder"]
    training_config = dict(full_config["training"])
    training_config["device"] = "cpu"
    training_config["n_instances_sample"] = 64
    training_config["n_features_sample"] = 8
    training_config["pairs_per_batch"] = 12
    training_config["checkpoint_every"] = 5
    training_config["es_patience"] = 1000

    corpus_dir = os.path.join(os.path.dirname(__file__), "toy_corpus")
    sampler = CorpusSampler(
        corpus_dir=corpus_dir,
        n_instances_sample=training_config["n_instances_sample"],
        n_features_sample=training_config["n_features_sample"],
        n_classes_sample=training_config["n_classes_sample"],
        random_state=training_config["random_state"],
    )

    encoder = Dataset2VecEncoder(encoder_config, n_classes_sample=training_config["n_classes_sample"])
    checkpoint_dir = tempfile.mkdtemp(prefix="encoder_ckpt_smoke_")
    trainer = EncoderTrainer(encoder, sampler, training_config, checkpoint_dir)

    train_result = trainer.train(max_epochs=40)
    print(f"=> training result: {train_result}")

    device = torch.device("cpu")
    encoder.eval()
    same_dataset_distance = average_similarity(encoder, sampler, device, "iris", "iris")
    cross_dataset_distance = average_similarity(encoder, sampler, device, "iris", "diabetes")
    print(f"=> mean L2 distance same-dataset (iris,iris)={same_dataset_distance:.4f}")
    print(f"=> mean L2 distance cross-dataset (iris,diabetes)={cross_dataset_distance:.4f}")
    assert same_dataset_distance < cross_dataset_distance, (
        same_dataset_distance, cross_dataset_distance
    )

    checkpoint_path = os.path.join(checkpoint_dir, EncoderTrainer.CHECKPOINT_FILENAME)
    assert os.path.isfile(checkpoint_path)

    resumed_encoder = Dataset2VecEncoder(encoder_config, n_classes_sample=training_config["n_classes_sample"])
    resumed_trainer = EncoderTrainer(resumed_encoder, sampler, training_config, checkpoint_dir)
    resumed_trainer.load_checkpoint(checkpoint_path)
    assert resumed_trainer.start_epoch == train_result["final_epoch"] or resumed_trainer.start_epoch % training_config["checkpoint_every"] == 0
    for original_param, resumed_param in zip(encoder.parameters(), resumed_encoder.parameters()):
        assert torch.allclose(original_param, resumed_param), "checkpoint round-trip mismatch"

    shutil.rmtree(checkpoint_dir)
    print("=> smoke test passed: same-dataset similarity > cross-dataset, checkpoint round-trip OK.")


if __name__ == "__main__":
    main()
