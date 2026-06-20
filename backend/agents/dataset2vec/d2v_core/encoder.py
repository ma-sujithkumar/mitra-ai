import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from d2v_core.sampling import CorpusSampler, DatasetPatch

logger = logging.getLogger(__name__)


class ResidualMLPBlock(nn.Module):
    """input_projection -> N residual units (each a stack of num_layers_per_unit
    Linear+ReLU layers with a residual skip) -> output_projection. Setting
    num_residual_units=0 collapses this to a plain input_projection ->
    output_projection MLP, used for the h-block which has no residual_blocks
    entry in config.yaml."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers_per_unit: int,
        num_residual_units: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.residual_units = nn.ModuleList(
            [
                self._build_residual_unit(hidden_dim, num_layers_per_unit)
                for _ in range(num_residual_units)
            ]
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def _build_residual_unit(self, hidden_dim: int, num_layers_per_unit: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(num_layers_per_unit):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_state = self.activation(self.input_projection(input_tensor))
        for residual_unit in self.residual_units:
            hidden_state = hidden_state + residual_unit(hidden_state)
        return self.output_projection(hidden_state)


class Dataset2VecEncoder(nn.Module):
    """Permutation-invariant hierarchical set-network: f_net embeds every
    (instance, feature) cell -> mean-pool over instances -> g_net embeds every
    per-feature representation -> mean-pool over features -> h_net produces the
    final fixed-length dataset embedding. Output length is always
    encoder_config['embedding_dim'] regardless of the input patch's
    (n_instances, n_features) shape."""

    def __init__(self, encoder_config: dict, n_classes_sample: int) -> None:
        super().__init__()
        self.n_classes_sample = n_classes_sample
        cell_input_dim = 1 + n_classes_sample

        f_block_config = encoder_config["f_block"]
        g_block_config = encoder_config["g_block"]
        h_block_config = encoder_config["h_block"]
        embedding_dim = encoder_config["embedding_dim"]

        self.f_net = ResidualMLPBlock(
            input_dim=cell_input_dim,
            hidden_dim=f_block_config["hidden"],
            num_layers_per_unit=f_block_config["layers"],
            num_residual_units=f_block_config["residual_blocks"],
            output_dim=f_block_config["hidden"],
        )
        self.g_net = ResidualMLPBlock(
            input_dim=f_block_config["hidden"],
            hidden_dim=g_block_config["hidden"],
            num_layers_per_unit=g_block_config["layers"],
            num_residual_units=g_block_config["residual_blocks"],
            output_dim=g_block_config["hidden"],
        )
        self.h_net = ResidualMLPBlock(
            input_dim=g_block_config["hidden"],
            hidden_dim=h_block_config["hidden"],
            num_layers_per_unit=h_block_config["layers"],
            num_residual_units=h_block_config.get("residual_blocks", 0),
            output_dim=embedding_dim,
        )

    def forward(self, patch: DatasetPatch, device: torch.device) -> torch.Tensor:
        feature_matrix = torch.as_tensor(
            patch.feature_matrix, dtype=torch.float32, device=device
        )
        target_repr = torch.as_tensor(
            patch.target_repr, dtype=torch.float32, device=device
        )
        n_instances, n_features = feature_matrix.shape

        feature_values = feature_matrix.unsqueeze(-1)  # (n_instances, n_features, 1)
        target_broadcast = target_repr.unsqueeze(1).expand(
            n_instances, n_features, self.n_classes_sample
        )
        cell_inputs = torch.cat([feature_values, target_broadcast], dim=-1)
        cell_inputs_flat = cell_inputs.reshape(n_instances * n_features, -1)

        cell_embeddings = self.f_net(cell_inputs_flat).reshape(n_instances, n_features, -1)
        per_feature_embeddings = cell_embeddings.mean(dim=0)  # (n_features, f_out_dim)

        per_feature_embeddings = self.g_net(per_feature_embeddings)
        dataset_embedding = per_feature_embeddings.mean(dim=0)  # (g_out_dim,)

        return self.h_net(dataset_embedding.unsqueeze(0)).squeeze(0)


class ContrastivePairLoss(nn.Module):
    """similarity = exp(-gamma * L2(embedding_a, embedding_b)); BCE against the
    pair label (1 = same dataset, 0 = different datasets). Same-dataset patches
    are pulled together, cross-dataset patches are pushed apart."""

    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma
        self.bce_loss = nn.BCELoss()

    def forward(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        l2_distance = torch.norm(embedding_a - embedding_b, p=2)
        similarity = torch.exp(-self.gamma * l2_distance).clamp(min=1e-6, max=1.0 - 1e-6)
        loss = self.bce_loss(similarity, label)
        return loss, similarity


class EncoderTrainer:
    """Owns the optimizer, loss, and checkpoint/resume lifecycle for contrastive
    pre-training. One epoch = one sampled batch of pairs_per_batch (patch_a,
    patch_b, label) triples; patches have varying (n, m) shapes so the batch is
    looped rather than padded into one tensor."""

    CHECKPOINT_FILENAME = "checkpoint.pt"

    def __init__(
        self,
        encoder: Dataset2VecEncoder,
        sampler: CorpusSampler,
        training_config: dict,
        checkpoint_dir: str,
    ) -> None:
        self.encoder = encoder
        self.sampler = sampler
        self.training_config = training_config
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        requested_device = training_config["device"]
        self.device = torch.device(
            requested_device if (requested_device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.encoder.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=training_config["learning_rate"]
        )
        self.loss_fn = ContrastivePairLoss(training_config["gamma"])

        self.start_epoch = 0
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    def run_epoch(self) -> float:
        pairs = self.sampler.sample_pair_batch(self.training_config["pairs_per_batch"])
        self.optimizer.zero_grad()
        total_loss = 0.0
        for patch_a, patch_b, label in pairs:
            embedding_a = self.encoder(patch_a, self.device)
            embedding_b = self.encoder(patch_b, self.device)
            label_tensor = torch.tensor(float(label), device=self.device)
            pair_loss, _ = self.loss_fn(embedding_a, embedding_b, label_tensor)
            pair_loss = pair_loss / len(pairs)
            pair_loss.backward()
            total_loss += pair_loss.item()
        self.optimizer.step()
        return total_loss

    def train(self, max_epochs: Optional[int] = None) -> dict:
        target_epochs = max_epochs if max_epochs is not None else self.training_config["epochs"]
        es_patience = self.training_config["es_patience"]
        es_min_delta = self.training_config["es_min_delta"]
        checkpoint_every = self.training_config["checkpoint_every"]

        final_epoch = self.start_epoch
        for epoch in range(self.start_epoch, target_epochs):
            epoch_loss = self.run_epoch()
            final_epoch = epoch + 1
            logger.info("=> epoch %d: loss=%.6f", final_epoch, epoch_loss)

            if epoch_loss < self.best_loss - es_min_delta:
                self.best_loss = epoch_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if final_epoch % checkpoint_every == 0:
                self.save_checkpoint(final_epoch)

            if self.epochs_without_improvement >= es_patience:
                logger.info(
                    "=> early stopping at epoch %d (no improvement for %d epochs).",
                    final_epoch,
                    es_patience,
                )
                break

        self.save_checkpoint(final_epoch)
        return {"final_epoch": final_epoch, "best_loss": self.best_loss}

    def save_checkpoint(self, epoch: int) -> str:
        checkpoint_path = os.path.join(self.checkpoint_dir, self.CHECKPOINT_FILENAME)
        torch.save(
            {
                "epoch": epoch,
                "encoder_state": self.encoder.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "epochs_without_improvement": self.epochs_without_improvement,
            },
            checkpoint_path,
        )
        logger.info("=> saved checkpoint at epoch %d to '%s'.", epoch, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.start_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.epochs_without_improvement = checkpoint["epochs_without_improvement"]
        logger.info("=> resumed from checkpoint at epoch %d.", self.start_epoch)


class EmbeddingGenerator:
    """Wraps a trained Dataset2VecEncoder to produce one averaged embedding per
    dataset (mean over several sampled patches, to reduce sampling variance)."""

    def __init__(self, encoder: Dataset2VecEncoder, device: torch.device, n_patches_per_dataset: int = 10) -> None:
        self.encoder = encoder
        self.device = device
        self.n_patches_per_dataset = n_patches_per_dataset
        self.encoder.eval()

    def encode_dataset(self, sampler: CorpusSampler, dataset_id: str) -> torch.Tensor:
        with torch.no_grad():
            patch_embeddings = [
                self.encoder(sampler.sample_patch(dataset_id), self.device)
                for _ in range(self.n_patches_per_dataset)
            ]
            return torch.stack(patch_embeddings, dim=0).mean(dim=0)
