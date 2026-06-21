# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
import logging
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from core.data_bundle import DataBundle
from models.base import BaseModel


logger = logging.getLogger(__name__)


ACTIVATION_MAP: dict = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}


def _build_fc_block(
    input_size: int,
    output_size: int,
    activation_name: str,
    dropout_rate: float,
) -> List[nn.Module]:
    """Returns [Linear, Activation, Dropout] layers for one FC block."""
    activation_class = ACTIVATION_MAP.get(activation_name, nn.ReLU)
    return [
        nn.Linear(input_size, output_size),
        activation_class(),
        nn.Dropout(p=dropout_rate),
    ]


def _build_fcnn(
    input_size: int,
    hidden_layers: List[int],
    num_outputs: int,
    activation_name: str,
    dropout_rate: float,
) -> nn.Sequential:
    """Build a fully-connected feedforward network from config."""
    layer_sizes = [input_size] + hidden_layers
    layers: List[nn.Module] = []
    for layer_index in range(len(layer_sizes) - 1):
        layers.extend(
            _build_fc_block(
                layer_sizes[layer_index],
                layer_sizes[layer_index + 1],
                activation_name,
                dropout_rate,
            )
        )
    layers.append(nn.Linear(layer_sizes[-1], num_outputs))
    return nn.Sequential(*layers)


class PyTorchFCNNClassifierWrapper(BaseModel):
    """Fully-connected neural network classifier built from config.yaml architecture."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder: LabelEncoder = LabelEncoder()

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)

        X_train = data.common.X_train.astype(np.float32)
        # Remap arbitrary class labels to consecutive integers 0..N-1.
        # CrossEntropyLoss requires target indices in [0, num_classes); without this
        # remapping a dataset with labels {0,1,3,...,14} causes "Target out of bounds".
        y_train = self.label_encoder.fit_transform(data.common.y_train).astype(np.int64)

        num_features = X_train.shape[1]
        # Derive num_classes from actual data so the model output layer matches the
        # dataset even when the config default (e.g. 10) doesn't match class count.
        num_classes = len(self.label_encoder.classes_)
        hidden_layers: List[int] = list(merged_config["hidden_layers"])
        activation_name: str = merged_config["activation"]
        dropout_rate: float = float(merged_config["dropout"])
        learning_rate: float = float(merged_config["learning_rate"])
        num_epochs: int = int(merged_config["epochs"])
        batch_size: int = int(merged_config["batch_size"])
        weight_decay: float = float(merged_config.get("weight_decay", 1e-4))

        self.model = _build_fcnn(
            num_features, hidden_layers, num_classes, activation_name, dropout_rate
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        X_tensor = torch.tensor(X_train)
        y_tensor = torch.tensor(y_train)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch_index in range(num_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            logger.debug(
                "=> PyTorchFCNNClassifier epoch %d/%d loss=%.4f",
                epoch_index + 1,
                num_epochs,
                epoch_loss / len(loader),
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            encoded_predictions = logits.argmax(dim=1).cpu().numpy()
        # Decode back to original class labels.
        return self.label_encoder.inverse_transform(encoded_predictions)

    def _prepare_for_serialization(self) -> nn.Module:
        return self.model.cpu()


class _CNNClassifierNet(nn.Module):
    """Internal CNN network for classification. Supports Conv1d and Conv2d."""

    def __init__(
        self,
        conv_dim: int,
        in_channels: int,
        conv_channels: List[int],
        kernel_size: int,
        fc_input_size: int,
        fc_layers: List[int],
        num_classes: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        conv_layer_class = nn.Conv2d if conv_dim == 2 else nn.Conv1d
        pool_layer_class = nn.MaxPool2d if conv_dim == 2 else nn.MaxPool1d

        conv_blocks: List[nn.Module] = []
        current_channels = in_channels
        for out_channels in conv_channels:
            conv_blocks += [
                conv_layer_class(current_channels, out_channels, kernel_size, padding=1),
                nn.ReLU(),
                pool_layer_class(2),
            ]
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_blocks)

        fc_sizes = [fc_input_size] + fc_layers
        fc_blocks: List[nn.Module] = [nn.Flatten()]
        for layer_index in range(len(fc_sizes) - 1):
            fc_blocks += [
                nn.Linear(fc_sizes[layer_index], fc_sizes[layer_index + 1]),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ]
        fc_blocks.append(nn.Linear(fc_sizes[-1], num_classes))
        self.fc_layers = nn.Sequential(*fc_blocks)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(input_tensor)
        return self.fc_layers(features)


def _compute_cnn_fc_input_size(
    conv_dim: int,
    input_spatial_size: int,
    conv_channels: List[int],
    kernel_size: int,
    num_conv_layers: int,
) -> int:
    """Compute the flattened size after all conv+pool layers."""
    spatial_size = input_spatial_size
    for _ in range(num_conv_layers):
        # conv with padding=1 preserves size; pool with kernel=2 halves it
        spatial_size = spatial_size // 2

    last_channels = conv_channels[-1]
    if conv_dim == 2:
        return last_channels * spatial_size * spatial_size
    return last_channels * spatial_size


class PyTorchCNNClassifierWrapper(BaseModel):
    """CNN classifier. conv_dim=2 for image input (MNIST), conv_dim=1 for 1D signals."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _reshape_for_cnn(
        self, X: np.ndarray, conv_dim: int
    ) -> np.ndarray:
        """Reshape flat (N, features) input into CNN-compatible shape."""
        num_samples = X.shape[0]
        num_features = X.shape[1]

        if conv_dim == 2:
            # Infer spatial dimension from sqrt of feature count
            spatial_side = int(math.isqrt(num_features))
            if spatial_side * spatial_side != num_features:
                # Pad to next perfect square
                next_square = (spatial_side + 1) ** 2
                X = np.pad(X, ((0, 0), (0, next_square - num_features)))
                spatial_side = spatial_side + 1
                logger.debug(
                    "=> PyTorchCNNClassifier: padded features to %d for 2D reshape.",
                    next_square,
                )
            return X.reshape(num_samples, 1, spatial_side, spatial_side)
        else:
            # Conv1d: treat each feature as one time step
            return X.reshape(num_samples, 1, num_features)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)

        X_train = data.common.X_train.astype(np.float32)
        y_train = data.common.y_train.astype(np.int64)

        conv_dim: int = int(merged_config["conv_dim"])
        conv_channels: List[int] = list(merged_config["conv_channels"])
        kernel_size: int = int(merged_config["kernel_size"])
        fc_layers: List[int] = list(merged_config["fc_layers"])
        num_classes: int = int(merged_config["num_classes"])
        dropout_rate: float = float(merged_config["dropout"])
        learning_rate: float = float(merged_config["learning_rate"])
        num_epochs: int = int(merged_config["epochs"])
        batch_size: int = int(merged_config["batch_size"])
        weight_decay: float = float(merged_config.get("weight_decay", 1e-4))

        X_reshaped = self._reshape_for_cnn(X_train, conv_dim)
        spatial_size = X_reshaped.shape[2]  # height (or length for 1D)

        fc_input_size = _compute_cnn_fc_input_size(
            conv_dim, spatial_size, conv_channels, kernel_size, len(conv_channels)
        )

        self.model = _CNNClassifierNet(
            conv_dim=conv_dim,
            in_channels=1,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            fc_input_size=fc_input_size,
            fc_layers=fc_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        X_tensor = torch.tensor(X_reshaped)
        y_tensor = torch.tensor(y_train)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch_index in range(num_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            logger.debug(
                "=> PyTorchCNNClassifier epoch %d/%d loss=%.4f",
                epoch_index + 1,
                num_epochs,
                epoch_loss / len(loader),
            )

        # Store conv_dim so predict() can apply the same reshape
        self.trained_conv_dim = conv_dim

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_reshaped = self._reshape_for_cnn(X.astype(np.float32), self.trained_conv_dim)
        X_tensor = torch.tensor(X_reshaped).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            predicted_classes = logits.argmax(dim=1).cpu().numpy()
        return predicted_classes

    def _prepare_for_serialization(self) -> nn.Module:
        return self.model.cpu()
