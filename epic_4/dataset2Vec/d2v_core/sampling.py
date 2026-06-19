import glob
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

TASK_TYPE_CLASSIFICATION = "classification"
TASK_TYPE_REGRESSION = "regression"


@dataclass
class RawCorpusEntry:
    dataset_id: str
    feature_matrix: np.ndarray  # (n_rows, n_cols), already standardized per-column
    target_vector: np.ndarray  # (n_rows,)
    task_type: str
    n_classes: int  # number of distinct classes (classification) or 0 (regression)


@dataclass
class DatasetPatch:
    dataset_id: str
    task_type: str
    feature_matrix: np.ndarray  # (n_instances_sample, n_features_sample)
    target_repr: np.ndarray  # (n_instances_sample, n_classes_sample)


def infer_task_type(target_vector: np.ndarray) -> str:
    is_integer_valued = np.allclose(target_vector, np.round(target_vector))
    n_unique_values = len(np.unique(target_vector))
    if is_integer_valued and n_unique_values <= max(20, int(0.05 * len(target_vector))):
        return TASK_TYPE_CLASSIFICATION
    return TASK_TYPE_REGRESSION


def standardize_columns(feature_matrix: np.ndarray) -> np.ndarray:
    column_means = feature_matrix.mean(axis=0, keepdims=True)
    column_stds = feature_matrix.std(axis=0, keepdims=True)
    column_stds[column_stds < 1e-8] = 1.0
    return (feature_matrix - column_means) / column_stds


class CorpusSampler:
    """Loads a corpus of *.npz files (each with X_train/y_train[/X_test/y_test])
    and samples random "patches" -- a fixed-shape (n_instances_sample x
    n_features_sample) slice of one dataset plus its target representation --
    used both for contrastive pre-training pairs and for final whole-corpus
    embedding generation."""

    def __init__(
        self,
        corpus_dir: str,
        n_instances_sample: int,
        n_features_sample: int,
        n_classes_sample: int,
        random_state: int,
    ) -> None:
        if corpus_dir is None:
            raise ValueError(
                "=> corpus_dir is required (got None). Set training.corpus_dir / "
                "sweep.corpus_dir in config.yaml before running this command."
            )
        if not os.path.isdir(corpus_dir):
            raise FileNotFoundError(f"=> corpus_dir '{corpus_dir}' does not exist.")

        self.n_instances_sample = n_instances_sample
        self.n_features_sample = n_features_sample
        self.n_classes_sample = n_classes_sample
        self.random_generator = np.random.RandomState(random_state)
        self.corpus: dict[str, RawCorpusEntry] = self._load_corpus(corpus_dir)
        if len(self.corpus) == 0:
            raise ValueError(f"=> no *.npz datasets found under corpus_dir '{corpus_dir}'.")
        logger.info("=> loaded %d datasets from corpus_dir '%s'.", len(self.corpus), corpus_dir)

    def _load_corpus(self, corpus_dir: str) -> dict[str, RawCorpusEntry]:
        corpus: dict[str, RawCorpusEntry] = {}
        for npz_path in sorted(glob.glob(os.path.join(corpus_dir, "*.npz"))):
            dataset_id = os.path.splitext(os.path.basename(npz_path))[0]
            npz_data = np.load(npz_path, allow_pickle=True)
            feature_matrix = np.asarray(npz_data["X_train"], dtype=np.float64)
            target_vector = np.asarray(npz_data["y_train"], dtype=np.float64).reshape(-1)
            feature_matrix = standardize_columns(feature_matrix)

            if "task_type" in npz_data:
                task_type = str(npz_data["task_type"])
            else:
                task_type = infer_task_type(target_vector)

            n_classes = len(np.unique(target_vector)) if task_type == TASK_TYPE_CLASSIFICATION else 0
            corpus[dataset_id] = RawCorpusEntry(
                dataset_id=dataset_id,
                feature_matrix=feature_matrix,
                target_vector=target_vector,
                task_type=task_type,
                n_classes=n_classes,
            )
        return corpus

    @property
    def dataset_ids(self) -> list[str]:
        return list(self.corpus.keys())

    def _build_target_repr(self, entry: RawCorpusEntry, row_indices: np.ndarray) -> np.ndarray:
        """Always returns shape (len(row_indices), n_classes_sample): one-hot
        (padded/truncated to n_classes_sample) for classification, or the
        standardized target value in slot 0 with the remaining slots zeroed for
        regression. This keeps the encoder's per-cell input dimension fixed
        regardless of task type, which is required since one f_net module is
        shared across all patches."""
        sampled_targets = entry.target_vector[row_indices]
        target_repr = np.zeros((len(row_indices), self.n_classes_sample), dtype=np.float64)

        if entry.task_type == TASK_TYPE_CLASSIFICATION:
            class_labels = np.unique(entry.target_vector)
            label_to_index = {label: index for index, label in enumerate(class_labels)}
            for row_position, target_value in enumerate(sampled_targets):
                class_index = label_to_index[target_value]
                if class_index < self.n_classes_sample:
                    target_repr[row_position, class_index] = 1.0
        else:
            target_mean = entry.target_vector.mean()
            target_std = entry.target_vector.std()
            target_std = target_std if target_std > 1e-8 else 1.0
            target_repr[:, 0] = (sampled_targets - target_mean) / target_std

        return target_repr

    def sample_patch(self, dataset_id: str) -> DatasetPatch:
        entry = self.corpus[dataset_id]
        n_rows, n_cols = entry.feature_matrix.shape

        row_indices = self.random_generator.choice(
            n_rows, size=min(self.n_instances_sample, n_rows), replace=n_rows < self.n_instances_sample
        )
        col_indices = self.random_generator.choice(
            n_cols, size=min(self.n_features_sample, n_cols), replace=n_cols < self.n_features_sample
        )

        sampled_features = entry.feature_matrix[np.ix_(row_indices, col_indices)]
        target_repr = self._build_target_repr(entry, row_indices)

        return DatasetPatch(
            dataset_id=dataset_id,
            task_type=entry.task_type,
            feature_matrix=sampled_features,
            target_repr=target_repr,
        )

    def sample_pair_batch(self, pairs_per_batch: int) -> list[tuple[DatasetPatch, DatasetPatch, int]]:
        """Samples `pairs_per_batch` (patch_a, patch_b, label) triples for
        contrastive training: label=1 if both patches come from the same
        dataset (positive pair), label=0 otherwise (negative pair). Roughly half
        positive, half negative."""
        dataset_ids = self.dataset_ids
        if len(dataset_ids) < 2:
            raise ValueError("=> sample_pair_batch requires at least 2 datasets in the corpus.")

        pairs: list[tuple[DatasetPatch, DatasetPatch, int]] = []
        for pair_index in range(pairs_per_batch):
            is_positive_pair = pair_index % 2 == 0
            dataset_id_a = self.random_generator.choice(dataset_ids)
            if is_positive_pair:
                dataset_id_b = dataset_id_a
            else:
                other_ids = [d for d in dataset_ids if d != dataset_id_a]
                dataset_id_b = self.random_generator.choice(other_ids)

            patch_a = self.sample_patch(dataset_id_a)
            patch_b = self.sample_patch(dataset_id_b)
            pairs.append((patch_a, patch_b, int(is_positive_pair)))
        return pairs
