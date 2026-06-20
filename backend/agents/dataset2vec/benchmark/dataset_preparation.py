"""Prepare test datasets for benchmarking."""

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DatasetPreparator:
    """Prepare and save datasets in NPZ format for Phase 3 benchmarking."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_and_save(self, name: str, features: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Path:
        """Prepare dataset with train/test split and save as NPZ."""
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if labels.dtype == object or labels.dtype.kind == "U":
            encoder = LabelEncoder()
            labels_encoded = encoder.fit_transform(labels)
        else:
            labels_encoded = labels.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded, test_size=test_size, random_state=42, stratify=labels_encoded
        )

        output_path = self.output_dir / f"{name}.npz"
        np.savez(
            output_path,
            X_train=X_train.astype(np.float32),
            y_train=y_train.astype(np.int32),
            X_test=X_test.astype(np.float32),
            y_test=y_test.astype(np.int32),
            task_type=np.array("classification", dtype=object),
        )
        logger.info("=> Saved %s: X_train=%s, X_test=%s", name, X_train.shape, X_test.shape)
        return output_path

    def download_car_evaluation(self) -> Optional[Path]:
        """Download and prepare Car Evaluation dataset."""
        logger.info("=> Downloading Car Evaluation Dataset...")
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
            data = urllib.request.urlopen(url, timeout=10).read().decode("utf-8")
            lines = data.strip().split("\n")
            features_map = {
                "buying": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
                "maint": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
                "doors": {"2": 0, "3": 1, "4": 2, "5more": 3},
                "persons": {"2": 0, "4": 1, "more": 2},
                "lug_boot": {"small": 0, "med": 1, "big": 2},
                "safety": {"low": 0, "med": 1, "high": 2},
                "class": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3},
            }
            features, labels = [], []
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) == 7:
                    row = [features_map["buying"][parts[0]], features_map["maint"][parts[1]],
                           features_map["doors"][parts[2]], features_map["persons"][parts[3]],
                           features_map["lug_boot"][parts[4]], features_map["safety"][parts[5]]]
                    features.append(row)
                    labels.append(features_map["class"][parts[6]])
            if features:
                return self.prepare_and_save("ext-car-evaluation", np.array(features, dtype=float), np.array(labels))
        except Exception as e:
            logger.warning("=> Car Evaluation download failed: %s", e)
        return None

    def download_monks(self) -> Optional[Path]:
        """Download and prepare Monks Problem dataset."""
        logger.info("=> Downloading Monks Dataset...")
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train"
            data = urllib.request.urlopen(url, timeout=10).read().decode("utf-8")
            lines = data.strip().split("\n")
            features, labels = [], []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 7:
                    labels.append(int(parts[0]))
                    features.append([int(p) for p in parts[1:7]])
            if features:
                return self.prepare_and_save("ext-monks", np.array(features, dtype=float), np.array(labels))
        except Exception as e:
            logger.warning("=> Monks download failed: %s", e)
        return None

    def download_spambase(self) -> Optional[Path]:
        """Download and prepare Spambase Email Classification dataset."""
        logger.info("=> Downloading Spambase Dataset...")
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
            data = urllib.request.urlopen(url, timeout=10).read().decode("utf-8")
            lines = data.strip().split("\n")
            features, labels = [], []
            for line in lines:
                parts = [float(x) for x in line.strip().split(",")]
                if len(parts) == 58:
                    features.append(parts[:-1])
                    labels.append(int(parts[-1]))
            if features and len(features) > 20:
                return self.prepare_and_save("ext-spambase", np.array(features, dtype=float), np.array(labels))
        except Exception as e:
            logger.warning("=> Spambase download failed: %s", e)
        return None

    def download_letter_recognition(self) -> Optional[Path]:
        """Download and prepare Letter Recognition dataset."""
        logger.info("=> Downloading Letter Recognition Dataset...")
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
            data = urllib.request.urlopen(url, timeout=10).read().decode("utf-8")
            lines = data.strip().split("\n")
            features, labels = [], []
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) == 17:
                    labels.append(ord(parts[0]) - ord("A"))
                    features.append([float(p) for p in parts[1:]])
            if features and len(features) > 20:
                return self.prepare_and_save("ext-letter-recognition", np.array(features, dtype=float), np.array(labels))
        except Exception as e:
            logger.warning("=> Letter Recognition download failed: %s", e)
        return None

    def prepare_all_external_datasets(self) -> dict[str, bool]:
        """Prepare all external benchmark datasets."""
        results = {}
        results["car-evaluation"] = self.download_car_evaluation() is not None
        results["monks"] = self.download_monks() is not None
        results["spambase"] = self.download_spambase() is not None
        results["letter-recognition"] = self.download_letter_recognition() is not None
        return results

    def list_prepared_datasets(self) -> list[str]:
        """List all prepared datasets in output directory."""
        return sorted([f.stem for f in self.output_dir.glob("*.npz")])
