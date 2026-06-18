#!/usr/bin/env python
import os
import sys
import numpy as np
import urllib.request
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import zipfile
import io
import csv

OUTPUT_DIR = Path(__file__).parent.parent / "test_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_and_save(name: str, X: np.ndarray, y: np.ndarray) -> None:
    """Prepare dataset with train/test split and save as NPZ."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure y is integer labels
    if y.dtype == object or y.dtype.kind == 'U':
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
    else:
        y_encoded = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    output_path = OUTPUT_DIR / f"{name}.npz"
    np.savez(
        output_path,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.int32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.int32),
        task_type=np.array("classification", dtype=object)
    )
    print(f"=> Saved {name}: X_train={X_train.shape}, X_test={X_test.shape}")

def download_car_evaluation():
    """Car Evaluation Dataset from UCI."""
    print("=> Downloading Car Evaluation Dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    try:
        data = urllib.request.urlopen(url, timeout=10).read().decode('utf-8')
        lines = data.strip().split('\n')

        features_map = {
            'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
            'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
            'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
            'persons': {'2': 0, '4': 1, 'more': 2},
            'lug_boot': {'small': 0, 'med': 1, 'big': 2},
            'safety': {'low': 0, 'med': 1, 'high': 2},
            'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        }

        X, y = [], []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 7:
                row = [features_map['buying'][parts[0]],
                       features_map['maint'][parts[1]],
                       features_map['doors'][parts[2]],
                       features_map['persons'][parts[3]],
                       features_map['lug_boot'][parts[4]],
                       features_map['safety'][parts[5]]]
                X.append(row)
                y.append(features_map['class'][parts[6]])

        if X:
            prepare_and_save("ext-car-evaluation", np.array(X, dtype=float), np.array(y))
    except Exception as e:
        print(f"   Failed: {e}")

def download_monks():
    """Monks Problem Dataset."""
    print("=> Downloading Monks Dataset...")
    try:
        # Use monks-1 dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train"
        data = urllib.request.urlopen(url, timeout=10).read().decode('utf-8')
        lines = data.strip().split('\n')

        X, y = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 7:
                y.append(int(parts[0]))
                X.append([int(p) for p in parts[1:7]])

        if X:
            prepare_and_save("ext-monks", np.array(X, dtype=float), np.array(y))
    except Exception as e:
        print(f"   Failed: {e}")

def download_spambase():
    """Spambase Email Classification Dataset."""
    print("=> Downloading Spambase Dataset...")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        data = urllib.request.urlopen(url, timeout=10).read().decode('utf-8')
        lines = data.strip().split('\n')

        X, y = [], []
        for line in lines:
            parts = [float(x) for x in line.strip().split(',')]
            if len(parts) == 58:
                X.append(parts[:-1])
                y.append(int(parts[-1]))

        if X and len(X) > 20:
            prepare_and_save("ext-spambase", np.array(X, dtype=float), np.array(y))
    except Exception as e:
        print(f"   Failed: {e}")

def download_credit_card():
    """Credit Card Default Dataset (subset)."""
    print("=> Downloading Credit Card Dataset...")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/default-of-credit-card/default%20of%20credit%20card%20clients.xls"
        # This one is tricky due to XLS format, skip if can't download easily
        print("   Skipped (XLS format requires special handling)")
    except Exception as e:
        print(f"   Failed: {e}")

def download_letter_recognition():
    """Letter Recognition Dataset."""
    print("=> Downloading Letter Recognition Dataset...")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
        data = urllib.request.urlopen(url, timeout=10).read().decode('utf-8')
        lines = data.strip().split('\n')

        X, y = [], []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 17:
                y.append(ord(parts[0]) - ord('A'))  # Convert A-Z to 0-25
                X.append([float(p) for p in parts[1:]])

        if X and len(X) > 20:
            prepare_and_save("ext-letter-recognition", np.array(X, dtype=float), np.array(y))
    except Exception as e:
        print(f"   Failed: {e}")

if __name__ == "__main__":
    print(f"=> Downloading external test datasets to {OUTPUT_DIR}...\n")

    download_car_evaluation()
    download_monks()
    download_spambase()
    download_letter_recognition()

    print(f"\n=> Done. Available test datasets:")
    for f in sorted(OUTPUT_DIR.glob("*.npz")):
        data = np.load(f)
        print(f"   {f.stem}: X_train={data['X_train'].shape}, X_test={data['X_test'].shape}")
