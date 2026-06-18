"""Benchmark configuration and constants."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration."""
    # Dataset preparation
    test_datasets_dir: str = "benchmark/test_datasets"
    external_datasets_dir: str = "benchmark/external_datasets"
    output_dir: str = "claude_outputs/benchmark"

    # Query execution
    top_k_neighbors: int = 5
    n_top_models: int = 10
    verify_results: bool = True
    cuda_device: str = "0"

    # Tolerance thresholds
    tolerance_delta: float = 0.05
    min_similarity_threshold: float = 0.0

    # Datasets to benchmark
    external_datasets: list[str] = None
    corpus_sanity_check: Optional[str] = "abalone"

    def __post_init__(self):
        if self.external_datasets is None:
            self.external_datasets = [
                "ext-car-evaluation",
                "ext-monks",
                "ext-spambase",
                "ext-letter-recognition",
            ]

# Dataset metadata for interpretation
EXTERNAL_DATASET_INFO = {
    "ext-car-evaluation": {
        "source": "UCI ML Repository",
        "task": "Multi-class classification (4 classes: unacc, acc, good, vgood)",
        "n_features": 6,
        "domain": "Automotive"
    },
    "ext-monks": {
        "source": "UCI ML Repository",
        "task": "Binary classification (monks dataset)",
        "n_features": 6,
        "domain": "Synthetic benchmark"
    },
    "ext-spambase": {
        "source": "UCI ML Repository",
        "task": "Binary spam/ham classification",
        "n_features": 57,
        "domain": "Email/text"
    },
    "ext-letter-recognition": {
        "source": "UCI ML Repository",
        "task": "Multi-class letter recognition (A-Z)",
        "n_features": 16,
        "domain": "Computer vision"
    },
}
