"""Benchmark configuration and constants."""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration."""
    test_datasets_dir: str = "benchmark/test_datasets"
    external_datasets_dir: str = "benchmark/external_datasets"
    output_dir: str = "claude_outputs/benchmark"
    
    top_k_neighbors: int = 5
    n_top_models: int = 10
    verify_results: bool = True
    cuda_device: str = "0"
    
    tolerance_delta: float = 0.05
    min_similarity_threshold: float = 0.0
    
    external_datasets: list[str] = field(default_factory=lambda: [
        "ext-car-evaluation",
        "ext-monks",
        "ext-spambase",
        "ext-letter-recognition",
    ])
    corpus_sanity_check: Optional[str] = "abalone"

EXTERNAL_DATASET_INFO = {
    "ext-car-evaluation": {
        "source": "UCI ML Repository",
        "task": "Multi-class classification (4 classes)",
        "n_features": 6,
        "domain": "Automotive"
    },
    "ext-monks": {
        "source": "UCI ML Repository",
        "task": "Binary classification",
        "n_features": 6,
        "domain": "Synthetic"
    },
    "ext-spambase": {
        "source": "UCI ML Repository",
        "task": "Binary spam/ham classification",
        "n_features": 57,
        "domain": "Email/text"
    },
    "ext-letter-recognition": {
        "source": "UCI ML Repository",
        "task": "Multi-class letter recognition",
        "n_features": 16,
        "domain": "Vision"
    },
}
