"""Execute Phase 3 queries across multiple datasets."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from benchmark.config import BenchmarkConfig

logger = logging.getLogger(__name__)


class QueryRunner:
    """Execute Phase 3 (query.py) on multiple datasets and collect results."""

    def __init__(self, config: BenchmarkConfig, python_binary: str, config_ini: str):
        self.config = config
        self.python_binary = python_binary
        self.config_ini = config_ini
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_query(self, dataset_path: str, dataset_name: str, output_subdir: Optional[str] = None) -> bool:
        """Execute query.py on a single dataset."""
        if output_subdir is None:
            output_subdir = dataset_name

        output_dir = self.output_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        verify_flag = ["--verify"] if self.config.verify_results else []

        command = [
            self.python_binary, "query.py", "-c", self.config_ini, "-i", dataset_path,
            "-o", str(output_dir), "-k", str(self.config.top_k_neighbors), *verify_flag, "-v",
        ]

        logger.info("=> Running query on %s...", dataset_name)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600,
                env={**subprocess.os.environ, "CUDA_VISIBLE_DEVICES": self.config.cuda_device})
            if result.returncode != 0:
                logger.error("=> Query failed for %s", dataset_name)
                return False
            logger.info("=> Query completed for %s", dataset_name)
            return True
        except subprocess.TimeoutExpired:
            logger.error("=> Query timeout for %s", dataset_name)
            return False
        except Exception as e:
            logger.error("=> Query error for %s: %s", dataset_name, e)
            return False

    def run_all_external_datasets(self) -> dict[str, bool]:
        """Execute queries on all external benchmark datasets."""
        results = {}
        external_dir = Path(self.config.external_datasets_dir)
        for dataset_name in self.config.external_datasets:
            dataset_path = external_dir / f"{dataset_name}.npz"
            if not dataset_path.exists():
                logger.warning("=> Dataset not found: %s", dataset_path)
                results[dataset_name] = False
                continue
            results[dataset_name] = self.run_query(str(dataset_path), dataset_name)
        return results

    def run_corpus_sanity_check(self) -> bool:
        """Execute query on an existing corpus dataset for sanity check."""
        if not self.config.corpus_sanity_check:
            logger.info("=> Corpus sanity check disabled")
            return True
        dataset_path = Path("corpus") / f"{self.config.corpus_sanity_check}.npz"
        if not dataset_path.exists():
            logger.warning("=> Sanity check dataset not found: %s", dataset_path)
            return False
        return self.run_query(str(dataset_path), self.config.corpus_sanity_check, output_subdir="sanity_check")

    def load_results(self, dataset_name: str) -> Optional[dict]:
        """Load dataset_prior.json results for a dataset."""
        results_path = self.output_dir / dataset_name / "dataset_prior.json"
        if not results_path.exists():
            logger.warning("=> Results not found: %s", results_path)
            return None
        try:
            with open(results_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error("=> Failed to load results for %s: %s", dataset_name, e)
            return None
