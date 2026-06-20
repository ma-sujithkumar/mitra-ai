"""Analyze and compare Phase 3 benchmarking results."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from benchmark.config import BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class VerificationStats:
    """Statistics from a single dataset's verification run."""
    dataset_name: str
    n_neighbors: int
    n_ranked_models: int
    n_within_tolerance: int
    mean_abs_delta: float
    best_achieved_metric: float
    best_expected_metric: float

    @property
    def transfer_quality(self) -> float:
        """Percentage of models within tolerance."""
        if self.n_ranked_models == 0:
            return 0.0
        return (self.n_within_tolerance / self.n_ranked_models) * 100

    @property
    def grade(self) -> str:
        """Qualitative grade based on transfer quality."""
        quality = self.transfer_quality
        if quality >= 70:
            return "EXCELLENT"
        elif quality >= 50:
            return "GOOD"
        elif quality >= 30:
            return "FAIR"
        else:
            return "POOR"


class ResultsAnalyzer:
    """Analyze benchmark results across datasets."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)

    def extract_verification_stats(self, results: dict) -> Optional[VerificationStats]:
        """Extract verification statistics from query results."""
        if not results or "verification_summary" not in results:
            return None

        summary = results["verification_summary"]
        ranked_models = results.get("ranked_models", [])

        best_achieved = 0.0
        best_expected = 0.0
        if ranked_models:
            for model in ranked_models:
                if "verification" in model and model["verification"].get("trained"):
                    achieved = model["verification"].get("achieved_metric", 0.0)
                    expected = model.get("expected_metric", 0.0)
                    best_achieved = max(best_achieved, achieved)
                    best_expected = max(best_expected, expected)

        return VerificationStats(
            dataset_name=results.get("query_dataset_id", "unknown"),
            n_neighbors=len(results.get("neighbors", [])),
            n_ranked_models=len(ranked_models),
            n_within_tolerance=summary.get("n_within_tolerance", 0),
            mean_abs_delta=summary.get("mean_abs_delta", float("nan")),
            best_achieved_metric=best_achieved,
            best_expected_metric=best_expected,
        )

    def analyze_all_results(self) -> pd.DataFrame:
        """Analyze all benchmark results."""
        stats_list = []
        for dataset_name in self.config.external_datasets:
            results_path = self.output_dir / dataset_name / "dataset_prior.json"
            if not results_path.exists():
                logger.warning("=> Results not found: %s", dataset_name)
                continue
            try:
                with open(results_path) as f:
                    results = json.load(f)
                stats = self.extract_verification_stats(results)
                if stats:
                    stats_list.append(stats)
            except Exception as e:
                logger.error("=> Failed to analyze %s: %s", dataset_name, e)

        if not stats_list:
            logger.warning("=> No results to analyze")
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "Dataset": stats.dataset_name,
                "Neighbors": stats.n_neighbors,
                "Models": stats.n_ranked_models,
                "Within Tolerance": f"{stats.n_within_tolerance}/{stats.n_ranked_models}",
                "Transfer Quality": f"{stats.transfer_quality:.1f}%",
                "Grade": stats.grade,
                "Mean |Delta|": f"{stats.mean_abs_delta:.4f}",
                "Best Achieved": f"{stats.best_achieved_metric:.4f}",
            }
            for stats in stats_list
        ])

    def get_summary_statistics(self, df: pd.DataFrame) -> dict:
        """Extract summary statistics from results DataFrame."""
        if df.empty:
            return {}
        transfer_qualities = []
        deltas = []
        for idx, row in df.iterrows():
            try:
                quality_str = row.get("Transfer Quality", "0%").rstrip("%")
                transfer_qualities.append(float(quality_str))
                delta_str = row.get("Mean |Delta|", "0")
                deltas.append(float(delta_str))
            except ValueError:
                continue
        return {
            "total_datasets": len(df),
            "avg_transfer_quality": np.mean(transfer_qualities) if transfer_qualities else 0.0,
            "avg_mean_delta": np.mean(deltas) if deltas else 0.0,
            "excellent_count": (df["Grade"] == "EXCELLENT").sum(),
            "good_count": (df["Grade"] == "GOOD").sum(),
            "fair_count": (df["Grade"] == "FAIR").sum(),
            "poor_count": (df["Grade"] == "POOR").sum(),
        }
