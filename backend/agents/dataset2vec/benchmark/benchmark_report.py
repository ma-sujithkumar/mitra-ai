"""Generate comprehensive benchmark reports."""

import json
import logging
from datetime import datetime
from pathlib import Path

from benchmark.config import BenchmarkConfig, EXTERNAL_DATASET_INFO
from benchmark.results_analyzer import ResultsAnalyzer

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """Generate formatted benchmark reports."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.analyzer = ResultsAnalyzer(config)

    def generate_summary_report(self) -> str:
        """Generate summary report of all benchmark results."""
        results_df = self.analyzer.analyze_all_results()
        summary_stats = self.analyzer.get_summary_statistics(results_df)

        report_lines = [
            "=" * 80,
            "DATASET2VEC PHASE 3 BENCHMARK REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY STATISTICS",
            "-" * 80,
            f"Total Datasets Tested:      {summary_stats.get('total_datasets', 0)}",
            f"Average Transfer Quality:   {summary_stats.get('avg_transfer_quality', 0.0):.1f}%",
            f"Average Mean |Delta|:       {summary_stats.get('avg_mean_delta', 0.0):.4f}",
            "",
            "GRADE DISTRIBUTION",
            "-" * 80,
            f"Excellent (70%+ transfer):  {summary_stats.get('excellent_count', 0)} datasets",
            f"Good      (50-70%):         {summary_stats.get('good_count', 0)} datasets",
            f"Fair      (30-50%):         {summary_stats.get('fair_count', 0)} datasets",
            f"Poor      (<30%):           {summary_stats.get('poor_count', 0)} datasets",
            "",
            "DETAILED RESULTS BY DATASET",
            "-" * 80,
        ]

        if not results_df.empty:
            report_lines.append(results_df.to_string(index=False))
        else:
            report_lines.append("No results available")

        report_lines.extend([
            "",
            "END OF REPORT",
            "=" * 80
        ])
        return "\n".join(report_lines)

    def generate_dataset_report(self, dataset_name: str) -> str:
        """Generate detailed report for a single dataset."""
        results_path = self.output_dir / dataset_name / "dataset_prior.json"
        if not results_path.exists():
            return f"Results not found for {dataset_name}"

        try:
            with open(results_path) as f:
                results = json.load(f)
        except Exception as e:
            return f"Failed to load results: {e}"

        report_lines = [
            "=" * 80,
            f"DATASET: {dataset_name}",
            "=" * 80,
            "",
            "QUERY CONFIGURATION",
            "-" * 80,
            f"Encoder Version:  {results.get('encoder_version', 'unknown')}",
            f"Primary Metric:   {results.get('primary_metric', 'unknown')}",
            f"Top-K Neighbors:  {results.get('top_k', 'unknown')}",
            "",
            "SIMILAR DATASETS FOUND (Neighbors)",
            "-" * 80,
        ]

        neighbors = results.get("neighbors", [])
        if neighbors:
            for idx, neighbor in enumerate(neighbors, 1):
                report_lines.append(f"{idx}. {neighbor.get('dataset_id', 'unknown')} "
                    f"(similarity: {neighbor.get('similarity', 0.0):.6f})")
                report_lines.append(f"   Best Model: {neighbor.get('best_model', 'unknown')} "
                    f"(f1_macro: {neighbor.get('metrics', {}).get('f1_macro', 0.0):.4f})")
        else:
            report_lines.append("No neighbors found")

        report_lines.extend(["", "END OF REPORT", "=" * 80])
        return "\n".join(report_lines)

    def save_summary_report(self) -> Path:
        """Save summary report to file."""
        report_content = self.generate_summary_report()
        report_path = self.output_dir / "BENCHMARK_SUMMARY.txt"
        report_path.write_text(report_content)
        logger.info("=> Saved summary report to %s", report_path)
        return report_path

    def save_dataset_report(self, dataset_name: str) -> Path:
        """Save detailed report for a dataset."""
        report_content = self.generate_dataset_report(dataset_name)
        report_path = self.output_dir / f"REPORT_{dataset_name}.txt"
        report_path.write_text(report_content)
        logger.info("=> Saved dataset report to %s", report_path)
        return report_path

    def save_all_reports(self) -> list[Path]:
        """Save all reports (summary + per-dataset)."""
        paths = [self.save_summary_report()]
        for dataset_name in self.config.external_datasets:
            paths.append(self.save_dataset_report(dataset_name))
        return paths
