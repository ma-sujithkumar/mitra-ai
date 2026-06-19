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
        """
        Initialize reporter.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.analyzer = ResultsAnalyzer(config)

    def generate_summary_report(self) -> str:
        """
        Generate summary report of all benchmark results.

        Returns:
            Formatted report string
        """
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

        report_lines.extend(
            [
                "",
                "INTERPRETATION",
                "-" * 80,
                "Transfer Quality: Percentage of top-N models whose suggestions generalize",
                "                   well to the test dataset (delta < tolerance_delta).",
                "Mean |Delta|:     Average absolute difference between expected and",
                "                   achieved metrics across all suggested models.",
                "Grade:            Qualitative assessment based on transfer quality.",
                "",
                "RECOMMENDATIONS",
                "-" * 80,
            ]
        )

        if summary_stats.get("excellent_count", 0) > 0:
            report_lines.append(
                "✓ System shows strong generalization on datasets similar to the training corpus."
            )
        if summary_stats.get("avg_transfer_quality", 0) > 50:
            report_lines.append("✓ Warm-start model suggestions are reliable for most datasets.")
        if summary_stats.get("avg_mean_delta", 0) < 0.05:
            report_lines.append("✓ Model suggestions predict performance very accurately.")
        if summary_stats.get("poor_count", 0) > 0:
            report_lines.append(
                "⚠ Some datasets show poor transfer. Consider fine-tuning suggested models."
            )

        report_lines.extend(["", "END OF REPORT", "=" * 80])
        return "\n".join(report_lines)

    def generate_dataset_report(self, dataset_name: str) -> str:
        """
        Generate detailed report for a single dataset.

        Args:
            dataset_name: Dataset identifier

        Returns:
            Formatted report string
        """
        results_path = self.output_dir / dataset_name / "dataset_prior.json"
        if not results_path.exists():
            return f"Results not found for {dataset_name}"

        try:
            with open(results_path) as file:
                results = json.load(file)
        except Exception as exception:
            return f"Failed to load results: {exception}"

        report_lines = [
            "=" * 80,
            f"DATASET: {dataset_name}",
            "=" * 80,
            "",
        ]

        # Dataset info if available
        if dataset_name in EXTERNAL_DATASET_INFO:
            info = EXTERNAL_DATASET_INFO[dataset_name]
            report_lines.extend(
                [
                    "DATASET INFORMATION",
                    "-" * 80,
                    f"Source:   {info.get('source', 'unknown')}",
                    f"Task:     {info.get('task', 'unknown')}",
                    f"Domain:   {info.get('domain', 'unknown')}",
                    f"Features: {info.get('n_features', 'unknown')}",
                    "",
                ]
            )

        # Encoder info
        report_lines.extend(
            [
                "QUERY CONFIGURATION",
                "-" * 80,
                f"Encoder Version:  {results.get('encoder_version', 'unknown')}",
                f"Primary Metric:   {results.get('primary_metric', 'unknown')}",
                f"Top-K Neighbors:  {results.get('top_k', 'unknown')}",
                "",
            ]
        )

        # Neighbors found
        report_lines.extend(
            [
                "SIMILAR DATASETS FOUND (Neighbors)",
                "-" * 80,
            ]
        )

        neighbors = results.get("neighbors", [])
        if neighbors:
            for idx, neighbor in enumerate(neighbors, 1):
                report_lines.append(
                    f"{idx}. {neighbor.get('dataset_id', 'unknown')} "
                    f"(similarity: {neighbor.get('similarity', 0.0):.6f})"
                )
                report_lines.append(
                    f"   Best Model: {neighbor.get('best_model', 'unknown')} "
                    f"(f1_macro: {neighbor.get('metrics', {}).get('f1_macro', 0.0):.4f})"
                )
        else:
            report_lines.append("No neighbors found")

        report_lines.append("")

        # Top model suggestions
        report_lines.extend(
            [
                "TOP MODEL SUGGESTIONS (Ranked)",
                "-" * 80,
            ]
        )

        ranked_models = results.get("ranked_models", [])
        if ranked_models:
            for idx, model in enumerate(ranked_models[:5], 1):
                verification = model.get("verification", {})
                within = "✓" if verification.get("within_tolerance", False) else "✗"
                report_lines.append(f"{idx}. {model.get('model_name', 'unknown')} {within}")
                report_lines.append(
                    f"   Expected: {model.get('expected_metric', 0.0):.4f} "
                    f"-> Achieved: {verification.get('achieved_metric', 0.0):.4f}"
                )
                report_lines.append(
                    f"   Delta: {verification.get('delta_vs_expected', 0.0):.4f}"
                )
        else:
            report_lines.append("No model suggestions available")

        report_lines.append("")

        # Verification summary
        if "verification_summary" in results:
            summary = results["verification_summary"]
            report_lines.extend(
                [
                    "VERIFICATION SUMMARY",
                    "-" * 80,
                    f"Models Within Tolerance: {summary.get('n_within_tolerance', 0)}"
                    f"/{summary.get('total_models', 0)}",
                    f"Mean Absolute Delta:     {summary.get('mean_abs_delta', 0.0):.4f}",
                    f"Best Achieved Metric:    {summary.get('best_achieved', 0.0):.4f}",
                    "",
                ]
            )

        report_lines.extend(["END OF REPORT", "=" * 80])
        return "\n".join(report_lines)

    def save_summary_report(self) -> Path:
        """
        Save summary report to file.

        Returns:
            Path to saved report
        """
        report_content = self.generate_summary_report()
        report_path = self.output_dir / "BENCHMARK_SUMMARY.txt"
        report_path.write_text(report_content)
        logger.info("=> Saved summary report to %s", report_path)
        return report_path

    def save_dataset_report(self, dataset_name: str) -> Path:
        """
        Save detailed report for a dataset.

        Args:
            dataset_name: Dataset identifier

        Returns:
            Path to saved report
        """
        report_content = self.generate_dataset_report(dataset_name)
        report_path = self.output_dir / f"REPORT_{dataset_name}.txt"
        report_path.write_text(report_content)
        logger.info("=> Saved dataset report to %s", report_path)
        return report_path

    def save_all_reports(self) -> list[Path]:
        """
        Save all reports (summary + per-dataset).

        Returns:
            List of saved report paths
        """
        paths = [self.save_summary_report()]

        for dataset_name in self.config.external_datasets:
            paths.append(self.save_dataset_report(dataset_name))

        return paths
