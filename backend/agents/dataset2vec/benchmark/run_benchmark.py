#!/usr/bin/env python
"""
Comprehensive Dataset2Vec Phase 3 Benchmarking Suite

Orchestrates dataset preparation, Phase 3 query execution, results analysis,
and report generation for benchmarking the warm-start model suggestion system.

Usage:
    python benchmark/run_benchmark.py -c config/config.ini [--skip-prepare] [--skip-sanity]
"""

import argparse
import configparser
import logging
import sys
from pathlib import Path

from benchmark.benchmark_report import BenchmarkReporter
from benchmark.config import BenchmarkConfig
from benchmark.dataset_preparation import DatasetPreparator
from benchmark.query_runner import QueryRunner
from benchmark.results_analyzer import ResultsAnalyzer


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Dataset2Vec Phase 3 Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark pipeline
  python benchmark/run_benchmark.py -c config/config.ini

  # Skip dataset preparation (datasets already prepared)
  python benchmark/run_benchmark.py -c config/config.ini --skip-prepare

  # Skip sanity check on corpus dataset
  python benchmark/run_benchmark.py -c config/config.ini --skip-sanity

  # Run with verbose logging
  python benchmark/run_benchmark.py -c config/config.ini -v
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Path to config.ini",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation (use existing datasets)",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip sanity check on corpus dataset",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_config(config_ini_path: str) -> tuple[BenchmarkConfig, str, str]:
    """
    Load configuration from config.ini.

    Args:
        config_ini_path: Path to config.ini

    Returns:
        Tuple of (BenchmarkConfig, python_binary_path, config_ini_path)
    """
    parser = configparser.ConfigParser()
    parser.read(config_ini_path)

    python_binary = parser.get("python", "PYTHON")
    config_yaml = parser.get("paths", "config_yaml")

    return BenchmarkConfig(), python_binary, config_ini_path


def main() -> int:
    """Execute benchmarking pipeline."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=> Dataset2Vec Phase 3 Benchmarking Suite")
    logger.info("=> Python %s.%s", sys.version_info.major, sys.version_info.minor)

    # Load configuration
    try:
        config, python_binary, config_ini = load_config(args.config)
    except Exception as exception:
        logger.error("=> Failed to load config: %s", exception)
        return 1

    logger.info("=> Configuration loaded from %s", args.config)
    logger.info("=> Python binary: %s", python_binary)
    logger.info("=> Output directory: %s", config.output_dir)

    # Step 1: Prepare datasets
    if not args.skip_prepare:
        logger.info("")
        logger.info("STEP 1: PREPARE EXTERNAL DATASETS")
        logger.info("-" * 80)

        preparator = DatasetPreparator(config.external_datasets_dir)
        results = preparator.prepare_all_external_datasets()

        for dataset_name, success in results.items():
            status = "✓" if success else "✗"
            logger.info("=> %s %s", status, dataset_name)

        prepared = preparator.list_prepared_datasets()
        logger.info("=> Ready: %d datasets", len(prepared))
    else:
        logger.info("=> Skipping dataset preparation (--skip-prepare)")

    # Step 2: Sanity check on corpus dataset
    if not args.skip_sanity:
        logger.info("")
        logger.info("STEP 2: SANITY CHECK (Query on corpus dataset)")
        logger.info("-" * 80)

        runner = QueryRunner(config, python_binary, args.config)
        success = runner.run_corpus_sanity_check()

        if success:
            logger.info("=> Sanity check passed ✓")
        else:
            logger.warning("=> Sanity check failed ✗ (continuing anyway)")
    else:
        logger.info("=> Skipping sanity check (--skip-sanity)")

    # Step 3: Run queries on external datasets
    logger.info("")
    logger.info("STEP 3: EXECUTE PHASE 3 QUERIES")
    logger.info("-" * 80)

    runner = QueryRunner(config, python_binary, args.config)
    query_results = runner.run_all_external_datasets()

    for dataset_name, success in query_results.items():
        status = "✓" if success else "✗"
        logger.info("=> %s %s", status, dataset_name)

    successful_count = sum(1 for v in query_results.values() if v)
    logger.info("=> Completed: %d/%d datasets", successful_count, len(query_results))

    if successful_count == 0:
        logger.error("=> No queries succeeded; skipping analysis and reporting")
        return 1

    # Step 4: Analyze results
    logger.info("")
    logger.info("STEP 4: ANALYZE RESULTS")
    logger.info("-" * 80)

    analyzer = ResultsAnalyzer(config)
    results_df = analyzer.analyze_all_results()

    if not results_df.empty:
        logger.info("=> Analysis complete:")
        logger.info(results_df.to_string(index=False))
    else:
        logger.warning("=> No results to analyze")

    # Step 5: Generate reports
    logger.info("")
    logger.info("STEP 5: GENERATE REPORTS")
    logger.info("-" * 80)

    reporter = BenchmarkReporter(config)
    report_paths = reporter.save_all_reports()

    for path in report_paths:
        logger.info("=> Saved %s", path)

    # Print summary report to console
    logger.info("")
    logger.info("SUMMARY REPORT")
    logger.info("-" * 80)
    summary = reporter.generate_summary_report()
    print(summary)

    logger.info("")
    logger.info("=> Benchmarking complete! See %s for full reports.", config.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
