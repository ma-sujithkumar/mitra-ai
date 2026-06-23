"""Standalone CLI to regenerate visualizations from an existing pipeline run directory."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from backend.agents.feature_engineering.visuals.dashboard import VisualDashboard


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.agents.feature_engineering.visuals.cli",
        description="Generate interactive HTML visualizations from a feature engineering pipeline run.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=str,
        help="Path to the pipeline output directory (contains feature_artifact.json).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose progress logging.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    artifact_path = run_dir / "feature_artifact.json"
    if not artifact_path.exists():
        print(
            f"ERROR: feature_artifact.json not found in {run_dir}. "
            "Is this a valid pipeline output directory?",
            file=sys.stderr,
        )
        return 1

    try:
        dashboard_path = VisualDashboard(run_dir, verbose=args.verbose).run()
        print(f"=> dashboard: {dashboard_path}")
        return 0
    except Exception as error:
        print(f"ERROR: visualization failed: {error}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
