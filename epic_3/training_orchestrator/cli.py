"""Command-line entry point for Epic-3 training orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

from .orchestrator import TrainingOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create training_jobs.json and optionally execute all jobs locally "
            "or in parallel with Ray to produce training_summary.json"
        )
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--model-library-root", default="model_library")
    parser.add_argument("--output", help="Path for training_jobs.json")

    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute-local",
        action="store_true",
        help="Run Onkar's local training worker after preparing jobs",
    )
    execution.add_argument(
        "--execute-ray",
        action="store_true",
        help="Run all prepared jobs in parallel through Onkar's Ray executor",
    )

    parser.add_argument("--target-column")
    parser.add_argument("--summary-output", help="Path for training_summary.json")
    parser.add_argument(
        "--ray-timeout-sec",
        type=float,
        help="Overall timeout for collecting the submitted Ray jobs",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    orchestrator = TrainingOrchestrator(args.model_library_root)
    manifest_path = Path(
        args.output or Path(args.session_dir) / "training_jobs.json"
    ).resolve()

    common_arguments = {
        "session_id": args.session_id,
        "metadata_path": args.metadata,
        "model_config_path": args.model_config,
        "train_path": args.train,
        "test_path": args.test,
        "session_dir": args.session_dir,
        "target_column": args.target_column,
        "manifest_path": manifest_path,
        "summary_path": args.summary_output,
    }

    if args.execute_local:
        summary = orchestrator.prepare_and_execute_local(**common_arguments)
        print(
            f"Local training finished: {summary.completed} completed, "
            f"{summary.failed} failed"
        )
        return

    if args.execute_ray:
        summary = orchestrator.prepare_and_execute_ray(
            **common_arguments,
            timeout_sec=args.ray_timeout_sec,
        )
        print(
            f"Ray training finished: {summary.completed} completed, "
            f"{summary.failed} failed"
        )
        return

    manifest = orchestrator.prepare(
        session_id=args.session_id,
        metadata_path=args.metadata,
        model_config_path=args.model_config,
        train_path=args.train,
        test_path=args.test,
        session_dir=args.session_dir,
        output_path=manifest_path,
    )
    print(f"Prepared {manifest.total_jobs} queued training jobs")


if __name__ == "__main__":
    main()
