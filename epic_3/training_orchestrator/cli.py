"""Command-line entry point for preparing Epic-3 training jobs."""

from __future__ import annotations

import argparse

from .orchestrator import TrainingOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create training_jobs.json from model_config.json"
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--model-library-root", default="model_library")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = TrainingOrchestrator(args.model_library_root).prepare(
        session_id=args.session_id,
        metadata_path=args.metadata,
        model_config_path=args.model_config,
        train_path=args.train,
        test_path=args.test,
        session_dir=args.session_dir,
        output_path=args.output,
    )
    print(f"Prepared {manifest.total_jobs} queued training jobs")


if __name__ == "__main__":
    main()
