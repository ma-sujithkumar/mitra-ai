"""CLI for executing one generated TrainingJob locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.agents.training_orchestrator.contracts import TrainingJob

from .trainer import train_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to one TrainingJob JSON object")
    parser.add_argument(
        "--model-library-root",
        default="model_library",
        help="Path containing ml_kit.py",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="CSV target column; default uses a conventional name or final column",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.job).read_text(encoding="utf-8"))
    job = TrainingJob.model_validate(payload)
    result = train_job(
        job,
        model_library_root=args.model_library_root,
        target_column=args.target_column,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
