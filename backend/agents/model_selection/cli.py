"""Command-line entry point for MITRA model selection."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .selector import select_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select trainable models only from model_library/ml_kit.py::MODEL_REGISTRY"
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument(
        "--feature-selection",
        required=True,
        help="Path to feature_selection.json",
    )
    parser.add_argument("--mini-data", help="Optional path to mini_data.csv")
    parser.add_argument(
        "--model-library-root",
        default="model_library",
        help="Path containing ml_kit.py and config/config.yaml",
    )
    parser.add_argument(
        "--output", default="model_config.json", help="Output model_config.json path"
    )
    parser.add_argument(
        "--report", default=None, help="Optional model_selection_report.json path"
    )
    parser.add_argument("--max-models", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    candidates = select_models(
        metadata_path=args.metadata,
        feature_selection_path=args.feature_selection,
        mini_data_path=args.mini_data,
        model_library_root=args.model_library_root,
        output_path=args.output,
        report_path=args.report,
        max_models=args.max_models,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "selected_models": [item.model_name for item in candidates],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
