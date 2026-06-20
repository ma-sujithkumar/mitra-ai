"""
run_judge.py - CLI entry point for the Judge Agent.

Usage:
    python run_judge.py -i <input_json> -o <output_dir> [-v] [--no-llm]

The input JSON must follow the adapter schema (see input_format_requirement.md).
Output is written to <output_dir>/judge_decision.json.
"""

import argparse
import json
import logging
import os
import sys

from .adapter import UpstreamAdapter
from .config_loader import load_judge_config
from .judge_agent import JudgeAgent
from .schemas import JudgeInput


def _configure_logging(verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=log_level,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge Agent: rank ML model candidates and nominate the top model."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input_json",
        default=None,
        help="Path to a pre-built adapter-schema input JSON.",
    )
    input_group.add_argument(
        "--hpt-json",
        default=None,
        help="Path to hpt_results.json from HyperparameterTuningAgent (alternative to -i).",
    )
    parser.add_argument(
        "--shap-dir",
        default=None,
        help="Root of SHAP output directory for this session (used with --hpt-json).",
    )
    parser.add_argument(
        "--task-type",
        default=None,
        choices=["classification", "regression"],
        help="Task type required when using --hpt-json.",
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        default=None,
        help="Directory for judge_decision.json output (REQUIRED, created if absent).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Run rule-only mode (skip the LLM rationale call).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if not args.output_dir:
        logger.error("=> -o/--output_dir is required.")
        sys.exit(1)

    logger.info("=> Loading config...")
    config = load_judge_config()
    top_n_shap = config.get("shap_top_n_features", 5)

    adapter = UpstreamAdapter()

    if args.hpt_json:
        # Build JudgeInput directly from hpt_results.json + optional SHAP outputs
        if not args.task_type:
            logger.error("=> --task-type is required when using --hpt-json.")
            sys.exit(1)
        if not os.path.exists(args.hpt_json):
            logger.error("=> hpt_results.json not found: %s", args.hpt_json)
            sys.exit(1)

        logger.info("=> Building JudgeInput from HPT results: %s", args.hpt_json)
        judge_input = adapter.adapt_from_hpt_results(
            hpt_json_path=args.hpt_json,
            task_type=args.task_type,
            shap_dir=args.shap_dir,
            top_n_shap=top_n_shap,
        )
        logger.info(
            "=> Built JudgeInput from HPT (%d candidates, shap_dir=%s).",
            len(judge_input.candidates),
            args.shap_dir,
        )
    else:
        # Legacy path: pre-built JSON input file
        if not os.path.exists(args.input_json):
            logger.error("=> Input JSON not found: %s", args.input_json)
            sys.exit(1)

        logger.info("=> Reading input JSON: %s", args.input_json)
        with open(args.input_json, "r") as input_file:
            raw_input = json.load(input_file)

        # Support both a pre-built JudgeInput dict and the raw adapter list format.
        if "candidates" in raw_input:
            judge_input = JudgeInput.model_validate(raw_input)
            logger.info(
                "=> Parsed JudgeInput directly (%d candidates).", len(judge_input.candidates)
            )
        else:
            candidate_raw_list = raw_input.get("candidate_models", [])
            judge_input = adapter.adapt_judge_input(
                candidate_raw_list=candidate_raw_list,
                dataset_id=raw_input.get("dataset_id"),
                minidata=raw_input.get("minidata"),
                metadata=raw_input.get("metadata"),
            )
            logger.info(
                "=> Adapted %d candidates via UpstreamAdapter.", len(judge_input.candidates)
            )

    os.makedirs(args.output_dir, exist_ok=True)

    use_llm = not args.no_llm
    logger.info("=> Running JudgeAgent (use_llm=%s)...", use_llm)
    agent = JudgeAgent(config=config)
    decision = agent.judge(judge_input=judge_input, use_llm=use_llm)

    output_path = os.path.join(args.output_dir, "judge_decision.json")
    with open(output_path, "w") as output_file:
        json.dump(decision.model_dump(), output_file, indent=2)

    logger.info("=> Decision written to: %s", output_path)
    logger.info("=> Selected model: %s", decision.selected_model)
    for ranked_model in decision.ranked_models:
        logger.info(
            "  Rank %d | %s | %s | score=%.4f",
            ranked_model.rank,
            ranked_model.model_name,
            ranked_model.verdict,
            ranked_model.score,
        )


if __name__ == "__main__":
    main()
