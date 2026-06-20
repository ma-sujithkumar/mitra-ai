import argparse
import sys
from pathlib import Path

from backend.agents.feature_engineering.orchestrator import FeatureEngineerOrchestrator


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Feature engineering pipeline (Google ADK harness, bring-your-own-LLM).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run the feature engineering pipeline on a dataset")
    run.add_argument("data", type=str, help="Path to CSV dataset")
    run.add_argument(
        "--task",
        type=str,
        required=False,
        default=None,
        choices=["classification", "regression"],
        help="Optional. If omitted, the pipeline infers from the target column using task_infer_nunique_threshold from config.",
    )
    run.add_argument("--target", type=str, required=True, help="Target column name")
    run.add_argument(
        "--model",
        type=str,
        required=True,
        help="ADK model string (e.g., 'gemini/gemini-2.0-flash', 'openai/gpt-4o'). Set the matching API key env var.",
    )
    run.add_argument("--config", type=str, default="config/config.yaml")

    args = parser.parse_args()

    if args.cmd == "run":
        orchestrator = FeatureEngineerOrchestrator(
            data_path=args.data,
            target_column=args.target,
            model_string=args.model,
            task=args.task,
            config_path=args.config,
        )
        output_dir, run_id = orchestrator.run()
        print(f"run_id: {run_id}")
        print(f"output_dir: {output_dir}")
        print(f"  - engineered_dataset.csv")
        print(f"  - feature_artifact.json")
        print(f"  - report.md")
        print(f"  - execution_log.txt")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
