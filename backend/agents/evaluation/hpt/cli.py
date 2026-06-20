#!/usr/bin/env python3
"""
Command-line interface for Hyperparameter Tuning Agent
"""
import argparse
import sys
import logging
from pathlib import Path

from .agent import HyperparameterTuningAgent


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning Agent - Find optimal model hyperparameters using Optuna"
    )
    
    parser.add_argument(
        "--session-id",
        required=True,
        help="Session ID (e.g., '20260618_123456')"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    parser.add_argument(
        "--max-trials",
        type=int,
        help="Override MAX_HPT_TRIALS from config.ini"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running tuning"
    )
    
    args = parser.parse_args()
    
    # Validate session exists
    session_root = Path(".mitra") / args.session_id
    if not session_root.exists():
        print(f"Error: Session directory not found: {session_root}", file=sys.stderr)
        sys.exit(1)
    
    # Create agent
    agent = HyperparameterTuningAgent(
        session_id=args.session_id,
        verbose=args.verbose
    )
    
    # Override max trials if specified
    if args.max_trials:
        agent.hpt_config['MAX_HPT_TRIALS'] = args.max_trials
        agent.logger.info(f"Overriding MAX_HPT_TRIALS to {args.max_trials}")
    
    # Dry run
    if args.dry_run:
        agent.logger.info("DRY RUN: Validating configuration")
        agent.logger.info(f"Session: {args.session_id}")
        agent.logger.info(f"Models to tune: {len(agent.model_config_sorted)}")
        
        for model in agent.model_config_sorted:
            hp_space = model.get('hp_space', {})
            agent.logger.info(f"  - {model.get('name')}: {len(hp_space)} hyperparameters")
        
        agent.logger.info("Dry run complete. Configuration is valid.")
        return
    
    # Run tuning
    try:
        results = agent.run()
        print(f"\n✓ Hyperparameter tuning completed successfully!")
        print(f"  - Models tuned: {len(results)}")
        print(f"  - Results saved to: {session_root}/hpt_results.json")
        
        # Print top model
        if results:
            top_model = max(results, key=lambda x: x.get('val_metrics', {}).get('accuracy', 0))
            print(f"\n🏆 Best model: {top_model.get('name')}")
            print(f"   Validation accuracy: {top_model.get('val_metrics', {}).get('accuracy', 0):.4f}")
            print(f"   Overfitting gap: {top_model.get('overfitting', {}).get('gap', 0):.4f}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()