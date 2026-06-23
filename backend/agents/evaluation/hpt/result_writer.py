"""
Result persistence for Hyperparameter Tuning Agent
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultWriter:
    """Write tuning results to hpt_results.json with atomic operations"""
    
    def __init__(self, session_id: str, hpt_config: Dict[str, Any]):
        """
        Initialize result writer
        
        Args:
            session_id: Unique session identifier
            hpt_config: HPT configuration from config.ini
        """
        self.session_id = session_id
        self.session_root = Path(".mitra") / session_id
        self.hpt_config = hpt_config
        
        self.filename = hpt_config.get('HPT_OUTPUT_FILENAME', 'hpt_results.json')
        self.output_path = self.session_root / self.filename
        self.temp_path = self.output_path.with_suffix('.tmp')
        
        # Create output directory if needed
        self.session_root.mkdir(parents=True, exist_ok=True)
    
    def write_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any] = None) -> None:
        """
        Write results to file with atomic operation
        
        Args:
            results: List of tuning results for each model
            summary: Optional summary metadata
        """
        # Prepare output data
        output_data = {
            'hpt_results': results,
            'metadata': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'problem_type': results[0].get('problem_type', 'unknown') if results else 'unknown',
                'primary_metric': results[0].get('primary_metric', 'unknown') if results else 'unknown',
                'summary': summary or {}
            }
        }
        
        # Write to temporary file
        with open(self.temp_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Atomic rename
        os.replace(self.temp_path, self.output_path)

        logger.info(f"Results written to {self.output_path}")
        logger.info(f"Total models tuned: {len(results)}")
        
        # Create a simplified version for quick viewing
        self._write_summary(results, summary)
    
    def _write_summary(self, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        """
        Write a simplified summary file for quick viewing
        
        Args:
            results: List of tuning results
            summary: Summary metadata
        """
        summary_path = self.session_root / "hpt_summary.json"
        
        summary_data = {
            'models_tuned': len(results),
            'top_models': sorted(
                results,
                key=lambda x: x.get('val_metrics', {}).get('accuracy', 0),
                reverse=True
            )[:5] if results else [],
            'summary': summary or {}
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
    
    def read_results(self) -> Dict[str, Any]:
        """
        Read previously written results
        
        Returns:
            dict: Results data
        """
        if not self.output_path.exists():
            return None
        
        with open(self.output_path, 'r') as f:
            return json.load(f)