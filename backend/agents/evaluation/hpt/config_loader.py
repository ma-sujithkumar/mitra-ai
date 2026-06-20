"""
Configuration loader for Hyperparameter Tuning Agent
Reads from config.ini and session-specific files
"""
import configparser
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration for the Hyperparameter Tuning Agent"""
    
    def __init__(self, session_id: str, config_path: str = "config.ini"):
        """
        Initialize configuration loader
        
        Args:
            session_id: Unique session identifier
            config_path: Path to project-wide config.ini
        """
        self.session_id = session_id
        self.session_root = Path(".mitra") / session_id
        self.config_path = Path(config_path)
        
        # Load config.ini
        self.config = configparser.ConfigParser()
        if self.config_path.exists():
            self.config.read(self.config_path)
        else:
            raise FileNotFoundError(f"config.ini not found at {config_path}")
        
        # Validate required sections
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all required config sections exist"""
        required_sections = ['paths', 'python', 'pipeline', 'training_api', 'hpt']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required config section '[{section}]' missing in config.ini")
    
    def get_hpt_config(self) -> Dict[str, Any]:
        """Extract HPT section from config.ini"""
        hpt = dict(self.config['hpt'])
        
        # Convert string values to appropriate types
        hpt['MAX_HPT_TRIALS'] = int(hpt.get('MAX_HPT_TRIALS', 5))
        hpt['OVERFITTING_GAP_THRESHOLD'] = float(hpt.get('OVERFITTING_GAP_THRESHOLD', 0.10))
        hpt['VAL_SPLIT_RATIO'] = float(hpt.get('VAL_SPLIT_RATIO', 0.2))
        hpt['HPT_N_JOBS'] = int(hpt.get('HPT_N_JOBS', 1))
        hpt['OPTUNA_SEED'] = int(hpt.get('OPTUNA_SEED', 42))
        
        return hpt
    
    def get_path_config(self) -> Dict[str, str]:
        """Extract paths section from config.ini"""
        return dict(self.config['paths'])
    
    def get_python_config(self) -> Dict[str, str]:
        """Extract python section from config.ini"""
        return dict(self.config['python'])
    
    def load_model_config(self) -> list:
        """
        Load model_config.json from session root
        
        Returns:
            list: Array of model entries with name, family, hp_space, priority
        """
        config_path = self.session_root / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"model_config.json not found at {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata.json from session root or reports/ subdirectory.

        The pipeline writes metadata into reports/metadata.json; earlier code
        placed it directly under the session root. Try both so HPT works
        regardless of which stage produced the session.
        """
        # Prefer reports/ location (current pipeline standard).
        candidates = [
            self.session_root / "reports" / "metadata.json",
            self.session_root / "metadata.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                with open(candidate, 'r') as metadata_file:
                    return json.load(metadata_file)
        raise FileNotFoundError(
            f"metadata.json not found at {self.session_root}. "
            f"Searched: {[str(path) for path in candidates]}"
        )
    
    def get_primary_metric(self, problem_type: str) -> str:
        """
        Get primary metric based on problem type from config.ini
        
        Args:
            problem_type: 'classification' or 'regression'
        
        Returns:
            str: Primary metric name (e.g., 'accuracy', 'r2')
        """
        if problem_type == 'classification':
            return self.config.get('hpt', 'HPT_PRIMARY_METRIC_CLASSIFICATION', fallback='accuracy')
        elif problem_type == 'regression':
            return self.config.get('hpt', 'HPT_PRIMARY_METRIC_REGRESSION', fallback='r2')
        else:
            raise ValueError(f"Unsupported problem_type: {problem_type}")
    
    def get_workspace_root(self) -> Path:
        """Get the workspace root directory for this session"""
        return self.session_root