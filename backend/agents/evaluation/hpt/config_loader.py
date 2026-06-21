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
        Load model_config.json from the session. Checks reports/ first (canonical
        write location used by the pipeline), then falls back to session root for
        legacy or test layouts.

        Returns:
            list: Array of model entries with name, family, hp_space, priority
        """
        candidates = [
            self.session_root / "reports" / "model_config.json",
            self.session_root / "model_config.json",
        ]
        for config_path in candidates:
            if config_path.exists():
                with open(config_path, 'r') as config_file:
                    return json.load(config_file)
        raise FileNotFoundError(
            f"model_config.json not found at {self.session_root}. "
            f"Searched: {[str(path) for path in candidates]}"
        )
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata for the session. Prefers metadata_model_selection.json (always
        has the canonical problem_type set by PipelinePrep), then falls back to
        reports/metadata.json and session root metadata.json.

        metadata.json may still carry problem_type='supervised' if the normalization
        step in training_service did not run before HPT was triggered.
        """
        candidates = [
            self.session_root / "reports" / "metadata_model_selection.json",
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
        Get primary metric based on problem type from config.ini.

        Maps legacy/raw problem_type values (e.g. 'supervised') to canonical ones
        before looking up the metric, so HPT does not fail when metadata.json
        was not normalized before HPT was triggered.

        Args:
            problem_type: problem type string (canonical or legacy)

        Returns:
            str: Primary metric name (e.g., 'accuracy', 'r2')
        """
        # Normalize legacy values that metadata.json may still carry.
        legacy_to_canonical = {
            "supervised": "classification",
            "unsupervised": "unsupervised",
        }
        canonical_type = legacy_to_canonical.get(problem_type, problem_type)

        metric_map = {
            "classification": self.config.get('hpt', 'HPT_PRIMARY_METRIC_CLASSIFICATION', fallback='accuracy'),
            "regression": self.config.get('hpt', 'HPT_PRIMARY_METRIC_REGRESSION', fallback='r2'),
        }
        if canonical_type not in metric_map:
            raise ValueError(f"Unsupported problem_type: {problem_type}")
        return metric_map[canonical_type]
    
    def load_default_hp_spaces(self) -> Dict[str, Any]:
        """
        Load the built-in default hyperparameter search spaces keyed by model name.
        Used when model_config.json carries an empty hp_space (which is always the
        case for models produced by the model-selection agent).
        """
        default_path = Path(__file__).parent / "default_hp_spaces.json"
        if not default_path.exists():
            return {}
        with open(default_path, 'r') as default_file:
            return json.load(default_file)

    def get_workspace_root(self) -> Path:
        """Get the workspace root directory for this session"""
        return self.session_root