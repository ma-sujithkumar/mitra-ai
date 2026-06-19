"""
Main Hyperparameter Tuning Agent
Orchestrates the entire hyperparameter tuning process
"""
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time

from .config_loader import ConfigLoader
from .data_loader import DataLoader
from .metrics import MetricsHandler
from .overfitting import OverfittingAnalyzer
from .optuna_wrapper import OptunaWrapper
from .result_writer import ResultWriter

# Import MLKit components (will be available when project is set up)
try:
    from model_library.ml_kit import MLKit
    from model_library.metrics.evaluators import compute_metrics
    from model_library.core.data_bundle import DataBundle
except ImportError:
    # Fallback for development - define placeholder functions
    def compute_metrics(y_true, y_pred, problem_type):
        return {'accuracy': 0.9, 'f1_macro': 0.85}
    
    class MLKit:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.model = None
        
        def train(self, data=None, **kwargs):
            self.model = type('Model', (), {'predict': lambda self, X: X})()
            return self.model
        
        def predict(self, X):
            return X

logger = logging.getLogger(__name__)


class HyperparameterTuningAgent:
    """
    Main agent for hyperparameter tuning using Optuna
    """
    
    def __init__(self, session_id: str, verbose: bool = False):
        """
        Initialize the Hyperparameter Tuning Agent
        
        Args:
            session_id: Unique session identifier
            verbose: Enable verbose logging
        """
        self.session_id = session_id
        self.verbose = verbose
        self.session_root = Path(".mitra") / session_id
        self.start_time = None
        self.end_time = None
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Initializing Hyperparameter Tuning Agent for session: {session_id}")
        
        # Load configurations
        self.config_loader = ConfigLoader(session_id)
        self.hpt_config = self.config_loader.get_hpt_config()
        
        # Load session artifacts
        self.metadata = self.config_loader.load_metadata()
        self.model_config = self.config_loader.load_model_config()
        self.model_config_sorted = sorted(self.model_config, 
                                         key=lambda x: x.get('priority', 999))
        
        # Initialize data loader
        self.data_loader = DataLoader(session_id, self.config_loader)
        
        # Initialize metrics handler
        self.problem_type = self.metadata['problem_type']
        self.primary_metric = self.config_loader.get_primary_metric(self.problem_type)
        self.metrics_handler = MetricsHandler(self.problem_type, self.primary_metric)
        
        # Initialize overfitting analyzer
        self.overfitting_threshold = float(self.hpt_config.get('OVERFITTING_GAP_THRESHOLD', 0.10))
        self.overfitting_analyzer = OverfittingAnalyzer(threshold=self.overfitting_threshold)
        
        # Initialize result writer
        self.result_writer = ResultWriter(session_id, self.hpt_config)
        
        # Store results
        self.results = []
        self.failed_models = []
        
        self.logger.info(f"Agent initialized. Problem type: {self.problem_type}, "
                        f"Primary metric: {self.primary_metric}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logger
        self.logger = logging.getLogger("HyperparameterTuningAgent")
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Set level
        level = logging.DEBUG if self.verbose else logging.INFO
        self.logger.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler
        self.logger.addHandler(console_handler)
        
        # Also log to file
        log_dir = self.session_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "hpt_agent.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _create_train_fn(self, model_entry: Dict[str, Any], data_bundle: DataBundle):
        """
        Create a training function for Optuna objective
        
        Args:
            model_entry: Model configuration from model_config.json
            data_bundle: DataBundle with train/val splits
        
        Returns:
            Callable: Training function that accepts hyperparameters
        """
        model_name = model_entry['name']
        model_family = model_entry.get('family', 'unknown')
        
        def train_fn(hp: Dict[str, Any]) -> Dict[str, Any]:
            """
            Train the model with given hyperparameters and return metrics
            
            Args:
                hp: Hyperparameters for this trial
            
            Returns:
                dict: Training results with metrics
            """
            try:
                # Initialize MLKit model with config overrides
                # The exact API may need adjustment based on MLKit implementation
                mlkit = MLKit(model_name=model_name)
                
                # Train with hyperparameters (override config defaults)
                mlkit.train(data=data_bundle, **hp)
                
                # Get predictions on validation set
                y_val_pred = mlkit.predict(data_bundle.X_val)
                
                # Get predictions on training set (for overfitting detection)
                y_train_pred = mlkit.predict(data_bundle.X_train)
                
                # Compute metrics
                val_metrics = compute_metrics(data_bundle.y_val, y_val_pred, self.problem_type)
                train_metrics = compute_metrics(data_bundle.y_train, y_train_pred, self.problem_type)
                
                return {
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'y_train_pred': y_train_pred,
                    'y_val_pred': y_val_pred
                }
                
            except Exception as e:
                self.logger.error(f"Training failed for {model_name} with hp={hp}: {e}")
                raise
        
        return train_fn
    
    def tune_model(self, model_entry: Dict[str, Any], data_bundle: DataBundle) -> Optional[Dict[str, Any]]:
        """
        Perform hyperparameter tuning for a single model
        
        Args:
            model_entry: Model configuration from model_config.json
            data_bundle: DataBundle with train/val split
        
        Returns:
            dict: Tuning results for this model, or None if failed
        """
        model_name = model_entry['name']
        hp_space = model_entry.get('hp_space', {})
        
        if not hp_space:
            self.logger.warning(f"No hp_space defined for {model_name}, skipping")
            return None
        
        n_trials = int(self.hpt_config.get('MAX_HPT_TRIALS', 5))
        
        self.logger.info(f"Starting tuning for {model_name} with {n_trials} trials")
        self.logger.debug(f"HP Space: {hp_space}")
        
        # Create training function
        train_fn = self._create_train_fn(model_entry, data_bundle)
        
        # Create Optuna wrapper
        optuna_wrapper = OptunaWrapper(
            model_name=model_name,
            hp_space=hp_space,
            config={
                **self.hpt_config,
                'primary_metric': self.primary_metric,
                'session_id': self.session_id
            },
            objective_fn=train_fn
        )
        
        # Run optimization
        start_time = time.time()
        optuna_result = optuna_wrapper.run_optimization()
        tuning_time = time.time() - start_time
        
        if optuna_result is None:
            self.logger.error(f"Tuning failed for {model_name}")
            return None

        selected_trial = optuna_result['selected_trial']

        # Compute per-parameter sensitivity from all trial history
        hyperparam_sensitivity = optuna_wrapper.compute_param_sensitivity(
            trial_results=optuna_result['all_trials'],
            primary_metric=self.primary_metric,
        )

        # Build result entry
        result_entry = {
            'name': model_name,
            'model_class': model_entry.get('family', 'unknown'),
            'family': model_entry.get('family', 'unknown'),
            'priority': model_entry.get('priority', 999),
            'best_hyperparameters': selected_trial['hyperparameters'],
            'val_metrics': selected_trial['val_metrics'],
            'train_metrics': selected_trial['train_metrics'],
            'overfitting': {
                'is_overfitted': selected_trial['is_overfitted'],
                'gap': selected_trial['overfitting_gap'],
                'train_vs_cv_gap': None  # Not using CV in this version
            },
            'complexity': self._estimate_complexity(model_entry, selected_trial['hyperparameters']),
            'hyperparam_sensitivity': hyperparam_sensitivity,
            'n_trials': optuna_result['n_trials_run'],
            'n_successful_trials': optuna_result['n_successful_trials'],
            'best_trial_number': selected_trial['trial_number'],
            'optuna_study_name': optuna_result['study'].study_name,
            'selection_method': optuna_result['selection_method'],
            'tuning_time_seconds': tuning_time
        }
        
        self.logger.info(f"Completed tuning for {model_name}: "
                        f"val_{self.primary_metric}={selected_trial['val_score']:.4f}, "
                        f"overfitting_gap={selected_trial['overfitting_gap']:.4f}")
        
        return result_entry
    
    def _estimate_complexity(self, model_entry: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate model complexity based on hyperparameters
        
        Args:
            model_entry: Model configuration
            hp: Hyperparameters
        
        Returns:
            dict: Complexity metrics
        """
        complexity = {
            'n_params': len(hp),
            'depth': hp.get('max_depth', hp.get('num_layers', 0)),
            'family_rank': model_entry.get('priority', 999)
        }
        
        # Add family-specific complexity metrics
        family = model_entry.get('family', '')
        if family == 'xgboost' or family == 'lightgbm':
            complexity['n_estimators'] = hp.get('n_estimators', 0)
            complexity['max_leaves'] = hp.get('max_leaves', 0)
        elif family == 'neural_network':
            complexity['neurons'] = hp.get('neurons', 0)
            complexity['num_layers'] = hp.get('num_layers', 0)
        elif family == 'random_forest':
            complexity['n_estimators'] = hp.get('n_estimators', 0)
            complexity['max_depth'] = hp.get('max_depth', 0)
        
        return complexity
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the hyperparameter tuning pipeline
        
        Returns:
            List of tuning results for all models
        """
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Starting Hyperparameter Tuning Agent")
        self.logger.info(f"Session: {self.session_id}")
        self.logger.info(f"Models to tune: {len(self.model_config_sorted)}")
        self.logger.info(f"Primary metric: {self.primary_metric}")
        self.logger.info(f"Overfitting threshold: {self.overfitting_threshold}")
        self.logger.info("=" * 60)
        
        # Load data and create validation split
        self.logger.info("Loading training data...")
        X, y, metadata = self.data_loader.load_train_data()
        
        val_ratio = float(self.hpt_config.get('VAL_SPLIT_RATIO', 0.2))
        self.logger.info(f"Creating validation split with ratio {val_ratio}")
        
        X_train, X_val, y_train, y_val = self.data_loader.create_validation_split(
            X, y, self.problem_type, val_ratio, 
            random_state=self.hpt_config.get('OPTUNA_SEED', 42)
        )
        
        self.logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
        
        # Create DataBundle
        data_bundle = self.data_loader.create_databundle(X_train, y_train, X_val, y_val)
        
        # Tune each model (sorted by priority)
        self.logger.info("Starting model tuning...")
        
        for idx, model_entry in enumerate(self.model_config_sorted, 1):
            model_name = model_entry.get('name', f'model_{idx}')
            priority = model_entry.get('priority', 999)
            
            self.logger.info(f"Tuning model {idx}/{len(self.model_config_sorted)}: {model_name} (priority={priority})")
            
            try:
                result = self.tune_model(model_entry, data_bundle)
                if result:
                    self.results.append(result)
                    self.logger.info(f"✓ {model_name} completed successfully")
                else:
                    self.failed_models.append(model_name)
                    self.logger.warning(f"✗ {model_name} failed")
            except Exception as e:
                self.logger.error(f"Error tuning {model_name}: {e}", exc_info=True)
                self.failed_models.append(model_name)
        
        # Save results
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info(f"Tuning complete in {total_time:.2f} seconds")
        self.logger.info(f"Successful: {len(self.results)} models")
        self.logger.info(f"Failed: {len(self.failed_models)} models")
        
        if self.failed_models:
            self.logger.warning(f"Failed models: {', '.join(self.failed_models)}")
        
        # Write results
        self.result_writer.write_results(self.results, {
            'total_time': total_time,
            'successful': len(self.results),
            'failed': len(self.failed_models),
            'failed_models': self.failed_models
        })
        
        # Update metadata with tuning completion info
        self._update_metadata(total_time)
        
        self.logger.info(f"Results saved to: {self.session_root / 'hpt_results.json'}")
        self.logger.info("=" * 60)
        
        return self.results
    
    def _update_metadata(self, total_time: float):
        """Update metadata.json with tuning completion info"""
        metadata_path = self.session_root / "metadata.json"
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['hpt_completed'] = True
            metadata['hpt_completion_time'] = time.time()
            metadata['hpt_total_time_seconds'] = total_time
            metadata['hpt_models_tuned'] = len(self.results)
            metadata['hpt_models_failed'] = len(self.failed_models)
            metadata['hpt_primary_metric'] = self.primary_metric
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Updated metadata at {metadata_path}")
        except Exception as e:
            self.logger.warning(f"Could not update metadata: {e}")