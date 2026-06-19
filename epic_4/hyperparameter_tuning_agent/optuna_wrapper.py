"""
Optuna wrapper for hyperparameter tuning with overfitting prevention
"""
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner
from optuna.study import Study
import logging
from typing import Dict, Any, Optional, Callable, List
import numpy as np

logger = logging.getLogger(__name__)


class OptunaWrapper:
    """Wrapper for Optuna hyperparameter optimization with overfitting prevention"""
    
    def __init__(self, 
                 model_name: str,
                 hp_space: Dict[str, Any],
                 config: Dict[str, Any],
                 objective_fn: Callable):
        """
        Initialize Optuna wrapper
        
        Args:
            model_name: Name of the model being tuned
            hp_space: Hyperparameter search space from model_config.json
            config: Configuration dictionary with HPT settings
            objective_fn: Function that trains model and returns metrics
        """
        self.model_name = model_name
        self.hp_space = hp_space
        self.config = config
        self.objective_fn = objective_fn
        self.n_trials = config.get('MAX_HPT_TRIALS', 5)
        self.seed = config.get('OPTUNA_SEED', 42)
        self.sampler_type = config.get('OPTUNA_SAMPLER', 'TPE')
        self.logger = logging.getLogger(__name__)
        
        # Overfitting threshold from config
        self.overfitting_threshold = config.get('OVERFITTING_GAP_THRESHOLD', 0.10)
        
        # Track trials
        self.trial_results = []
        self.best_non_overfitted_trial = None
        self.best_non_overfitted_score = -float('inf')
        self.least_overfitted_trial = None
        self.least_overfitted_gap = float('inf')
    
    def create_study(self) -> Study:
        """
        Create Optuna study with configured sampler and pruner
        
        Returns:
            optuna.Study: Configured study object
        """
        # Choose sampler
        if self.sampler_type.upper() == 'TPE':
            sampler = TPESampler(seed=self.seed)
        elif self.sampler_type.upper() == 'RANDOM':
            sampler = RandomSampler(seed=self.seed)
        else:
            self.logger.warning(f"Unknown sampler {self.sampler_type}, using TPE")
            sampler = TPESampler(seed=self.seed)
        
        # Pruner for early stopping of unpromising trials
        pruner = MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=10,
            interval_steps=1
        )
        
        # Create study
        study = optuna.create_study(
            direction='maximize',  # Always maximize the primary metric
            sampler=sampler,
            pruner=pruner,
            study_name=f"hpt_{self.model_name}_{self.config.get('session_id', 'default')}"
        )
        
        return study
    
    def sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters from the search space using Optuna's suggest methods
        
        Args:
            trial: Optuna trial object
        
        Returns:
            dict: Sampled hyperparameters
        """
        params = {}
        
        for param_name, param_spec in self.hp_space.items():
            param_type = param_spec.get('type')
            
            try:
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_spec['low'],
                        param_spec['high'],
                        step=param_spec.get('step', 1)
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_spec['low'],
                        param_spec['high'],
                        log=param_spec.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_spec['choices']
                    )
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
            except Exception as e:
                self.logger.error(f"Error sampling {param_name}: {e}")
                raise
        
        return params
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
        
        Returns:
            float: Primary metric score (maximized)
        """
        # 1. Sample hyperparameters
        hp = self.sample_hyperparameters(trial)
        
        # 2. Train and evaluate
        try:
            result = self.objective_fn(hp)
        except Exception as e:
            self.logger.error(f"Training failed for trial {trial.number}: {e}")
            return -float('inf')
        
        # 3. Extract metrics
        train_metrics = result.get('train_metrics', {})
        val_metrics = result.get('val_metrics', {})
        primary_metric = self.config.get('primary_metric', 'accuracy')
        
        train_score = train_metrics.get(primary_metric, 0.0)
        val_score = val_metrics.get(primary_metric, 0.0)
        
        # 4. Compute overfitting gap
        gap = train_score - val_score
        
        # 5. Store trial data for later analysis
        trial_data = {
            'trial_number': trial.number,
            'hyperparameters': hp,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_score': train_score,
            'val_score': val_score,
            'overfitting_gap': gap,
            'is_overfitted': gap > self.overfitting_threshold
        }
        self.trial_results.append(trial_data)
        
        # 6. Track best non-overfitted trial
        if not trial_data['is_overfitted']:
            if val_score > self.best_non_overfitted_score:
                self.best_non_overfitted_score = val_score
                self.best_non_overfitted_trial = trial_data
        
        # 7. Track least overfitted trial (fallback)
        if gap < self.least_overfitted_gap:
            self.least_overfitted_gap = gap
            self.least_overfitted_trial = trial_data
        
        # 8. Apply overfitting penalty if needed
        if trial_data['is_overfitted']:
            penalty = min(gap, 0.5)  # Cap penalty at 0.5
            penalized_score = val_score - penalty
            self.logger.debug(
                f"Trial {trial.number}: Overfitted (gap={gap:.3f}), "
                f"penalty={penalty:.3f}, score={penalized_score:.3f}"
            )
            return penalized_score
        
        return val_score
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run the full optimization process
        
        Returns:
            dict: Optimization results including best trial and all trials
        """
        # Create study
        study = self.create_study()
        
        self.logger.info(f"Starting optimization for {self.model_name} with {self.n_trials} trials")
        
        # Run optimization
        try:
            study.optimize(self.objective, n_trials=self.n_trials)
        except Exception as e:
            self.logger.error(f"Optuna optimization failed: {e}")
            return None
        
        # Select best trial (prefer non-overfitted)
        selected_trial = None
        selection_method = None
        
        if self.best_non_overfitted_trial is not None:
            selected_trial = self.best_non_overfitted_trial
            selection_method = 'best_non_overfitted'
            self.logger.info(f"Selected best non-overfitted trial: {selected_trial['trial_number']}")
        elif self.least_overfitted_trial is not None:
            selected_trial = self.least_overfitted_trial
            selection_method = 'fallback_least_overfitted'
            self.logger.warning(f"All trials overfitted. Selected least-overfitted trial: {selected_trial['trial_number']}")
        else:
            self.logger.error(f"No successful trials for {self.model_name}")
            return None
        
        # Build result
        result = {
            'study': study,
            'selected_trial': selected_trial,
            'selection_method': selection_method,
            'all_trials': self.trial_results,
            'best_trial_number': selected_trial['trial_number'],
            'best_hyperparameters': selected_trial['hyperparameters'],
            'best_val_metrics': selected_trial['val_metrics'],
            'best_train_metrics': selected_trial['train_metrics'],
            'best_overfitting_gap': selected_trial['overfitting_gap'],
            'best_is_overfitted': selected_trial['is_overfitted'],
            'n_trials_run': len(self.trial_results),
            'n_successful_trials': sum(1 for t in self.trial_results if not t.get('is_overfitted', True))
        }
        
        return result