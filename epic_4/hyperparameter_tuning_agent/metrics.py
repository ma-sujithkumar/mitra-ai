"""
Metrics handling for Hyperparameter Tuning Agent
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MetricsHandler:
    """Handle metrics computation and resolution"""
    
    # Supported metric keys
    CLASSIFICATION_METRICS = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    REGRESSION_METRICS = ['mse', 'rmse', 'mae', 'r2']
    
    def __init__(self, problem_type: str, primary_metric: str):
        """
        Initialize metrics handler
        
        Args:
            problem_type: 'classification' or 'regression'
            primary_metric: Primary metric to optimize
        """
        self.problem_type = problem_type
        self.primary_metric = primary_metric
        
        # Validate primary metric
        self._validate_primary_metric()
    
    def _validate_primary_metric(self):
        """Validate that the primary metric is appropriate for the problem type"""
        if self.problem_type == 'classification':
            valid_metrics = self.CLASSIFICATION_METRICS
        else:
            valid_metrics = self.REGRESSION_METRICS
        
        if self.primary_metric not in valid_metrics:
            raise ValueError(
                f"Primary metric '{self.primary_metric}' not valid for {self.problem_type}. "
                f"Valid metrics: {valid_metrics}"
            )
    
    def get_primary_score(self, metrics: Dict[str, float]) -> float:
        """Extract the primary metric score from metrics dictionary"""
        return metrics.get(self.primary_metric, 0.0)
    
    def compute_overfitting_gap(self, train_metrics: Dict[str, float], 
                               val_metrics: Dict[str, float]) -> float:
        """
        Compute overfitting gap using primary metric
        
        Args:
            train_metrics: Metrics on training set
            val_metrics: Metrics on validation set
        
        Returns:
            float: Gap (train_score - val_score)
        """
        train_score = self.get_primary_score(train_metrics)
        val_score = self.get_primary_score(val_metrics)
        return train_score - val_score
    
    def is_overfitted(self, train_metrics: Dict[str, float], 
                     val_metrics: Dict[str, float],
                     threshold: float = 0.10) -> bool:
        """
        Check if model is overfitted based on primary metric gap
        
        Args:
            train_metrics: Metrics on training set
            val_metrics: Metrics on validation set
            threshold: Maximum acceptable gap
        
        Returns:
            bool: True if overfitted
        """
        gap = self.compute_overfitting_gap(train_metrics, val_metrics)
        return gap > threshold
    
    def validate_metrics_complete(self, metrics: Dict[str, float]) -> bool:
        """
        Validate that all required metrics are present
        
        Args:
            metrics: Metrics dictionary
        
        Returns:
            bool: True if all required metrics present
        """
        if self.problem_type == 'classification':
            required = self.CLASSIFICATION_METRICS
        else:
            required = self.REGRESSION_METRICS
        
        return all(m in metrics for m in required)