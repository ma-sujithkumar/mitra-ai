"""
Overfitting analysis for Hyperparameter Tuning Agent
"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OverfittingResult:
    """Result of overfitting analysis"""
    is_overfitted: bool
    gap: float
    train_score: float
    val_score: float
    threshold: float
    penalty_applied: float
    recommended_action: str


class OverfittingAnalyzer:
    """
    Analyze and handle overfitting during hyperparameter tuning
    """
    
    def __init__(self, threshold: float = 0.10, penalty_factor: float = 0.5):
        """
        Initialize overfitting analyzer
        
        Args:
            threshold: Maximum acceptable overfitting gap
            penalty_factor: Factor to penalize overfitted trials (0-1)
        """
        self.threshold = threshold
        self.penalty_factor = penalty_factor
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, train_score: float, val_score: float) -> OverfittingResult:
        """
        Analyze overfitting between train and validation scores
        
        Args:
            train_score: Primary metric score on training set
            val_score: Primary metric score on validation set
        
        Returns:
            OverfittingResult with analysis details
        """
        gap = train_score - val_score
        is_overfitted = gap > self.threshold
        
        # Determine recommended action
        if is_overfitted:
            if gap > self.threshold * 2:
                action = "High overfitting: reduce model complexity"
            else:
                action = "Moderate overfitting: consider regularization"
        else:
            action = "Good generalization"
        
        return OverfittingResult(
            is_overfitted=is_overfitted,
            gap=gap,
            train_score=train_score,
            val_score=val_score,
            threshold=self.threshold,
            penalty_applied=self._calculate_penalty(gap) if is_overfitted else 0.0,
            recommended_action=action
        )
    
    def _calculate_penalty(self, gap: float) -> float:
        """
        Calculate penalty to apply to overfitted trials
        
        Args:
            gap: Overfitting gap
        
        Returns:
            float: Penalty value (capped at 0.5)
        """
        # Penalty increases with gap, but capped at 0.5
        penalty = gap * self.penalty_factor
        return min(penalty, 0.5)
    
    def apply_penalty(self, val_score: float, gap: float) -> float:
        """
        Apply overfitting penalty to validation score
        
        Args:
            val_score: Original validation score
            gap: Overfitting gap
        
        Returns:
            float: Penalized score
        """
        if gap <= self.threshold:
            return val_score
        
        penalty = self._calculate_penalty(gap)
        penalized_score = val_score - penalty
        
        self.logger.debug(
            f"Overfitting penalty applied: original={val_score:.4f}, "
            f"penalty={penalty:.4f}, penalized={penalized_score:.4f}"
        )
        
        return penalized_score
    
    def get_fallback_score(self, trial_results: list) -> Dict:
        """
        Get the least-overfitted trial from results (fallback when all overfit)
        
        Args:
            trial_results: List of trial results with overfitting data
        
        Returns:
            dict: The trial with the smallest overfitting gap
        """
        if not trial_results:
            return None
        
        return min(trial_results, key=lambda t: t.get('overfitting_gap', float('inf')))