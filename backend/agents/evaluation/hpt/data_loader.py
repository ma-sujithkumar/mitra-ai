"""
Data loading and validation for Hyperparameter Tuning Agent
Creates DataBundle from CSV files
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import json
import logging

# Import MLKit components (will be available when project is set up)
try:
    from model_library.core.data_bundle import DataBundle, CommonData
except ImportError:
    # Fallback for development - define placeholder
    class DataBundle:
        def __init__(self, X_train, y_train, X_val, y_val):
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.train = CommonData(X_train, y_train)
            self.val = CommonData(X_val, y_val)
    
    class CommonData:
        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.data = X if isinstance(X, pd.DataFrame) else None
            self.label = y if isinstance(y, pd.Series) else None

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare data for hyperparameter tuning"""
    
    def __init__(self, session_id: str, config_loader):
        """
        Initialize data loader
        
        Args:
            session_id: Unique session identifier
            config_loader: ConfigLoader instance
        """
        self.session_id = session_id
        self.session_root = Path(".mitra") / session_id
        self.config_loader = config_loader
        self.metadata = config_loader.load_metadata()
        
        # Get paths
        self.train_path = self.session_root / "data/train.csv"
        self.test_path = self.session_root / "data/test.csv"  # HIDDEN during HPT
        self.metadata_path = self.session_root / "metadata.json"
    
    def load_train_data(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Load training data and separate features/target
        
        Returns:
            Tuple of (X, y, metadata)
        """
        df = pd.read_csv(self.train_path)
        target_col = self.metadata['target_col']
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in train.csv")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return X, y, self.metadata
    
    def create_validation_split(self, X: pd.DataFrame, y: pd.Series, 
                                problem_type: str, val_ratio: float = 0.2,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/validation split with appropriate strategy
        
        Args:
            X: Features DataFrame
            y: Target Series
            problem_type: 'classification' or 'regression'
            val_ratio: Validation split ratio
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        if problem_type == 'classification':
            # Use stratified split for classification to maintain class distribution
            try:
                # For multi-class, use StratifiedShuffleSplit
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=val_ratio, 
                    stratify=y, random_state=random_state
                )
                logger.info(f"Created stratified validation split: {len(X_train)} train, {len(X_val)} val")
            except ValueError as e:
                # Fallback if stratification fails (e.g., too few samples per class)
                logger.warning(f"Stratification failed: {e}. Falling back to random split.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=val_ratio, random_state=random_state
                )
        else:
            # Regression - use random split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_ratio, random_state=random_state
            )
            logger.info(f"Created random validation split: {len(X_train)} train, {len(X_val)} val")
        
        return X_train, X_val, y_train, y_val
    
    def create_databundle(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> DataBundle:
        """
        Create MLKit DataBundle from train/val splits
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        
        Returns:
            DataBundle compatible with MLKit
        """
        # Create DataBundle using MLKit's expected format
        # This is the standard MLKit DataBundle API
        return DataBundle(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
    
    def get_complete_data_for_test(self, model_name: str, best_params: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get full training data (no split) for final training with best parameters.
        Note: This should ONLY be used after hyperparameter tuning is complete.
        The test set is NEVER used during tuning.
        
        Args:
            model_name: Model name
            best_params: Best hyperparameters found
        
        Returns:
            Tuple of (X_full, y_full) - FULL training data
        """
        X, y, _ = self.load_train_data()
        return X, y
    
    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the HIDDEN test set for final evaluation.
        IMPORTANT: This should NEVER be called during hyperparameter tuning.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        df = pd.read_csv(self.test_path)
        target_col = self.metadata['target_col']
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test.csv")
        
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
        
        return X_test, y_test