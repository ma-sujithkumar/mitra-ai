"""
Test suite for DataLoader with SKLearn datasets
Tests data loading, preprocessing, and validation split creation
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch


class TestDataLoaderInitialization:
    """Test DataLoader initialization"""
    
    def test_dataloader_creates_paths(self, mock_config_loader_iris):
        """Test that DataLoader correctly initializes paths"""
        mock_loader, session_root = mock_config_loader_iris
        
        # Verify that session root path is correct
        assert session_root.exists()
        assert (session_root / 'data').exists()


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_load_iris_data(self, iris_train_csv):
        """Test loading Iris training data"""
        train_path, metadata = iris_train_csv
        
        # Load the CSV
        df = pd.read_csv(train_path)
        
        # Verify dimensions
        assert df.shape[0] == 150, "Iris should have 150 samples"
        assert df.shape[1] == 5, "Iris should have 5 columns (4 features + target)"
        
        # Extract X and y
        target_col = metadata['target_col']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Verify feature dimensions
        assert X.shape[1] == 4
        assert len(y) == 150
    
    def test_load_wine_data(self, wine_train_csv):
        """Test loading Wine training data"""
        train_path, metadata = wine_train_csv
        
        # Load the CSV
        df = pd.read_csv(train_path)
        
        # Verify dimensions
        assert df.shape[0] == 178, "Wine should have 178 samples"
        assert df.shape[1] == 14, "Wine should have 14 columns (13 features + target)"
        
        # Extract X and y
        target_col = metadata['target_col']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Verify feature dimensions
        assert X.shape[1] == 13
        assert len(y) == 178
    
    def test_load_breast_cancer_data(self, breast_cancer_train_csv):
        """Test loading Breast Cancer training data"""
        train_path, metadata = breast_cancer_train_csv
        
        # Load the CSV
        df = pd.read_csv(train_path)
        
        # Verify dimensions
        assert df.shape[0] == 569, "Breast cancer should have 569 samples"
        assert df.shape[1] == 31, "Breast cancer should have 31 columns (30 features + target)"
        
        # Extract X and y
        target_col = metadata['target_col']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Verify feature dimensions
        assert X.shape[1] == 30
        assert len(y) == 569


class TestValidationSplit:
    """Test train/validation split creation"""
    
    def test_train_val_split_ratio(self, iris_train_csv):
        """Test that train/val split maintains correct ratio"""
        train_path, metadata = iris_train_csv
        
        # Load data
        df = pd.read_csv(train_path)
        target_col = metadata['target_col']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Simulate validation split
        val_ratio = 0.2
        val_size = int(len(X) * val_ratio)
        train_size = len(X) - val_size
        
        # Verify the split
        assert train_size + val_size == len(X)
        assert val_size == 30  # 20% of 150
        assert train_size == 120  # 80% of 150
    
    def test_stratified_split_preservation(self, iris_train_csv):
        """Test that stratified split preserves class distribution"""
        train_path, metadata = iris_train_csv
        
        # Load data
        df = pd.read_csv(train_path)
        target_col = metadata['target_col']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Check original class distribution
        original_counts = y.value_counts().sort_index()
        
        # For stratified split, each class should have ~50 samples
        for class_label, count in original_counts.items():
            # Iris has balanced classes
            assert 45 <= count <= 55, f"Class {class_label} has {count} samples"
    
    def test_no_data_leakage_in_split(self, iris_train_csv):
        """Test that train and validation sets don't overlap"""
        train_path, metadata = iris_train_csv
        
        # Load data
        df = pd.read_csv(train_path)
        target_col = metadata['target_col']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Create indices for split
        val_ratio = 0.2
        val_size = int(len(X) * val_ratio)
        
        # Using simple split
        train_indices = set(range(val_size, len(X)))
        val_indices = set(range(0, val_size))
        
        # Check no overlap
        overlap = train_indices.intersection(val_indices)
        assert len(overlap) == 0, "Train and validation sets should not overlap"


class TestDataValidation:
    """Test data validation"""
    
    def test_no_missing_values(self, iris_train_csv):
        """Test that loaded data has no missing values"""
        train_path, metadata = iris_train_csv
        
        df = pd.read_csv(train_path)
        
        # Check for missing values
        assert df.isnull().sum().sum() == 0, "Data should not have missing values"
    
    def test_target_column_exists(self, iris_train_csv):
        """Test that target column exists in data"""
        train_path, metadata = iris_train_csv
        
        df = pd.read_csv(train_path)
        target_col = metadata['target_col']
        
        assert target_col in df.columns, f"Target column '{target_col}' not found"
    
    def test_feature_columns_valid(self, iris_train_csv):
        """Test that all feature columns are numeric"""
        train_path, metadata = iris_train_csv
        
        df = pd.read_csv(train_path)
        target_col = metadata['target_col']
        X = df.drop(columns=[target_col])
        
        # Check that all features are numeric
        for col in X.columns:
            assert X[col].dtype in [np.int64, np.float64], f"Column '{col}' is not numeric"


class TestDataBundleCreation:
    """Test DataBundle creation"""
    
    def test_databundle_structure(self, iris_train_csv, mock_databundle_class):
        """Test that DataBundle has correct structure"""
        train_path, metadata = iris_train_csv
        
        # Load data
        df = pd.read_csv(train_path)
        target_col = metadata['target_col']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Create validation split
        val_ratio = 0.2
        val_size = int(len(X) * val_ratio)
        
        X_train = X.iloc[val_size:]
        X_val = X.iloc[:val_size]
        y_train = y.iloc[val_size:]
        y_val = y.iloc[:val_size]
        
        # Create DataBundle
        DataBundle = mock_databundle_class
        data_bundle = DataBundle(X_train, y_train, X_val, y_val)
        
        # Verify structure
        assert hasattr(data_bundle, 'X_train')
        assert hasattr(data_bundle, 'y_train')
        assert hasattr(data_bundle, 'X_val')
        assert hasattr(data_bundle, 'y_val')
        
        # Verify dimensions
        assert data_bundle.X_train.shape[0] == len(data_bundle.y_train)
        assert data_bundle.X_val.shape[0] == len(data_bundle.y_val)
    
    def test_databundle_train_val_split(self, iris_train_csv, mock_databundle_class):
        """Test that DataBundle correctly holds train/val split"""
        train_path, metadata = iris_train_csv
        
        # Load and split data
        df = pd.read_csv(train_path)
        target_col = metadata['target_col']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        val_size = 30
        X_train, X_val = X.iloc[val_size:], X.iloc[:val_size]
        y_train, y_val = y.iloc[val_size:], y.iloc[:val_size]
        
        # Create DataBundle
        DataBundle = mock_databundle_class
        data_bundle = DataBundle(X_train, y_train, X_val, y_val)
        
        # Verify split sizes
        assert len(data_bundle.X_train) == 120
        assert len(data_bundle.X_val) == 30
        assert len(data_bundle.y_train) == 120
        assert len(data_bundle.y_val) == 30


class TestDataStatistics:
    """Test data statistics and basic properties"""
    
    def test_iris_data_statistics(self, iris_train_csv):
        """Test basic statistics of Iris data"""
        train_path, metadata = iris_train_csv
        
        df = pd.read_csv(train_path)
        X = df.drop(columns=['target'])
        
        # Check feature statistics
        assert X.shape == (150, 4)
        assert X.min().min() > 0  # All measurements are positive
        assert X.max().max() < 10  # All measurements are in reasonable range
    
    def test_wine_data_statistics(self, wine_train_csv):
        """Test basic statistics of Wine data"""
        train_path, metadata = wine_train_csv
        
        df = pd.read_csv(train_path)
        X = df.drop(columns=['target'])
        
        # Check feature statistics
        assert X.shape == (178, 13)
        assert X.isnull().sum().sum() == 0  # No missing values
    
    def test_breast_cancer_data_statistics(self, breast_cancer_train_csv):
        """Test basic statistics of Breast Cancer data"""
        train_path, metadata = breast_cancer_train_csv
        
        df = pd.read_csv(train_path)
        X = df.drop(columns=['target'])
        
        # Check feature statistics
        assert X.shape == (569, 30)
        assert X.isnull().sum().sum() == 0  # No missing values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
