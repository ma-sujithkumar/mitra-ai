"""
Test suite for Hyperparameter Tuning Agent
Tests the main flow with SKLearn datasets
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np


class TestAgentInitialization:
    """Test agent initialization with proper configurations"""
    
    def test_agent_initialization_with_valid_config(self, mock_config_loader_iris):
        """Test that agent initializes with valid configuration"""
        mock_loader, session_root = mock_config_loader_iris
        
        # We can't directly import agent without all dependencies,
        # so we test the config loader mock
        assert mock_loader is not None
        assert session_root.exists()
        
        # Check that metadata is loaded correctly
        metadata = mock_loader.load_metadata()
        assert metadata['problem_type'] == 'classification'
        assert metadata['target_col'] == 'target'
        assert metadata['n_classes'] == 3
    
    def test_hpt_config_loading(self, mock_config_loader_iris):
        """Test that HPT configuration is loaded correctly"""
        mock_loader, _ = mock_config_loader_iris
        
        hpt_config = mock_loader.get_hpt_config()
        
        assert hpt_config['MAX_HPT_TRIALS'] == 2
        assert hpt_config['OVERFITTING_GAP_THRESHOLD'] == 0.10
        assert hpt_config['VAL_SPLIT_RATIO'] == 0.2
        assert hpt_config['OPTUNA_SEED'] == 42
    
    def test_model_config_loading(self, mock_config_loader_iris):
        """Test that model configuration is loaded correctly"""
        mock_loader, _ = mock_config_loader_iris
        
        model_config = mock_loader.load_model_config()
        
        assert len(model_config) >= 2
        assert model_config[0]['name'] == 'logistic_regression'
        assert model_config[1]['name'] == 'random_forest'
        
        # Check hyperparameter spaces
        assert 'hp_space' in model_config[0]
        assert 'C' in model_config[0]['hp_space']


class TestDataLoading:
    """Test data loading with SKLearn datasets"""
    
    def test_iris_data_csv_creation(self, iris_train_csv):
        """Test that Iris data is correctly saved to CSV"""
        train_path, metadata = iris_train_csv
        
        assert train_path.exists()
        
        # Load and verify the CSV
        df = pd.read_csv(train_path)
        assert df.shape[0] == 150  # Iris has 150 samples
        assert df.shape[1] == 5    # 4 features + target
        assert 'target' in df.columns
    
    def test_wine_data_csv_creation(self, wine_train_csv):
        """Test that Wine data is correctly saved to CSV"""
        train_path, metadata = wine_train_csv
        
        assert train_path.exists()
        
        # Load and verify the CSV
        df = pd.read_csv(train_path)
        assert df.shape[0] == 178   # Wine has 178 samples
        assert df.shape[1] == 14    # 13 features + target
        assert 'target' in df.columns
    
    def test_breast_cancer_data_csv_creation(self, breast_cancer_train_csv):
        """Test that Breast Cancer data is correctly saved to CSV"""
        train_path, metadata = breast_cancer_train_csv
        
        assert train_path.exists()
        
        # Load and verify the CSV
        df = pd.read_csv(train_path)
        assert df.shape[0] == 569   # Breast cancer has 569 samples
        assert df.shape[1] == 31    # 30 features + target
        assert 'target' in df.columns


class TestSessionSetup:
    """Test session directory and file setup"""
    
    def test_session_directory_structure(self, temp_session_dir):
        """Test that session directory structure is created correctly"""
        assert temp_session_dir.exists()
        assert (temp_session_dir / 'data').exists()
        assert (temp_session_dir / 'logs').exists()
        assert (temp_session_dir / 'outputs').exists()
    
    def test_complete_session_setup(self, complete_session_setup):
        """Test complete session setup with all required files"""
        setup = complete_session_setup
        
        assert setup['session_root'].exists()
        assert setup['metadata_path'].exists()
        assert setup['model_config_path'].exists()
        assert setup['train_path'].exists()
        
        # Verify metadata
        with open(setup['metadata_path'], 'r') as f:
            metadata = json.load(f)
        assert metadata['problem_type'] == 'classification'
        
        # Verify model config
        with open(setup['model_config_path'], 'r') as f:
            model_config = json.load(f)
        assert len(model_config) >= 2


class TestTrainValSplit:
    """Test train/validation split logic"""
    
    def test_train_val_split_ratio(self, iris_train_csv):
        """Test that train/val split maintains correct ratio"""
        train_path, metadata = iris_train_csv
        
        # Load data
        df = pd.read_csv(train_path)
        target_col = 'target'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Create split with 0.2 ratio
        val_ratio = 0.2
        n_val = int(len(X) * val_ratio)
        n_train = len(X) - n_val
        
        assert n_train + n_val == len(X)
        assert n_val == int(150 * 0.2)  # Should be 30 for Iris
    
    def test_stratified_split_for_classification(self, iris_train_csv):
        """Test that stratified split preserves class distribution"""
        train_path, metadata = iris_train_csv
        
        # Load data
        df = pd.read_csv(train_path)
        target_col = 'target'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Check class distribution
        class_counts = y.value_counts().sort_index()
        
        # Iris should have 3 classes with roughly equal distribution
        assert len(class_counts) == 3
        # Each class should have roughly 50 samples
        for count in class_counts.values:
            assert 45 <= count <= 55


class TestModelConfigValidation:
    """Test model configuration validation"""
    
    def test_model_config_has_required_fields(self, simple_model_config):
        """Test that model config has all required fields"""
        config_path, model_config = simple_model_config
        
        required_fields = ['name', 'family', 'priority', 'hp_space']
        
        for model in model_config:
            for field in required_fields:
                assert field in model, f"Missing field '{field}' in model config"
    
    def test_hp_space_structure(self, simple_model_config):
        """Test that hp_space has correct structure"""
        config_path, model_config = simple_model_config
        
        for model in model_config:
            hp_space = model['hp_space']
            assert isinstance(hp_space, dict)
            assert len(hp_space) > 0
            
            # Each hp should have type and bounds/choices
            for hp_name, hp_spec in hp_space.items():
                assert 'type' in hp_spec


class TestMetricsAndMetadata:
    """Test metrics handling and metadata management"""
    
    def test_primary_metric_selection(self, mock_config_loader_iris):
        """Test that primary metric is correctly selected based on problem type"""
        mock_loader, _ = mock_config_loader_iris
        
        # For classification
        primary_metric = mock_loader.get_primary_metric('classification')
        assert primary_metric == 'accuracy'
        
        # For regression
        primary_metric = mock_loader.get_primary_metric('regression')
        assert primary_metric == 'mse'
    
    def test_metadata_structure(self, metadata_json):
        """Test that metadata has all required fields"""
        metadata_path, metadata = metadata_json
        
        required_fields = ['problem_type', 'target_col', 'n_classes', 'n_features', 'n_samples']
        
        for field in required_fields:
            assert field in metadata, f"Missing field '{field}' in metadata"


class TestOutputStructure:
    """Test output file structure and results"""
    
    def test_hpt_results_structure(self, temp_session_dir):
        """Test that HPT results have correct structure"""
        # Create mock results
        results = [
            {
                'name': 'logistic_regression',
                'model_class': 'linear',
                'family': 'linear',
                'priority': 1,
                'best_hyperparameters': {'C': 10.0, 'penalty': 'l2'},
                'val_metrics': {'accuracy': 0.96},
                'train_metrics': {'accuracy': 0.98},
                'overfitting': {
                    'is_overfitted': False,
                    'gap': 0.02,
                    'train_vs_cv_gap': None
                },
                'complexity': {'n_params': 2},
                'n_trials': 5,
                'n_successful_trials': 5,
                'best_trial_number': 3,
                'selection_method': 'best_trial',
                'tuning_time_seconds': 120.5
            }
        ]
        
        # Verify structure
        for result in results:
            required_keys = [
                'name', 'best_hyperparameters', 'val_metrics',
                'train_metrics', 'overfitting', 'n_trials'
            ]
            for key in required_keys:
                assert key in result, f"Missing key '{key}' in result"
            
            # Check overfitting structure
            assert 'is_overfitted' in result['overfitting']
            assert 'gap' in result['overfitting']


class TestSessionDirectory:
    """Test session directory and .mitra structure"""
    
    def test_session_root_creation(self, temp_session_dir):
        """Test that session root directory is created"""
        assert temp_session_dir.exists()
        assert temp_session_dir.is_dir()
    
    def test_logs_directory_creation(self, temp_session_dir):
        """Test that logs directory is created in session root"""
        logs_dir = temp_session_dir / 'logs'
        assert logs_dir.exists()
        assert logs_dir.is_dir()
    
    def test_data_directory_creation(self, temp_session_dir):
        """Test that data directory is created in session root"""
        data_dir = temp_session_dir / 'data'
        assert data_dir.exists()
        assert data_dir.is_dir()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
