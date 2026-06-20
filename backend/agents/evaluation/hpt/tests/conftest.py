"""
Pytest configuration and fixtures for Hyperparameter Tuning Agent tests
Uses SKLearn datasets for testing
"""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler


class SklearnDatasetFixture:
    """Fixture to provide SKLearn datasets"""
    
    @staticmethod
    def get_iris_data() -> tuple:
        """Get Iris dataset as pandas DataFrame and Series"""
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        metadata = {
            'problem_type': 'classification',
            'target_col': 'target',
            'n_classes': len(np.unique(data.target)),
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        return X, y, metadata
    
    @staticmethod
    def get_wine_data() -> tuple:
        """Get Wine dataset as pandas DataFrame and Series"""
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        metadata = {
            'problem_type': 'classification',
            'target_col': 'target',
            'n_classes': len(np.unique(data.target)),
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        return X, y, metadata
    
    @staticmethod
    def get_breast_cancer_data() -> tuple:
        """Get Breast Cancer dataset as pandas DataFrame and Series"""
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        metadata = {
            'problem_type': 'classification',
            'target_col': 'target',
            'n_classes': len(np.unique(data.target)),
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        return X, y, metadata


@pytest.fixture(scope='function')
def temp_session_dir():
    """Create a temporary session directory structure"""
    temp_dir = tempfile.mkdtemp(prefix='hpt_test_session_')
    session_root = Path(temp_dir)
    
    # Create subdirectories
    (session_root / 'data').mkdir(parents=True, exist_ok=True)
    (session_root / 'logs').mkdir(parents=True, exist_ok=True)
    (session_root / 'outputs').mkdir(parents=True, exist_ok=True)
    
    yield session_root
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope='function')
def iris_train_csv(temp_session_dir):
    """Create Iris training data CSV file"""
    X, y, metadata = SklearnDatasetFixture.get_iris_data()
    
    # Combine features and target
    train_data = X.copy()
    train_data['target'] = y
    
    # Save to CSV
    train_path = temp_session_dir / 'data' / 'train.csv'
    train_data.to_csv(train_path, index=False)
    
    return train_path, metadata


@pytest.fixture(scope='function')
def wine_train_csv(temp_session_dir):
    """Create Wine training data CSV file"""
    X, y, metadata = SklearnDatasetFixture.get_wine_data()
    
    # Combine features and target
    train_data = X.copy()
    train_data['target'] = y
    
    # Save to CSV
    train_path = temp_session_dir / 'data' / 'train.csv'
    train_data.to_csv(train_path, index=False)
    
    return train_path, metadata


@pytest.fixture(scope='function')
def breast_cancer_train_csv(temp_session_dir):
    """Create Breast Cancer training data CSV file"""
    X, y, metadata = SklearnDatasetFixture.get_breast_cancer_data()
    
    # Combine features and target
    train_data = X.copy()
    train_data['target'] = y
    
    # Save to CSV
    train_path = temp_session_dir / 'data' / 'train.csv'
    train_data.to_csv(train_path, index=False)
    
    return train_path, metadata


@pytest.fixture(scope='function')
def metadata_json(temp_session_dir):
    """Create metadata.json file"""
    metadata = {
        'problem_type': 'classification',
        'target_col': 'target',
        'n_classes': 3,
        'n_features': 4,
        'n_samples': 150
    }
    
    metadata_path = temp_session_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path, metadata


@pytest.fixture(scope='function')
def simple_model_config(temp_session_dir):
    """Create a simple model_config.json for testing"""
    model_config = [
        {
            'name': 'logistic_regression',
            'family': 'linear',
            'priority': 1,
            'hp_space': {
                'C': {'type': 'float', 'low': 0.001, 'high': 100.0},
                'penalty': {'type': 'categorical', 'choices': ['l1', 'l2']}
            }
        },
        {
            'name': 'random_forest',
            'family': 'tree_ensemble',
            'priority': 2,
            'hp_space': {
                'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20}
            }
        }
    ]
    
    config_path = temp_session_dir / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    return config_path, model_config


@pytest.fixture(scope='function')
def complete_session_setup(temp_session_dir, iris_train_csv, metadata_json, simple_model_config):
    """
    Complete session setup with all required files
    Returns: (session_root, metadata_dict, model_config_list)
    """
    session_root = temp_session_dir
    
    # Unpack fixtures
    train_path, train_metadata = iris_train_csv
    metadata_path, metadata_dict = metadata_json
    model_config_path, model_config = simple_model_config
    
    return {
        'session_root': session_root,
        'metadata': metadata_dict,
        'model_config': model_config,
        'train_path': train_path,
        'metadata_path': metadata_path,
        'model_config_path': model_config_path
    }


@pytest.fixture(scope='function')
def mock_config_loader_iris(temp_session_dir, iris_train_csv, metadata_json, simple_model_config):
    """
    Create a mock ConfigLoader with iris data
    """
    # Create mock ConfigLoader class
    class MockConfigLoader:
        def __init__(self, session_root):
            self.session_root = session_root
            self.metadata_data = {
                'problem_type': 'classification',
                'target_col': 'target',
                'n_classes': 3,
                'n_features': 4,
                'n_samples': 150
            }
        
        def load_metadata(self):
            return self.metadata_data
        
        def load_model_config(self):
            model_config_path = self.session_root / 'model_config.json'
            with open(model_config_path, 'r') as f:
                return json.load(f)
        
        def get_hpt_config(self):
            return {
                'MAX_HPT_TRIALS': 2,
                'OVERFITTING_GAP_THRESHOLD': 0.10,
                'VAL_SPLIT_RATIO': 0.2,
                'HPT_N_JOBS': 1,
                'OPTUNA_SEED': 42
            }
        
        def get_path_config(self):
            return {'WORKSPACE_ROOT': str(self.session_root)}
        
        def get_python_config(self):
            return {'PYTHON': 'python'}
        
        def get_primary_metric(self, problem_type):
            if problem_type == 'classification':
                return 'accuracy'
            return 'mse'
    
    # Unpack fixtures to populate session
    train_path, train_metadata = iris_train_csv
    metadata_path, metadata_dict = metadata_json
    model_config_path, model_config = simple_model_config
    
    # Create the mock loader
    mock_loader = MockConfigLoader(temp_session_dir)
    
    return mock_loader, temp_session_dir


@pytest.fixture(scope='function')
def mock_databundle_class():
    """Create mock DataBundle class for testing"""
    class DataBundle:
        def __init__(self, X_train, y_train, X_val, y_val):
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
    
    return DataBundle


@pytest.fixture(scope='session')
def sklearn_datasets():
    """Provide access to different SKLearn datasets"""
    return {
        'iris': SklearnDatasetFixture.get_iris_data,
        'wine': SklearnDatasetFixture.get_wine_data,
        'breast_cancer': SklearnDatasetFixture.get_breast_cancer_data
    }
