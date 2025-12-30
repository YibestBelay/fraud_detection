"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import FraudDataPreprocessor


class TestFraudDataPreprocessor:
    """Test cases for FraudDataPreprocessor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'Class': np.random.randint(0, 2, n_samples)
        }
        return pd.DataFrame(data)
    
    def test_init(self):
        """Test initialization."""
        preprocessor = FraudDataPreprocessor(random_state=42)
        assert preprocessor.random_state == 42
        assert preprocessor.scaler is not None
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        preprocessor = FraudDataPreprocessor()
        
        # Test with creditcard dataset
        preprocessor.target_column = 'Class'
        X, y = preprocessor.prepare_features(sample_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == sample_data.shape[0]
        assert y.shape[0] == sample_data.shape[0]
        assert len(preprocessor.feature_columns) == 3
        assert 'Class' not in X.columns
    
    def test_split_data(self, sample_data):
        """Test stratified train-test split."""
        preprocessor = FraudDataPreprocessor(random_state=42)
        
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X, y, test_size=0.2
        )
        
        # Check shapes
        assert X_train.shape[0] == 800
        assert X_test.shape[0] == 200
        assert y_train.shape[0] == 800
        assert y_test.shape[0] == 200
        
        # Check stratification (fraud rates should be similar)
        fraud_rate_train = y_train.mean()
        fraud_rate_test = y_test.mean()
        fraud_rate_original = y.mean()
        
        assert abs(fraud_rate_train - fraud_rate_original) < 0.05
        assert abs(fraud_rate_test - fraud_rate_original) < 0.05
    
    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        preprocessor = FraudDataPreprocessor()
        
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Check shapes preserved
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        
        # Check scaling (mean ~0, std ~1 for training set)
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])