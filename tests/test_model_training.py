"""
Unit tests for model training module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.model_training import FraudDetectionModels


class TestFraudDetectionModels:
    """Test cases for FraudDetectionModels."""
    
    @pytest.fixture
    def sample_imbalanced_data(self):
        """Create imbalanced sample data for testing."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            weights=[0.9, 0.1],  # Imbalanced
            random_state=42
        )
        return X, y
    
    def test_init(self):
        """Test initialization."""
        trainer = FraudDetectionModels(random_state=42)
        assert trainer.random_state == 42
        assert trainer.models == {}
        assert trainer.results == {}
        assert trainer.best_model is None
    
    def test_train_baseline(self, sample_imbalanced_data):
        """Test baseline model training."""
        X, y = sample_imbalanced_data
        trainer = FraudDetectionModels()
        
        model = trainer.train_baseline(X, y)
        
        assert model is not None
        assert 'logistic_regression' in trainer.models
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_train_random_forest(self, sample_imbalanced_data):
        """Test Random Forest training."""
        X, y = sample_imbalanced_data
        trainer = FraudDetectionModels()
        
        model = trainer.train_random_forest(
            X, y, 
            n_estimators=50,
            max_depth=5
        )
        
        assert model is not None
        assert 'random_forest' in trainer.models
        assert model.n_estimators == 50
    
    def test_train_xgboost(self, sample_imbalanced_data):
        """Test XGBoost training."""
        X, y = sample_imbalanced_data
        trainer = FraudDetectionModels()
        
        model = trainer.train_xgboost(
            X, y,
            n_estimators=50,
            max_depth=3
        )
        
        assert model is not None
        assert 'xgboost' in trainer.models
    
    def test_evaluate_model(self, sample_imbalanced_data):
        """Test model evaluation."""
        X, y = sample_imbalanced_data
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        trainer = FraudDetectionModels()
        model = trainer.train_baseline(X_train, y_train)
        
        metrics = trainer.evaluate_model(model, X_test, y_test, "Test Model")
        
        # Check all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 
                           'roc_auc', 'pr_auc']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
        
        # Check results are stored
        assert "Test Model" in trainer.results
    
    def test_cross_validate(self, sample_imbalanced_data):
        """Test cross-validation."""
        X, y = sample_imbalanced_data
        trainer = FraudDetectionModels()
        
        model = trainer.train_baseline(X, y)
        cv_results = trainer.cross_validate(model, X, y, model_name="CV Test")
        
        # Check cross-validation results structure
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 
                           'roc_auc', 'average_precision']
        
        for metric in required_metrics:
            assert metric in cv_results
            assert 'mean' in cv_results[metric]
            assert 'std' in cv_results[metric]
            assert 'scores' in cv_results[metric]
            assert len(cv_results[metric]['scores']) == 5  # 5-fold CV
    
    def test_compare_models(self, sample_imbalanced_data):
        """Test model comparison."""
        X, y = sample_imbalanced_data
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        trainer = FraudDetectionModels()
        
        # Train multiple models
        lr_model = trainer.train_baseline(X_train, y_train)
        rf_model = trainer.train_random_forest(X_train, y_train, n_estimators=20)
        
        # Evaluate models
        trainer.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        trainer.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # Compare models
        comparison_df = trainer.compare_models()
        
        assert comparison_df is not None
        assert len(comparison_df) == 2  # Two models
        assert 'Logistic Regression' in comparison_df.index
        assert 'Random Forest' in comparison_df.index
    
    def test_select_best_model(self, sample_imbalanced_data):
        """Test model selection."""
        X, y = sample_imbalanced_data
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        trainer = FraudDetectionModels()
        
        # Train and evaluate models
        lr_model = trainer.train_baseline(X_train, y_train)
        rf_model = trainer.train_random_forest(X_train, y_train, n_estimators=20)
        
        trainer.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        trainer.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # Select best model
        best_name, best_model = trainer.select_best_model(metric='f1')
        
        assert best_name is not None
        assert best_model is not None
        assert best_name in trainer.models
        assert trainer.best_model == best_model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])