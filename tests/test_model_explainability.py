"""
Unit tests for model explainability module (Task 3).
"""

import pytest
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Mock SHAP since it's heavy to import in tests
class MockExplainer:
    def __init__(self, expected_value=0.1):
        self.expected_value = expected_value
    
    def shap_values(self, X):
        return np.random.randn(len(X), X.shape[1])


class MockModel:
    def __init__(self, n_features):
        self.n_features = n_features
        self.feature_importances_ = np.random.rand(n_features)
    
    def predict(self, X):
        return np.random.randint(0, 2, len(X))
    
    def predict_proba(self, X):
        probs = np.random.rand(len(X), 2)
        return probs / probs.sum(axis=1, keepdims=True)


# Import after mocks
from src.model_explainability import FraudModelExplainer


class TestFraudModelExplainer:
    """Test cases for FraudModelExplainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for explainability testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        return X, y
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MockModel(n_features=10)
    
    @pytest.fixture
    def explainer(self, mock_model):
        """Create a FraudModelExplainer instance."""
        feature_names = [f'feature_{i}' for i in range(10)]
        return FraudModelExplainer(
            model=mock_model,
            feature_names=feature_names,
            random_state=42
        )
    
    def test_init(self, mock_model):
        """Test initialization."""
        feature_names = ['feat1', 'feat2', 'feat3']
        explainer = FraudModelExplainer(mock_model, feature_names)
        
        assert explainer.model == mock_model
        assert explainer.feature_names == feature_names
        assert explainer.explainer is None
        assert explainer.shap_values is None
        assert explainer.insights == {}
    
    def test_extract_builtin_feature_importance_tree(self):
        """Test built-in feature importance for tree-based models."""
        # Create a mock tree model
        class MockTreeModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4])
        
        model = MockTreeModel()
        feature_names = ['feat1', 'feat2', 'feat3', 'feat4']
        explainer = FraudModelExplainer(model, feature_names)
        
        importance_df = explainer.extract_builtin_feature_importance(top_n=3)
        
        assert importance_df is not None
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == 4  # All features
        assert importance_df['importance'].sum() > 0
        
        # Check sorting
        assert importance_df['importance'].iloc[0] >= importance_df['importance'].iloc[1]
    
    def test_extract_builtin_feature_importance_linear(self):
        """Test built-in feature importance for linear models."""
        # Create a mock linear model
        class MockLinearModel:
            def __init__(self):
                self.coef_ = np.array([[0.5, -0.3, 0.2, -0.4]])
        
        model = MockLinearModel()
        feature_names = ['feat1', 'feat2', 'feat3', 'feat4']
        explainer = FraudModelExplainer(model, feature_names)
        
        importance_df = explainer.extract_builtin_feature_importance(top_n=2)
        
        assert importance_df is not None
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 4
        assert (importance_df['importance'] >= 0).all()  # Absolute values
    
    def test_extract_builtin_feature_importance_no_method(self):
        """Test built-in feature importance for models without importance."""
        # Create a mock model without feature_importances_ or coef_
        class MockNoImportanceModel:
            pass
        
        model = MockNoImportanceModel()
        feature_names = ['feat1', 'feat2']
        explainer = FraudModelExplainer(model, feature_names)
        
        importance_df = explainer.extract_builtin_feature_importance()
        
        assert importance_df is None
    
    @pytest.mark.skip(reason="SHAP computation is heavy for tests")
    def test_compute_shap_values(self, explainer, sample_data):
        """Test SHAP value computation."""
        X, _ = sample_data
        
        explainer.compute_shap_values(X, sample_size=50)
        
        assert explainer.explainer is not None
        assert explainer.shap_values is not None
        assert explainer.shap_values.shape == (50, X.shape[1])
    
    @pytest.mark.skip(reason="SHAP computation is heavy for tests")
    def test_compute_shap_values_small_sample(self, explainer, sample_data):
        """Test SHAP value computation with small sample."""
        X, _ = sample_data
        
        # Use sample smaller than dataset
        explainer.compute_shap_values(X, sample_size=30)
        
        assert explainer.shap_values.shape[0] == 30
    
    def test_plot_shap_summary_no_values(self, explainer, sample_data):
        """Test SHAP summary plot without computed values."""
        X, _ = sample_data
        
        # Should print warning but not crash
        explainer.plot_shap_summary(X, max_display=10)
        assert True
    
    def test_plot_shap_force_plots_no_values(self, explainer, sample_data):
        """Test SHAP force plots without computed values."""
        X, y = sample_data
        
        # Should print warning but not crash
        explainer.plot_shap_force_plots(X, y, n_cases=2)
        assert True
    
    def test_find_interesting_cases(self, explainer, sample_data):
        """Test finding interesting cases for visualization."""
        X, y = sample_data
        
        # Create mock predictions
        y_pred = np.random.randint(0, 2, len(y))
        y_pred_proba = np.random.rand(len(y))
        
        cases = explainer._find_interesting_cases(X, y, y_pred, y_pred_proba, n_cases=3)
        
        assert isinstance(cases, list)
        assert len(cases) <= 3
        
        for case in cases:
            assert 'type' in case
            assert 'index' in case
            assert 'actual' in case
            assert 'pred' in case
            assert 'pred_prob' in case
            
            assert case['type'] in ['true_positive', 'false_positive', 
                                   'false_negative', 'true_negative', 'borderline']
            assert 0 <= case['pred_prob'] <= 1
    
    def test_find_interesting_cases_no_tp(self):
        """Test finding cases when no true positives exist."""
        # Create data with no true positives
        X = pd.DataFrame({'feat': [1, 2, 3, 4]})
        y = pd.Series([0, 0, 0, 0])  # All negative
        y_pred = pd.Series([1, 1, 1, 1])  # All predicted positive
        y_pred_proba = pd.Series([0.6, 0.7, 0.8, 0.9])
        
        explainer = FraudModelExplainer(MockModel(1), ['feat'])
        cases = explainer._find_interesting_cases(X, y, y_pred, y_pred_proba, n_cases=2)
        
        # Should still return cases (FP, FN, or TN)
        assert isinstance(cases, list)
    
    def test_compare_feature_importance_no_shap(self, explainer):
        """Test feature importance comparison without SHAP values."""
        # Create mock built-in importance
        builtin_importance = pd.DataFrame({
            'feature': ['feat1', 'feat2', 'feat3'],
            'importance': [0.5, 0.3, 0.2]
        })
        
        comparison = explainer.compare_feature_importance(
            builtin_importance=builtin_importance,
            top_n=3
        )
        
        assert comparison is None
    
    def test_generate_business_recommendations_no_shap(self, explainer, sample_data):
        """Test business recommendations without SHAP values."""
        X, y = sample_data
        
        recommendations = explainer.generate_business_recommendations(X, y, threshold=0.5)
        
        assert recommendations == {}  # Empty dict when no SHAP values
    
    def test_generate_explainability_report(self, explainer):
        """Test explainability report generation."""
        report = explainer.generate_explainability_report()
        
        assert isinstance(report, dict)
        assert 'insights' in report
        assert 'summary' in report
        
        summary = report['summary']
        assert 'shap_values_computed' in summary
        assert 'explainer_available' in summary
        assert 'total_features' in summary
        assert 'analysis_completed' in summary
    
    def test_generate_explainability_report_with_insights(self, explainer):
        """Test report generation with some insights."""
        # Add some insights
        explainer.insights = {
            'builtin_importance': {
                'type': 'Gini Importance',
                'top_features': [{'feature': 'feat1', 'importance': 0.5}]
            }
        }
        
        report = explainer.generate_explainability_report()
        
        assert report['summary']['analysis_completed'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])