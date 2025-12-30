"""
Unit tests for EDA module (Task 1).
"""

import pytest
import pandas as pd
import numpy as np
from src.eda import FraudEDA
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class TestFraudEDA:
    """Test cases for FraudEDA."""
    
    @pytest.fixture
    def sample_fraud_data(self):
        """Create sample fraud detection data."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'user_id': np.random.randint(1, 100, n_samples),
            'purchase_value': np.random.exponential(100, n_samples),
            'age': np.random.randint(18, 70, n_samples),
            'purchase_hour': np.random.randint(0, 24, n_samples),
            'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], n_samples),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
            'class': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])  # Imbalanced
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def eda(self):
        """Create a FraudEDA instance."""
        return FraudEDA(figsize=(10, 6))
    
    def test_init(self, eda):
        """Test initialization."""
        assert eda.figsize == (10, 6)
        assert eda.insights == {}
    
    def test_analyze_class_distribution(self, eda, sample_fraud_data):
        """Test class distribution analysis."""
        stats = eda.analyze_class_distribution(sample_fraud_data, target_col='class')
        
        assert 'class_counts' in stats
        assert 'class_percentages' in stats
        assert 'total_samples' in stats
        assert 'imbalance_ratio' in stats
        
        assert stats['total_samples'] == len(sample_fraud_data)
        assert 0 <= stats['imbalance_ratio'] <= 100
    
    def test_analyze_class_distribution_error(self, eda, sample_fraud_data):
        """Test class distribution with invalid target column."""
        with pytest.raises(ValueError):
            eda.analyze_class_distribution(sample_fraud_data, target_col='nonexistent')
    
    def test_univariate_analysis_numerical(self, eda, sample_fraud_data):
        """Test univariate analysis for numerical features."""
        numerical_cols = ['purchase_value', 'age']
        
        eda.univariate_analysis(
            sample_fraud_data,
            numerical_cols=numerical_cols,
            categorical_cols=[]
        )
        
        assert 'numerical_stats' in eda.insights
        stats = eda.insights['numerical_stats']
        
        # Check statistics are calculated
        for col in numerical_cols:
            assert col in stats.index
            assert 'mean' in stats.columns
            assert 'std' in stats.columns
            assert 'skewness' in stats.columns
    
    def test_univariate_analysis_categorical(self, eda, sample_fraud_data):
        """Test univariate analysis for categorical features."""
        categorical_cols = ['country', 'device_type']
        
        eda.univariate_analysis(
            sample_fraud_data,
            numerical_cols=[],
            categorical_cols=categorical_cols
        )
        
        assert 'categorical_stats' in eda.insights
        stats = eda.insights['categorical_stats']
        
        for col in categorical_cols:
            assert col in stats.index
            assert 'unique_values' in stats.columns
            assert 'top_category' in stats.columns
    
    def test_bivariate_analysis_numerical(self, eda, sample_fraud_data):
        """Test bivariate analysis for numerical features."""
        numerical_cols = ['purchase_value', 'age']
        
        eda.bivariate_analysis(
            sample_fraud_data,
            target_col='class',
            numerical_cols=numerical_cols,
            categorical_cols=[]
        )
        
        assert 'numerical_correlations' in eda.insights
        correlations = eda.insights['numerical_correlations']
        
        assert isinstance(correlations, pd.DataFrame)
        assert 'correlation' in correlations.columns
    
    def test_bivariate_analysis_categorical(self, eda, sample_fraud_data):
        """Test bivariate analysis for categorical features."""
        categorical_cols = ['country', 'device_type']
        
        eda.bivariate_analysis(
            sample_fraud_data,
            target_col='class',
            numerical_cols=[],
            categorical_cols=categorical_cols
        )
        
        # Should not raise errors
        assert True
    
    def test_bivariate_analysis_error(self, eda, sample_fraud_data):
        """Test bivariate analysis with invalid target column."""
        with pytest.raises(ValueError):
            eda.bivariate_analysis(
                sample_fraud_data,
                target_col='nonexistent',
                numerical_cols=['purchase_value'],
                categorical_cols=[]
            )
    
    def test_correlation_analysis(self, eda, sample_fraud_data):
        """Test correlation analysis."""
        numerical_cols = ['purchase_value', 'age', 'purchase_hour']
        
        eda.correlation_analysis(
            sample_fraud_data[numerical_cols]
        )
        
        assert 'correlation_matrix' in eda.insights
        assert 'high_correlation_pairs' in eda.insights
        
        corr_matrix = eda.insights['correlation_matrix']
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (len(numerical_cols), len(numerical_cols))
    
    def test_correlation_analysis_insufficient_columns(self, eda, sample_fraud_data):
        """Test correlation analysis with insufficient numerical columns."""
        # Single column should not fail
        eda.correlation_analysis(sample_fraud_data[['purchase_value']])
        assert True
    
    def test_outlier_analysis(self, eda, sample_fraud_data):
        """Test outlier analysis."""
        numerical_cols = ['purchase_value', 'age']
        
        eda.outlier_analysis(
            sample_fraud_data,
            numerical_cols=numerical_cols,
            threshold=3.0
        )
        
        assert 'outlier_stats' in eda.insights
        outlier_stats = eda.insights['outlier_stats']
        
        assert isinstance(outlier_stats, dict)
    
    def test_outlier_analysis_no_outliers(self, eda):
        """Test outlier analysis on data with no outliers."""
        # Create data with no outliers
        data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100),
            'col2': np.random.normal(10, 2, 100)
        })
        
        eda.outlier_analysis(data, threshold=5.0)  # High threshold
        assert True  # Should not raise errors
    
    def test_generate_eda_report(self, eda, sample_fraud_data):
        """Test EDA report generation."""
        # Run some analyses
        eda.analyze_class_distribution(sample_fraud_data, 'class')
        eda.univariate_analysis(sample_fraud_data)
        eda.bivariate_analysis(sample_fraud_data, 'class')
        
        report = eda.generate_eda_report()
        
        assert isinstance(report, dict)
        assert 'insights' in report
        assert 'summary' in report
        
        summary = report['summary']
        assert 'total_features_analyzed' in summary
        assert 'imbalance_ratio' in summary
        assert 'fraud_rate' in summary
    
    def test_generate_eda_report_empty(self, eda):
        """Test EDA report generation without any analysis."""
        report = eda.generate_eda_report()
        
        assert isinstance(report, dict)
        assert 'insights' in report
        assert 'summary' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])