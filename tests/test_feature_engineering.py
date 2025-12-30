"""
Unit tests for feature engineering module (Task 1).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering import FraudFeatureEngineer


class TestFraudFeatureEngineer:
    """Test cases for FraudFeatureEngineer."""
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for feature engineering."""
        np.random.seed(42)
        n_samples = 100
        
        # Generate timestamps
        base_time = datetime(2023, 1, 1, 10, 0, 0)
        purchase_times = [base_time + timedelta(hours=i) for i in range(n_samples)]
        signup_times = [t - timedelta(days=np.random.randint(1, 30)) for t in purchase_times]
        
        data = {
            'user_id': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'device_id': np.random.choice(['d1', 'd2', 'd3', 'd4', 'd5'], n_samples),
            'purchase_time': purchase_times,
            'signup_time': signup_times,
            'purchase_value': np.random.exponential(100, n_samples),
            'ip_address': ['192.168.1.' + str(i%255) for i in range(n_samples)],
            'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], n_samples),
            'class': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create a FraudFeatureEngineer instance."""
        return FraudFeatureEngineer(verbose=False)
    
    def test_init(self, feature_engineer):
        """Test initialization."""
        assert feature_engineer.verbose == False
        assert feature_engineer.feature_stats == {}
        assert feature_engineer.engineered_features == []
    
    def test_create_time_based_features(self, feature_engineer, sample_transaction_data):
        """Test time-based feature creation."""
        df_engineered = feature_engineer.create_time_based_features(
            sample_transaction_data,
            purchase_time_col='purchase_time',
            signup_time_col='signup_time'
        )
        
        # Check new features created
        assert 'purchase_hour' in df_engineered.columns
        assert 'purchase_dayofweek' in df_engineered.columns
        assert 'purchase_dayofmonth' in df_engineered.columns
        assert 'purchase_month' in df_engineered.columns
        assert 'is_weekend' in df_engineered.columns
        assert 'is_business_hours' in df_engineered.columns
        assert 'time_since_signup_days' in df_engineered.columns
        assert 'time_since_signup_hours' in df_engineered.columns
        assert 'is_new_user' in df_engineered.columns
        assert 'purchase_hour_sin' in df_engineered.columns
        assert 'purchase_hour_cos' in df_engineered.columns
        
        # Check feature values
        assert df_engineered['purchase_hour'].min() >= 0
        assert df_engineered['purchase_hour'].max() <= 23
        assert df_engineered['is_weekend'].isin([0, 1]).all()
        assert df_engineered['is_business_hours'].isin([0, 1]).all()
        
        # Check time since signup
        assert (df_engineered['time_since_signup_hours'] >= 0).all()
    
    def test_create_time_based_features_no_signup(self, feature_engineer, sample_transaction_data):
        """Test time-based features without signup time."""
        df_no_signup = sample_transaction_data.drop(columns=['signup_time'])
        df_engineered = feature_engineer.create_time_based_features(
            df_no_signup,
            purchase_time_col='purchase_time'
        )
        
        # Should still create time-based features
        assert 'purchase_hour' in df_engineered.columns
        assert 'purchase_dayofweek' in df_engineered.columns
        
        # Should not create signup-related features
        assert 'time_since_signup_days' not in df_engineered.columns
        assert 'is_new_user' not in df_engineered.columns
    
    def test_create_frequency_features(self, feature_engineer, sample_transaction_data):
        """Test frequency and velocity feature creation."""
        # First add time-based features
        df_with_time = feature_engineer.create_time_based_features(
            sample_transaction_data,
            purchase_time_col='purchase_time'
        )
        
        df_engineered = feature_engineer.create_frequency_features(
            df_with_time,
            user_id_col='user_id',
            device_id_col='device_id',
            purchase_time_col='purchase_time',
            window_hours=[1, 24]  # Test with smaller windows
        )
        
        # Check frequency features
        assert 'user_total_transactions' in df_engineered.columns
        assert 'device_total_transactions' in df_engineered.columns
        assert 'unique_users_per_device' in df_engineered.columns
        assert 'transactions_last_1h' in df_engineered.columns
        assert 'transactions_last_24h' in df_engineered.columns
        assert 'hours_since_first_transaction' in df_engineered.columns
        assert 'transactions_per_hour' in df_engineered.columns
        assert 'has_rapid_transactions' in df_engineered.columns
        
        # Check values
        assert (df_engineered['user_total_transactions'] > 0).all()
        assert (df_engineered['transactions_last_1h'] >= 0).all()
        assert (df_engineered['transactions_last_24h'] >= 0).all()
        assert df_engineered['has_rapid_transactions'].isin([0, 1]).all()
    
    def test_create_aggregate_features(self, feature_engineer, sample_transaction_data):
        """Test aggregate feature creation."""
        df_engineered = feature_engineer.create_aggregate_features(
            sample_transaction_data,
            group_cols=['user_id', 'country'],
            value_cols=['purchase_value'],
            functions=['mean', 'std', 'min', 'max']
        )
        
        # Check aggregate features created
        assert 'purchase_value_mean_by_user_id' in df_engineered.columns
        assert 'purchase_value_std_by_user_id' in df_engineered.columns
        assert 'purchase_value_min_by_user_id' in df_engineered.columns
        assert 'purchase_value_max_by_user_id' in df_engineered.columns
        assert 'purchase_value_mean_by_country' in df_engineered.columns
        assert 'purchase_value_deviation_from_user_id_mean' in df_engineered.columns
        
        # Check values
        assert not df_engineered['purchase_value_mean_by_user_id'].isnull().any()
    
    def test_create_interaction_features(self, feature_engineer, sample_transaction_data):
        """Test interaction feature creation."""
        # Add some numerical features first
        df_with_numerical = sample_transaction_data.copy()
        df_with_numerical['age'] = np.random.randint(18, 70, len(df_with_numerical))
        df_with_numerical['session_duration'] = np.random.exponential(300, len(df_with_numerical))
        
        df_engineered = feature_engineer.create_interaction_features(
            df_with_numerical,
            feature_pairs=[('purchase_value', 'age'), ('purchase_value', 'session_duration')]
        )
        
        # Check interaction features created
        assert 'purchase_value_x_age' in df_engineered.columns
        assert 'purchase_value_div_age' in df_engineered.columns
        assert 'purchase_value_x_session_duration' in df_engineered.columns
        assert 'purchase_value_div_session_duration' in df_engineered.columns
        
        # Check interaction calculations
        for idx, row in df_engineered.iterrows():
            assert row['purchase_value_x_age'] == row['purchase_value'] * row['age']
            if row['age'] != 0:
                assert row['purchase_value_div_age'] == row['purchase_value'] / row['age']
    
    def test_create_geolocation_features(self, feature_engineer, sample_transaction_data):
        """Test geolocation feature creation."""
        df_engineered = feature_engineer.create_geolocation_features(
            sample_transaction_data,
            ip_col='ip_address',
            country_col='country'
        )
        
        # Check IP features
        assert 'ip_octet_1' in df_engineered.columns
        assert 'ip_octet_2' in df_engineered.columns
        assert 'ip_octet_3' in df_engineered.columns
        assert 'ip_octet_4' in df_engineered.columns
        assert 'is_private_ip' in df_engineered.columns
        
        # Check country features
        assert 'country_transaction_count' in df_engineered.columns
        if 'class' in df_engineered.columns:
            assert 'country_fraud_rate' in df_engineered.columns
            assert 'is_high_risk_country' in df_engineered.columns
        
        # Check IP parsing
        assert (df_engineered['ip_octet_1'] >= 0).all()
        assert (df_engineered['ip_octet_1'] <= 255).all()
        assert df_engineered['is_private_ip'].isin([0, 1]).all()
    
    def test_create_all_features(self, feature_engineer, sample_transaction_data):
        """Test complete feature engineering pipeline."""
        df_engineered = feature_engineer.create_all_features(
            sample_transaction_data,
            target_col='class'
        )
        
        # Check that features were added
        original_cols = set(sample_transaction_data.columns)
        engineered_cols = set(df_engineered.columns)
        added_features = engineered_cols - original_cols
        
        assert len(added_features) > 0
        assert len(feature_engineer.engineered_features) == len(added_features)
        
        # Check feature stats
        assert 'initial_features' in feature_engineer.feature_stats
        assert 'engineered_features' in feature_engineer.feature_stats
        assert 'total_features' in feature_engineer.feature_stats
        assert 'feature_categories' in feature_engineer.feature_stats
        
        assert feature_engineer.feature_stats['initial_features'] == len(sample_transaction_data.columns)
        assert feature_engineer.feature_stats['total_features'] == len(df_engineered.columns)
        assert feature_engineer.feature_stats['engineered_features'] == len(added_features)
    
    def test_get_feature_importance_report(self, feature_engineer, sample_transaction_data):
        """Test feature importance report generation."""
        # First create features
        df_engineered = feature_engineer.create_all_features(
            sample_transaction_data,
            target_col='class'
        )
        
        # Get importance report
        importance_df = feature_engineer.get_feature_importance_report(
            df_engineered,
            target_col='class',
            top_n=10
        )
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'correlation_abs' in importance_df.columns
        assert 'correlation' in importance_df.columns
        assert 'category' in importance_df.columns
        
        assert len(importance_df) <= 10
        assert (importance_df['correlation_abs'] >= 0).all()
        assert (importance_df['correlation_abs'] <= 1).all()
        
        # Check categories
        valid_categories = ['Time-based', 'Frequency', 'Aggregate', 'Interaction', 
                          'Geolocation', 'Original']
        assert importance_df['category'].isin(valid_categories).all()
    
    def test_get_feature_importance_report_no_target(self, feature_engineer, sample_transaction_data):
        """Test feature importance report without target column."""
        df_engineered = feature_engineer.create_all_features(
            sample_transaction_data,
            target_col='class'
        )
        
        df_no_target = df_engineered.drop(columns=['class'])
        
        with pytest.raises(ValueError):
            feature_engineer.get_feature_importance_report(
                df_no_target,
                target_col='class',
                top_n=10
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])