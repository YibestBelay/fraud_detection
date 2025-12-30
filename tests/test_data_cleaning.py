"""
Unit tests for data cleaning module (Task 1).
"""

import pytest
import pandas as pd
import numpy as np
from src.data_cleaning import FraudDataCleaner


class TestFraudDataCleaner:
    """Test cases for FraudDataCleaner."""
    
    @pytest.fixture
    def sample_dirty_data(self):
        """Create sample data with missing values, duplicates, and type issues."""
        data = {
            'user_id': [1, 2, 2, 3, 4, 5, 5, 6, 7, 8],
            'purchase_value': [100.0, 200.0, 200.0, np.nan, 500.0, 600.0, 600.0, 700.0, 800.0, 900.0],
            'age': ['25', '30', '30', '35', '40', '45', '45', '50', '55', '60'],
            'ip_address': ['192.168.1.1', '10.0.0.1', '10.0.0.1', '172.16.0.1', '192.168.1.2', 
                          '192.168.1.3', '192.168.1.3', '10.0.0.2', '172.16.0.2', '192.168.1.4'],
            'purchase_time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 11:00:00',
                             '2023-01-01 12:00:00', '2023-01-01 13:00:00', '2023-01-01 14:00:00',
                             '2023-01-01 14:00:00', '2023-01-01 15:00:00', '2023-01-01 16:00:00',
                             '2023-01-01 17:00:00'],
            'category': ['electronics', 'clothing', 'clothing', 'home', 'electronics', 
                        'clothing', 'clothing', 'home', 'electronics', np.nan]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def cleaner(self):
        """Create a FraudDataCleaner instance."""
        return FraudDataCleaner(verbose=False)
    
    def test_init(self, cleaner):
        """Test initialization."""
        assert cleaner.verbose == False
        assert cleaner.cleaning_report == {}
    
    def test_load_data_csv(self, cleaner, tmp_path):
        """Test loading CSV data."""
        # Create a test CSV file
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        test_file = tmp_path / "test.csv"
        test_data.to_csv(test_file, index=False)
        
        # Test loading
        df = cleaner.load_data(str(test_file))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ['col1', 'col2']
    
    def test_load_data_error(self, cleaner):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            cleaner.load_data("non_existent_file.csv")
    
    def test_analyze_missing_values(self, cleaner, sample_dirty_data):
        """Test missing value analysis."""
        missing_stats = cleaner.analyze_missing_values(sample_dirty_data)
        
        assert 'total_missing' in missing_stats
        assert 'missing_per_column' in missing_stats
        assert 'missing_percentage' in missing_stats
        assert missing_stats['total_missing'] == 2  # 1 NaN in purchase_value, 1 in category
    
    def test_handle_missing_values_median(self, cleaner, sample_dirty_data):
        """Test missing value handling with median strategy."""
        df_clean = cleaner.handle_missing_values(sample_dirty_data, strategy='median')
        
        # Check no missing values remain
        assert df_clean.isnull().sum().sum() == 0
        
        # Check median imputation
        median_value = sample_dirty_data['purchase_value'].median()
        assert df_clean.loc[3, 'purchase_value'] == median_value
        
        # Check mode imputation for categorical
        assert df_clean.loc[9, 'category'] == sample_dirty_data['category'].mode()[0]
    
    def test_handle_missing_values_drop_columns(self, cleaner):
        """Test dropping columns with high missing percentage."""
        # Create dataframe with column that has >30% missing
        data = {
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'col2': [1, 2, np.nan, np.nan, np.nan, 6, 7, 8, 9, 10]  # 30% missing
        }
        df = pd.DataFrame(data)
        
        df_clean = cleaner.handle_missing_values(df, strategy='median', threshold=0.3)
        assert 'col2' not in df_clean.columns  # Should be dropped
        assert 'col1' in df_clean.columns  # Should remain
    
    def test_remove_duplicates_without_subset(self, cleaner, sample_dirty_data):
        """Test duplicate removal without specifying subset."""
        df_clean = cleaner.remove_duplicates(sample_dirty_data)
        
        # Should remove rows 2 and 6 (duplicates of rows 1 and 5)
        assert df_clean.shape[0] == 8  # 10 - 2 duplicates
        assert df_clean.duplicated().sum() == 0
    
    def test_remove_duplicates_with_subset(self, cleaner, sample_dirty_data):
        """Test duplicate removal with specific columns."""
        df_clean = cleaner.remove_duplicates(sample_dirty_data, subset=['user_id', 'purchase_value'])
        
        # Should remove rows where user_id and purchase_value are duplicates
        assert df_clean.shape[0] == 8
        assert df_clean.duplicated(subset=['user_id', 'purchase_value']).sum() == 0
    
    def test_correct_data_types(self, cleaner, sample_dirty_data):
        """Test data type corrections."""
        df_clean = cleaner.correct_data_types(sample_dirty_data)
        
        # Check type conversions
        assert pd.api.types.is_numeric_dtype(df_clean['age'])  # Converted from string
        assert pd.api.types.is_datetime64_any_dtype(df_clean['purchase_time'])  # Converted to datetime
        assert pd.api.types.is_categorical_dtype(df_clean['category'])  # Converted to categorical
        
        # Check memory optimization
        original_memory = sample_dirty_data.memory_usage(deep=True).sum()
        optimized_memory = df_clean.memory_usage(deep=True).sum()
        assert optimized_memory <= original_memory
    
    def test_clean_fraud_data(self, cleaner, sample_dirty_data):
        """Test comprehensive cleaning."""
        df_clean = cleaner.clean_fraud_data(sample_dirty_data)
        
        # Validate cleaning
        assert df_clean.isnull().sum().sum() == 0
        assert df_clean.duplicated().sum() == 0
        assert 'cleaning_report' in cleaner.cleaning_report
    
    def test_validate_cleaning(self, cleaner, sample_dirty_data):
        """Test cleaning validation."""
        df_clean = sample_dirty_data.dropna().drop_duplicates()
        cleaner.validate_cleaning(sample_dirty_data, df_clean)
        
        # Should not raise any errors
        assert True
    
    def test_generate_cleaning_report(self, cleaner, sample_dirty_data):
        """Test cleaning report generation."""
        # Perform some cleaning
        cleaner.handle_missing_values(sample_dirty_data)
        cleaner.remove_duplicates(sample_dirty_data)
        
        report = cleaner.generate_cleaning_report()
        
        assert isinstance(report, dict)
        assert 'missing_values' in report
        assert 'duplicates' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])