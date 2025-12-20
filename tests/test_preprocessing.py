"""
Unit tests for data preprocessing.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""
    
    def setup_method(self):
        self.processor = DataPreprocessor()
        self.sample_data = pd.DataFrame({
            'userid': [1, 2, 2, 3],
            'signuptime': ['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03'],
            'purchasetime': ['2023-01-01 10:00', '2023-01-02 12:00', 
                            '2023-01-02 12:00', '2023-01-03 14:00'],
            'purchasevalue': [100, 200, 200, 300],
            'class': [0, 1, 1, 0]
        })
    
    def test_clean_data_removes_duplicates(self):
        """Test duplicate removal."""
        cleaned = self.processor.clean_data(self.sample_data)
        assert len(cleaned) == 3  # One duplicate removed
    
    def test_engineer_features_creates_new_columns(self):
        """Test feature engineering adds new columns."""
        cleaned = self.processor.clean_data(self.sample_data)
        engineered = self.processor.engineer_features(cleaned)
        assert 'hour_of_day' in engineered.columns
        assert 'time_since_signup' in engineered.columns