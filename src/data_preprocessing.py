import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDataPreprocessor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.original_shape = self.df.shape
        
    def clean_data(self):
        """Execute complete cleaning pipeline"""
        self._convert_dtypes()
        self._handle_missing_values()
        self._remove_duplicates()
        self._validate_cleaning()
        return self.df
    
    def _convert_dtypes(self):
        """Convert columns to correct data types with justification"""
        # Time columns - critical for time-based features
        self.df['signup_time'] = pd.to_datetime(
            self.df['signup_time'], errors='coerce'
        )
        self.df['purchase_time'] = pd.to_datetime(
            self.df['purchase_time'], errors='coerce'
        )
        
        # Categorical columns
        categorical_cols = ['source', 'browser', 'sex', 'device_id']
        self.df[categorical_cols] = self.df[categorical_cols].astype('category')
        
        # Numerical columns
        self.df['purchase_value'] = pd.to_numeric(
            self.df['purchase_value'], errors='coerce'
        )
        self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
        
        print("✓ Data types converted: datetime for times, category for strings, numeric for values")
    
    def _handle_missing_values(self):
        """Impute or drop missing values with strategic justification"""
        missing_report = self.df.isnull().sum()
        print(f"Missing values before treatment:\n{missing_report[missing_report > 0]}")
        
        # Strategy 1: Drop time records if critical timestamps are missing
        time_mask = self.df[['signup_time', 'purchase_time']].isnull().any(axis=1)
        self.df = self.df[~time_mask].copy()
        
        # Strategy 2: Impute numerical with median (robust to outliers)
        self.df['purchase_value'].fillna(
            self.df['purchase_value'].median(), inplace=True
        )
        
        # Strategy 3: Impute age with mode (most common age group)
        self.df['age'].fillna(self.df['age'].mode()[0], inplace=True)
        
        # Strategy 4: For categorical, use 'Unknown' category
        self.df['source'].fillna('Unknown', inplace=True)
        self.df['browser'].fillna('Unknown', inplace=True)
        
        print("✓ Missing values handled: times dropped, numerics imputed, categoricals labeled")
    
    def _remove_duplicates(self):
        """Remove exact and near-duplicates"""
        initial_count = len(self.df)
        
        # Remove exact duplicates across all columns
        self.df.drop_duplicates(inplace=True)
        
        # Remove suspicious transaction duplicates
        # Same user, same device, same value within 1 minute
        self.df.sort_values(['user_id', 'purchase_time'], inplace=True)
        duplicate_mask = (
            self.df.duplicated(
                subset=['user_id', 'device_id', 'purchase_value'], keep='first'
            ) & 
            (self.df['purchase_time'].diff().dt.total_seconds().abs() < 60)
        )
        self.df = self.df[~duplicate_mask].reset_index(drop=True)
        
        removed = initial_count - len(self.df)
        print(f"✓ Duplicates removed: {removed} records ({removed/initial_count:.1%})")
    
    def _validate_cleaning(self):
        """Validate cleaning operations"""
        print("\n" + "="*50)
        print("CLEANING VALIDATION REPORT")
        print("="*50)
        print(f"Original shape: {self.original_shape}")
        print(f"Final shape: {self.df.shape}")
        print(f"Records removed: {self.original_shape[0] - self.df.shape[0]}")
        print(f"Missing values remaining: {self.df.isnull().sum().sum()}")
        print(f"Duplicate rows remaining: {self.df.duplicated().sum()}")
        print("="*50)