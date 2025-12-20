import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDataPreprocessor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.original_shape = self.df.shape
        
        # Standardize critical ID columns as strings immediately
        id_cols = ['user_id', 'device_id']
        for col in id_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
        
    def clean_data(self):
        """Execute complete cleaning pipeline"""
        self._convert_dtypes()
        self._handle_missing_values()
        self._remove_duplicates()
        self._validate_cleaning()
        return self.df
    
    def _convert_dtypes(self):
        """Convert columns to correct data types"""
        # Time columns
        self.df['signup_time'] = pd.to_datetime(self.df['signup_time'], errors='coerce')
        self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'], errors='coerce')
        
        # Numerical
        self.df['purchase_value'] = pd.to_numeric(self.df['purchase_value'], errors='coerce')
        self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
        
        # Categorical-like (as string, not category)
        for col in ['source', 'browser', 'sex', 'ip_address']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
        
        print("✓ Data types converted: datetime, numeric, and string (not categorical)")
    
    def _handle_missing_values(self):
        """Handle missing values"""
        missing_report = self.df.isnull().sum()
        if missing_report.sum() > 0:
            print(f"Missing values before treatment:\n{missing_report[missing_report > 0]}")
        
        # Drop rows with missing critical timestamps
        time_mask = self.df[['signup_time', 'purchase_time']].isnull().any(axis=1)
        self.df = self.df[~time_mask].copy()
        
        # Numerical imputation
        self.df['purchase_value'].fillna(self.df['purchase_value'].median(), inplace=True)
        self.df['age'].fillna(self.df['age'].mode()[0], inplace=True)
        
        # Categorical: replace 'nan' string and fill
        for col in ['source', 'browser', 'sex', 'ip_address']:
            if col in self.df.columns:
                self.df[col].replace('nan', 'Unknown', inplace=True)
                self.df[col].fillna('Unknown', inplace=True)
        
        print("✓ Missing values handled")
    
    def _remove_duplicates(self):
        """Remove duplicates"""
        initial = len(self.df)
        self.df.drop_duplicates(inplace=True)
        
        # Near-duplicates: same user/device/value within 60 sec
        self.df.sort_values(['user_id', 'purchase_time'], inplace=True, ignore_index=True)
        time_diff = self.df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
        near_dup = (
            self.df.duplicated(subset=['user_id', 'device_id', 'purchase_value'], keep='first') &
            (time_diff < 60)
        )
        self.df = self.df[~near_dup].reset_index(drop=True)
        
        removed = initial - len(self.df)
        print(f"✓ Duplicates removed: {removed} records ({removed/initial:.1%})")
    
    def _validate_cleaning(self):
        """Validation report"""
        print("\n" + "="*50)
        print("CLEANING VALIDATION REPORT")
        print("="*50)
        print(f"Original shape: {self.original_shape}")
        print(f"Final shape: {self.df.shape}")
        print(f"Records removed: {self.original_shape[0] - self.df.shape[0]}")
        print(f"Missing values remaining: {self.df.isnull().sum().sum()}")
        print(f"Duplicate rows remaining: {self.df.duplicated().sum()}")
        print("="*50)