"""
Ruthlessly efficient data cleaning for fraud detection.
No mercy for dirty data.
"""
import pandas as pd
import numpy as np
from typing import Tuple
import ipaddress

class FraudDataCleaner:
    """Clean fraud data with military precision."""
    
    def __init__(self, fraud_path: str, ip_country_path: str):
        self.fraud_path = fraud_path
        self.ip_country_path = ip_country_path
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data with proper data types."""
        # Load fraud data
        dtype_map = {
            'user_id': 'str',
            'device_id': 'str',
            'source': 'category',
            'browser': 'category',
            'sex': 'category',
            'ip_address': 'str',
            'class': 'int8'
        }
        
        date_cols = ['signup_time', 'purchase_time']
        fraud_df = pd.read_csv(
            self.fraud_path, 
            dtype=dtype_map,
            parse_dates=date_cols,
            infer_datetime_format=True
        )
        
        # Load IP-country mapping
        ip_df = pd.read_csv(
            self.ip_country_path,
            dtype={
                'lower_bound_ip_address': 'str',
                'upper_bound_ip_address': 'str',
                'country': 'category'
            }
        )
        
        return fraud_df, ip_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ruthlessly eliminate or impute missing values with justification."""
        df_clean = df.copy()
        
        # Document missing values BEFORE cleaning
        missing_report = df_clean.isnull().sum()
        missing_percentage = (missing_report / len(df_clean)) * 100
        
        print("=== MISSING VALUES REPORT ===")
        for col in missing_report.index:
            if missing_report[col] > 0:
                print(f"{col}: {missing_report[col]} missing ({missing_percentage[col]:.2f}%)")
        
        # Strategy per column (JUSTIFICATION REQUIRED)
        missing_strategy = {
            'age': 'median',  # Age has reasonable distribution for median
            'browser': 'mode',  # Categorical, use most frequent
            'device_id': 'unknown',  # String column
            'source': 'mode',
            'sex': 'mode'
        }
        
        for col, strategy in missing_strategy.items():
            if col in df_clean.columns and df_clean[col].isnull().any():
                if strategy == 'median':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    print(f"IMPUTED {col}: {strategy} (robust to outliers)")
                elif strategy == 'mode':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                    print(f"IMPUTED {col}: {strategy} (most common value)")
                elif strategy == 'unknown':
                    df_clean[col] = df_clean[col].fillna('unknown')
                    print(f"IMPUTED {col}: '{strategy}' (explicit missing category)")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Eliminate duplicate records with prejudice."""
        initial_rows = len(df)
        df_deduped = df.drop_duplicates(
            subset=['user_id', 'purchase_time', 'purchase_value'],
            keep='first'
        )
        duplicates_removed = initial_rows - len(df_deduped)
        
        print(f"\n=== DUPLICATE REMOVAL ===")
        print(f"Initial rows: {initial_rows}")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Final rows: {len(df_deduped)}")
        print(f"Percentage removed: {duplicates_removed/initial_rows*100:.2f}%")
        
        return df_deduped
    
    def correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce proper data types for memory efficiency."""
        df_corrected = df.copy()
        
        # Convert to optimal types
        type_conversions = {
            'user_id': 'str',
            'device_id': 'str',
            'source': 'category',
            'browser': 'category',
            'sex': 'category',
            'age': 'int16',
            'purchase_value': 'float32',
            'class': 'int8'
        }
        
        for col, dtype in type_conversions.items():
            if col in df_corrected.columns:
                try:
                    df_corrected[col] = df_corrected[col].astype(dtype)
                    print(f"CONVERTED {col} -> {dtype}")
                except Exception as e:
                    print(f"FAILED {col} conversion: {e}")
        
        # Memory usage comparison
        print(f"\n=== MEMORY OPTIMIZATION ===")
        print(f"Original memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Optimized memory: {df_corrected.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Reduction: {(1 - df_corrected.memory_usage(deep=True).sum()/df.memory_usage(deep=True).sum())*100:.1f}%")
        
        return df_corrected
    
    def clean_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute full cleaning pipeline."""
        print("=== STARTING DATA CLEANING PIPELINE ===")
        fraud_df, ip_df = self.load_data()
        
        print("\n1. Handling missing values...")
        fraud_df = self.handle_missing_values(fraud_df)
        
        print("\n2. Removing duplicates...")
        fraud_df = self.remove_duplicates(fraud_df)
        
        print("\n3. Correcting data types...")
        fraud_df = self.correct_data_types(fraud_df)
        
        return fraud_df, ip_df