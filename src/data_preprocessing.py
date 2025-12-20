"""
Data preprocessing module for fraud detection.
Handles cleaning, feature engineering, and transformation.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import ipaddress
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess fraud detection data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.categorical_mappings = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform data cleaning operations."""
        logger.info(f"Cleaning data with shape: {df.shape}")
        
        # Convert datetimes
        df['signuptime'] = pd.to_datetime(df['signuptime'])
        df['purchasetime'] = pd.to_datetime(df['purchasetime'])
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Correct data types
        df = self._correct_data_types(df)
        
        logger.info(f"Cleaned data shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strategy for handling missing values."""
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        return df
    
    def _correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types."""
        type_mapping = {
            'userid': 'str',
            'deviceid': 'str',
            'ipaddress': 'str'
        }
        
        for col, dtype in type_mapping.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for fraud detection."""
        logger.info("Engineering features...")
        
        # Time-based features
        df['hour_of_day'] = df['purchasetime'].dt.hour
        df['day_of_week'] = df['purchasetime'].dt.dayofweek
        
        # Time since signup
        df['time_since_signup'] = (df['purchasetime'] - df['signuptime']).dt.total_seconds() / 3600
        
        # Sort for rolling features
        df = df.sort_values(['userid', 'purchasetime'])
        
        # Transaction frequency
        df['txn_count_24h'] = df.groupby('userid')['purchasetime'].transform(
            lambda x: x.rolling('24h', on=x).count()
        )
        
        logger.info(f"Added {len([c for c in df.columns if c not in ['userid', 'signuptime', 'purchasetime']])} new features")
        return df